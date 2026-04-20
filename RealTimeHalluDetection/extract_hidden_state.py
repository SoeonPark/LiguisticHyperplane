"""
Utilities for precise answer-token alignment and hidden-state extraction.

The important invariant is:
  prompt tokens + answer tokens must be recoverable as an exact token span in
  the teacher-forced input sequence.

This module exposes the same pooled strategies used by the existing probe
pipeline:
  - prompt_last: hidden state at the final prompt token
  - first: first answer token
  - mean: mean over answer tokens
  - last: last answer token
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

import config

HIDDEN_STATE_CACHE_VERSION = 3
TOKENWISE_CACHE_VERSION = 2


@dataclass
class AnswerSpanAlignment:
    prompt: str
    answer: str
    full_text: str
    boundary: str
    input_len: int
    prompt_len: int
    full_len: int
    answer_start: int
    answer_end: int
    answer_len: int
    answer_token_ids: List[int]
    answer_tokens: List[str]
    decoded_answer_span: str

    def to_json_dict(self) -> Dict:
        payload = asdict(self)
        payload["boundary"] = repr(self.boundary)
        return payload


def build_full_text(prompt: str, answer: str) -> Tuple[str, str]:
    answer = answer.strip()
    if not answer:
        return prompt, ""
    return prompt + " " + answer, answer


def build_full_text_candidates(
    prompt: str,
    answer: str,
    boundaries: Sequence[str] = (" ", "", "\n", "\n "),
) -> List[Tuple[str, str]]:
    answer = answer.strip()
    if not answer:
        return []

    candidates: List[Tuple[str, str]] = []
    seen = set()
    for boundary in boundaries:
        full_text = prompt + boundary + answer
        if full_text not in seen:
            seen.add(full_text)
            candidates.append((boundary, full_text))
    return candidates


def _as_list(input_ids) -> List[int]:
    if isinstance(input_ids, torch.Tensor):
        return input_ids.detach().cpu().view(-1).tolist()
    if hasattr(input_ids, "tolist"):
        return list(input_ids.tolist())
    return list(input_ids)


def token_ids_no_special(tokenizer, text: str) -> List[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    return _as_list(encoded.input_ids)


def _find_subsequence(sequence: Sequence[int], pattern: Sequence[int]) -> int:
    if not pattern or len(pattern) > len(sequence):
        return -1
    last_start = len(sequence) - len(pattern)
    for start in range(last_start + 1):
        if list(sequence[start:start + len(pattern)]) == list(pattern):
            return start
    return -1


def _tokens_from_ids(tokenizer, token_ids: Sequence[int]) -> List[str]:
    try:
        return tokenizer.convert_ids_to_tokens(list(token_ids))
    except Exception:
        return [str(x) for x in token_ids]


def find_answer_token_span(
    tokenizer,
    full_input_ids: torch.Tensor,
    prompt: str,
    full_text: str,
    boundary: str = "",
) -> AnswerSpanAlignment:
    """
    Return exact answer span metadata for a teacher-forced prompt+answer input.

    Raises ValueError when the prompt is not a strict token prefix of full_text
    or when the full_text tokenization cannot be found inside full_input_ids.
    """
    prompt_ids = token_ids_no_special(tokenizer, prompt)
    full_ids = token_ids_no_special(tokenizer, full_text)

    if not full_ids:
        raise ValueError("full_text tokenized to an empty sequence")
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("answer span is empty after tokenization")
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "prompt tokens are not an exact prefix of reconstructed full_text; "
            "try another answer boundary"
        )

    input_ids = _as_list(full_input_ids[0])
    full_start = _find_subsequence(input_ids, full_ids)
    if full_start < 0:
        raise ValueError("failed to align full_text tokenization inside model input ids")

    start_idx = full_start + len(prompt_ids)
    end_idx = full_start + len(full_ids)
    answer_ids = full_ids[len(prompt_ids):]
    slice_ids = input_ids[start_idx:end_idx]
    if slice_ids != answer_ids:
        raise ValueError("aligned answer tokens do not match expected answer span")

    decoded = tokenizer.decode(slice_ids, skip_special_tokens=True)
    return AnswerSpanAlignment(
        prompt=prompt,
        answer=full_text[len(prompt) + len(boundary):],
        full_text=full_text,
        boundary=boundary,
        input_len=len(input_ids),
        prompt_len=len(prompt_ids),
        full_len=len(full_ids),
        answer_start=int(start_idx),
        answer_end=int(end_idx),
        answer_len=int(end_idx - start_idx),
        answer_token_ids=[int(x) for x in slice_ids],
        answer_tokens=_tokens_from_ids(tokenizer, slice_ids),
        decoded_answer_span=decoded,
    )


def align_answer_span(
    tokenizer,
    prompt: str,
    answer: str,
    device: Optional[torch.device | str] = None,
    boundaries: Sequence[str] = (" ", "", "\n", "\n "),
) -> Tuple[Dict[str, torch.Tensor], AnswerSpanAlignment]:
    """
    Try boundary variants and return tokenizer inputs plus exact span metadata.
    This does not require a model, only a tokenizer.
    """
    errors = []
    for boundary, full_text in build_full_text_candidates(prompt, answer, boundaries):
        inputs = tokenizer(full_text, return_tensors="pt")
        if device is not None:
            inputs = inputs.to(device)
        try:
            alignment = find_answer_token_span(
                tokenizer,
                inputs.input_ids,
                prompt,
                full_text,
                boundary=boundary,
            )
            return inputs, alignment
        except ValueError as exc:
            errors.append(f"{boundary!r}: {exc}")

    raise ValueError(
        "failed to align answer token span with any prompt/answer boundary. "
        + " | ".join(errors[:6])
    )


def tokenize_with_answer_span(model, tokenizer, prompt: str, answer: str):
    device = next(model.parameters()).device
    inputs, alignment = align_answer_span(tokenizer, prompt, answer, device=device)
    return inputs, alignment.answer_start, alignment.answer_end


def pool_hidden_states(
    layer_hidden: torch.Tensor,
    start_idx: int,
    end_idx: int,
    strategy: str = "first",
) -> np.ndarray:
    span = layer_hidden[start_idx:end_idx]
    if span.shape[0] == 0:
        span = layer_hidden[-1:]

    if strategy == "first":
        vec = span[0]
    elif strategy == "mean":
        vec = span.mean(dim=0)
    elif strategy == "last":
        vec = span[-1]
    else:
        raise ValueError("strategy must be one of: prompt_last, first, mean, last")

    return vec.float().cpu().numpy()


def forward_hidden_states_only(model, inputs):
    forward_kwargs = {
        **inputs,
        "output_hidden_states": True,
        "use_cache": False,
        "return_dict": True,
    }

    backbone = getattr(model, "model", None)
    if backbone is not None and backbone is not model:
        return backbone(**forward_kwargs)

    base_model = getattr(model, "base_model", None)
    if base_model is not None and base_model is not model:
        return base_model(**forward_kwargs)

    return model(**forward_kwargs)


def extract_answer_span_hidden_states(
    outputs,
    start_idx: int,
    end_idx: int,
) -> List[torch.Tensor]:
    layer_spans = []
    for layer_hs in outputs.hidden_states[1:]:
        span = layer_hs[0, start_idx:end_idx, :]
        if span.shape[0] == 0:
            span = layer_hs[0, -1:, :]
        layer_spans.append(span.detach().cpu())
    return layer_spans


def extract_hidden_states_single(
    model,
    tokenizer,
    prompt: str,
    answer: str,
    strategy: str = "first",
    return_alignment: bool = False,
):
    if not answer.strip():
        raise ValueError("empty answer after stripping")

    device = next(model.parameters()).device

    if strategy == "prompt_last":
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        start_idx = int(inputs.input_ids.shape[1] - 1)
        end_idx = int(inputs.input_ids.shape[1])
        alignment = AnswerSpanAlignment(
            prompt=prompt,
            answer="",
            full_text=prompt,
            boundary="",
            input_len=int(inputs.input_ids.shape[1]),
            prompt_len=int(inputs.input_ids.shape[1]),
            full_len=int(inputs.input_ids.shape[1]),
            answer_start=start_idx,
            answer_end=end_idx,
            answer_len=1,
            answer_token_ids=_as_list(inputs.input_ids[0, start_idx:end_idx]),
            answer_tokens=_tokens_from_ids(tokenizer, _as_list(inputs.input_ids[0, start_idx:end_idx])),
            decoded_answer_span=tokenizer.decode(inputs.input_ids[0, start_idx:end_idx]),
        )
    else:
        inputs, alignment = align_answer_span(tokenizer, prompt, answer, device=device)
        start_idx = alignment.answer_start
        end_idx = alignment.answer_end

    with torch.no_grad():
        outputs = forward_hidden_states_only(model, inputs)

    layer_spans = extract_answer_span_hidden_states(outputs, start_idx, end_idx)
    layer_vectors = [
        pool_hidden_states(span, 0, span.shape[0], strategy="first" if strategy == "prompt_last" else strategy)
        for span in layer_spans
    ]

    del outputs
    del inputs
    stacked = np.stack(layer_vectors)
    if return_alignment:
        return stacked, alignment
    return stacked


def extract_tokenwise_hidden_states_single(
    model,
    tokenizer,
    prompt: str,
    answer: str,
) -> Dict:
    if not answer.strip():
        raise ValueError("empty answer after stripping")

    device = next(model.parameters()).device
    inputs, alignment = align_answer_span(tokenizer, prompt, answer, device=device)

    with torch.no_grad():
        outputs = forward_hidden_states_only(model, inputs)

    layer_spans = extract_answer_span_hidden_states(
        outputs,
        alignment.answer_start,
        alignment.answer_end,
    )
    tokenwise_hidden_states = torch.stack(
        [span.to(dtype=torch.float16) for span in layer_spans],
        dim=0,
    ).contiguous()

    record = {
        "full_input_ids": inputs.input_ids[0].detach().cpu(),
        "answer_input_ids": inputs.input_ids[0, alignment.answer_start:alignment.answer_end].detach().cpu(),
        "answer_start": int(alignment.answer_start),
        "answer_end": int(alignment.answer_end),
        "alignment": alignment.to_json_dict(),
        "hidden_states": tokenwise_hidden_states,
    }

    del outputs
    del inputs
    return record


def extract_all_hidden_states(
    model,
    tokenizer,
    cases: List[Dict],
    strategy: str = "first",
    prompt_key: str = "prompt_w_context",
    answer_key: str = "ans_w_context",
    label_key: str = "label",
    max_samples: Optional[int] = None,
    return_alignments: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]] | Tuple[np.ndarray, Optional[np.ndarray], List[Dict]]:
    selected_cases = cases if max_samples is None else cases[:max_samples]
    all_hs = []
    all_labels = []
    alignments = []

    model.eval()
    for idx, item in enumerate(tqdm(selected_cases, desc=f"Extracting hidden states [{strategy}]")):
        prompt = item[prompt_key]
        answer = item[answer_key]
        try:
            hs, alignment = extract_hidden_states_single(
                model,
                tokenizer,
                prompt,
                answer,
                strategy=strategy,
                return_alignment=True,
            )
        except Exception as exc:
            ident = item.get("input_index", item.get("question", idx))
            raise RuntimeError(f"hidden-state extraction failed for item={ident!r}") from exc

        all_hs.append(hs)
        if label_key in item and item[label_key] is not None:
            all_labels.append(int(item[label_key]))
        alignments.append({
            "case_index": int(idx),
            "input_index": item.get("input_index"),
            "label": item.get(label_key),
            **alignment.to_json_dict(),
        })

    hidden_states = np.stack(all_hs)
    labels = np.array(all_labels, dtype=np.int64) if len(all_labels) == len(all_hs) else None

    if labels is not None:
        vals, counts = np.unique(labels, return_counts=True)
        dist = {int(v): int(c) for v, c in zip(vals, counts)}
        print(f"Extraction complete: shape={hidden_states.shape}, label_dist={dist}")
    else:
        print(f"Extraction complete: shape={hidden_states.shape}, labels=unavailable")

    if return_alignments:
        return hidden_states, labels, alignments
    return hidden_states, labels


def extract_tokenwise_hidden_states(
    model,
    tokenizer,
    cases: List[Dict],
    max_samples: Optional[int] = None,
    prompt_key: str = "prompt_w_context",
    answer_key: str = "ans_w_context",
) -> List[Dict]:
    selected_cases = cases if max_samples is None else cases[:max_samples]
    records = []
    model.eval()

    for idx, item in enumerate(tqdm(selected_cases, desc="Extracting tokenwise hidden states")):
        record = extract_tokenwise_hidden_states_single(
            model,
            tokenizer,
            item[prompt_key],
            item[answer_key],
        )
        record.update({
            "case_index": int(idx),
            "input_index": item.get("input_index"),
            "question": item.get("question"),
            "gold_answer": item.get("gold_answer"),
            "model_answer": item.get(answer_key),
            "label": item.get("label"),
            "case": item.get("case"),
            "answer_type": item.get("answer_type"),
        })
        records.append(record)
    return records


def _hidden_state_paths(save_dir: str, strategy: str) -> Tuple[str, str, str]:
    hs_path = os.path.join(save_dir, f"hs_{strategy}.npy")
    lbl_path = os.path.join(save_dir, f"labels_{strategy}.npy")
    meta_path = os.path.join(save_dir, f"meta_{strategy}.json")
    return hs_path, lbl_path, meta_path


def _tokenwise_paths(save_dir: str, name: str) -> Tuple[str, str]:
    payload_path = os.path.join(save_dir, f"{name}.pt")
    meta_path = os.path.join(save_dir, f"{name}.meta.json")
    return payload_path, meta_path


def _build_hidden_state_metadata(
    hidden_states: np.ndarray,
    labels: Optional[np.ndarray],
    strategy: str,
    metadata: Optional[Dict] = None,
) -> Dict:
    label_distribution = {}
    if labels is not None:
        label_vals, label_counts = np.unique(labels, return_counts=True)
        label_distribution = {
            str(int(label)): int(count)
            for label, count in zip(label_vals, label_counts)
        }
    meta = {
        "cache_version": HIDDEN_STATE_CACHE_VERSION,
        "strategy": strategy,
        "num_samples": int(hidden_states.shape[0]),
        "num_layers": int(hidden_states.shape[1]),
        "hidden_dim": int(hidden_states.shape[2]),
        "label_distribution": label_distribution,
    }
    if metadata:
        meta.update(metadata)
    return meta


def save_hidden_states(
    hidden_states: np.ndarray,
    labels: Optional[np.ndarray],
    strategy: str,
    out_dir: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> None:
    save_dir = out_dir if out_dir is not None else config.HIDDEN_STATE_DIR
    os.makedirs(save_dir, exist_ok=True)

    hs_path, lbl_path, meta_path = _hidden_state_paths(save_dir, strategy)
    np.save(hs_path, hidden_states)
    if labels is not None:
        np.save(lbl_path, labels)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            _build_hidden_state_metadata(hidden_states, labels, strategy, metadata),
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    print(f"Saved hidden states -> {hs_path}  shape={hidden_states.shape}")
    if labels is not None:
        print(f"Saved labels        -> {lbl_path}  shape={labels.shape}")
    print(f"Saved metadata      -> {meta_path}")


def load_hidden_state_metadata(strategy: str, hs_dir: Optional[str] = None) -> Dict:
    load_dir = hs_dir if hs_dir is not None else config.HIDDEN_STATE_DIR
    _, _, meta_path = _hidden_state_paths(load_dir, strategy)
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def hidden_state_cache_is_current(
    strategy: str,
    hs_dir: Optional[str] = None,
    expected_metadata: Optional[Dict] = None,
) -> Tuple[bool, str]:
    load_dir = hs_dir if hs_dir is not None else config.HIDDEN_STATE_DIR
    hs_path, lbl_path, meta_path = _hidden_state_paths(load_dir, strategy)

    for path in (hs_path, meta_path):
        if not os.path.exists(path):
            return False, f"missing cache file: {path}"

    try:
        meta = load_hidden_state_metadata(strategy, hs_dir=load_dir)
    except Exception as exc:
        return False, f"failed to read hidden-state metadata: {exc}"

    if meta.get("cache_version") != HIDDEN_STATE_CACHE_VERSION:
        return False, (
            f"hidden-state cache version mismatch "
            f"({meta.get('cache_version')} != {HIDDEN_STATE_CACHE_VERSION})"
        )
    if meta.get("strategy") != strategy:
        return False, f"strategy mismatch in cache ({meta.get('strategy')} != {strategy})"

    if meta.get("label_distribution") and not os.path.exists(lbl_path):
        return False, f"missing label cache file: {lbl_path}"

    if expected_metadata:
        for key, expected_value in expected_metadata.items():
            if meta.get(key) != expected_value:
                return False, f"metadata mismatch for '{key}' ({meta.get(key)} != {expected_value})"

    return True, "ok"


def load_hidden_states(strategy: str, hs_dir: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    load_dir = hs_dir if hs_dir is not None else config.HIDDEN_STATE_DIR
    hs_path, lbl_path, _ = _hidden_state_paths(load_dir, strategy)

    hidden_states = np.load(hs_path)
    labels = np.load(lbl_path) if os.path.exists(lbl_path) else None

    print(f"Loaded hidden states: {hs_path}  shape={hidden_states.shape}")
    if labels is not None:
        print(f"Loaded labels:        {lbl_path}  shape={labels.shape}")
    else:
        print("Loaded labels:        unavailable")
    return hidden_states, labels


def save_tokenwise_hidden_states(
    records: List[Dict],
    name: str = "tokenwise_w_context",
    out_dir: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> None:
    save_dir = out_dir if out_dir is not None else config.HIDDEN_STATE_DIR
    os.makedirs(save_dir, exist_ok=True)

    payload_path, meta_path = _tokenwise_paths(save_dir, name)
    torch.save(records, payload_path)

    label_dist = {}
    for record in records:
        key = str(record.get("label"))
        label_dist[key] = label_dist.get(key, 0) + 1

    meta = {
        "cache_version": TOKENWISE_CACHE_VERSION,
        "name": name,
        "num_samples": len(records),
        "label_distribution": label_dist,
    }
    if metadata:
        meta.update(metadata)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"Saved tokenwise cache -> {payload_path}  samples={len(records)}")
    print(f"Saved tokenwise meta  -> {meta_path}")


def load_tokenwise_hidden_states(
    name: str = "tokenwise_w_context",
    hs_dir: Optional[str] = None,
) -> List[Dict]:
    load_dir = hs_dir if hs_dir is not None else config.HIDDEN_STATE_DIR
    payload_path, _ = _tokenwise_paths(load_dir, name)
    records = torch.load(payload_path, map_location="cpu")
    print(f"Loaded tokenwise cache: {payload_path}  samples={len(records)}")
    return records


def write_alignment_report(records: Iterable[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved alignment report -> {path}")
