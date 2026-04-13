"""
extract_hidden_state.py

Extract per-layer hidden states at the answer token position for each case.

For each sample we run a forward pass with teacher forcing on the full
`prompt + answer` sequence, then collect the hidden state on the answer span
across ALL transformer layers.

Token position strategies:
  "first" : hidden state of the first answer token
  "mean"  : mean over all answer tokens
  "last"  : hidden state of the last answer token
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

import config

HIDDEN_STATE_CACHE_VERSION = 2
TOKENWISE_CACHE_VERSION = 1


# ── Token Position Utilities ──────────────────────────────────────────────────

def build_full_text(prompt: str, answer: str) -> Tuple[str, str]:
    """
    Default reconstruction used for metadata/debugging.
    The actual extraction path tries several boundary variants because
    tokenizers such as Llama-2's SentencePiece can be sensitive to how the
    generated continuation is reattached to the prompt.
    """
    answer = answer.strip()
    if not answer:
        return prompt, ""
    return prompt + " " + answer, answer


def build_full_text_candidates(prompt: str, answer: str) -> List[str]:
    """
    Candidate teacher-forced strings to try when reconstructing prompt+answer.
    """
    # breakpoint()
    answer = answer.strip()
    if not answer:
        return []

    candidates: List[str] = []
    # breakpoint()
    for sep in (" ", "", "\n", "\n "):
        full_text = prompt + sep + answer
        if full_text not in candidates:
            candidates.append(full_text)
    """
    (Pdb) full_text
    "Context: Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer. 
    In the film, two teenage girls cause their respective parents much concern when they start to become interested in boys. 
    The parents' bickering about which girl is the worse influence causes more problems than it solves. 
    Shirley Temple Black (April 23, 1928 – February 10, 2014) was an American actress, singer, dancer, businesswoman, 
    and diplomat who was Hollywood's number one box-office draw as a child actress from 1935 to 1938. As an adult, 
    she was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States.
    \nQuestion: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?
    \nAnswer:\n Chief of Protocol of the United States"
    
    *After ```full_text = prompt + sep + answer```*
    (Pdb) candidates
    ["Context: Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer. 
    In the film, two teenage girls cause their respective parents much concern when they start to become interested in boys. 
    The parents' bickering about which girl is the worse influence causes more problems than it solves. 
    Shirley Temple Black (April 23, 1928 – February 10, 2014) was an American actress, singer, dancer, businesswoman, 
    and diplomat who was Hollywood's number one box-office draw as a child actress from 1935 to 1938. As an adult, 
    she was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States.
    \nQuestion: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?
    \nAnswer:\n Chief of Protocol of the United States"]
    """
    return candidates


def _find_subsequence(sequence: List[int], pattern: List[int]) -> int:
    if not pattern or len(pattern) > len(sequence):
        return -1
    last_start = len(sequence) - len(pattern)
    for start in range(last_start + 1):
        if sequence[start:start + len(pattern)] == pattern:
            return start
    return -1

def find_answer_token_span(
    tokenizer,
    full_input_ids: torch.Tensor,
    prompt: str,
    full_text: str,
) -> Tuple[int, int]:
    """
    Return (start_idx, end_idx) of the answer tokens in the full sequence.
    end_idx is exclusive (Python-slice style).
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    breakpoint()

    if not full_ids:
        raise ValueError("Full teacher-forced text tokenized to an empty sequence.")
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("Answer span is empty after tokenization.")
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Prompt tokens are not a prefix of the reconstructed full text.")

    breakpoint()
    input_ids = full_input_ids[0].detach().cpu().tolist()
    full_start = _find_subsequence(input_ids, full_ids)
    breakpoint()
    if full_start < 0:
        raise ValueError("Failed to align full_text tokenization inside model input ids.")

    breakpoint()
    start_idx = full_start + len(prompt_ids)
    end_idx = full_start + len(full_ids)

    breakpoint()
    answer_ids = full_ids[len(prompt_ids):]
    slice_ids = input_ids[start_idx:end_idx]
    if slice_ids != answer_ids:
        raise ValueError("Aligned answer tokens do not match the expected answer span.")

    breakpoint()    
    return start_idx, end_idx


def tokenize_with_answer_span(model, tokenizer, prompt: str, answer: str):
    """
    Tokenize teacher-forced inputs and find a robust answer span.
    Tries several prompt/answer boundary variants and picks the first one that
    aligns at the token level.
    """
    device = next(model.parameters()).device
    errors = []

    for full_text in build_full_text_candidates(prompt, answer):
        inputs = tokenizer(full_text, return_tensors="pt").to(device)
        try:
            start_idx, end_idx = find_answer_token_span(
                tokenizer,
                inputs.input_ids,
                prompt,
                full_text,
            )
            return inputs, start_idx, end_idx
        except ValueError as e:
            boundary_preview = repr(full_text[len(prompt):len(prompt) + 6])
            errors.append(f"{boundary_preview}: {e}")

    # breakpoint()
    raise ValueError(
        "Failed to align answer token span with any prompt/answer boundary variant. "
        + " | ".join(errors[:4])
    )


def pool_hidden_states(
    layer_hidden: torch.Tensor,  # (seq_len, hidden_dim)
    start_idx: int,
    end_idx: int,
    strategy: str = "first",
) -> np.ndarray:
    """
    Extract a single vector from the answer span of a layer's hidden states.

    Args:
        layer_hidden : (seq_len, hidden_dim) tensor for one layer
        start_idx    : first answer token index (inclusive)
        end_idx      : last answer token index (exclusive)
        strategy     : "first" | "mean" | "last" | "all"
        # But option 'all' means that execute all three strategies and concatenate the results parallelly -> Each three strategy results will be stored

    Returns:
        1-D numpy array of shape (hidden_dim,)
    """
    breakpoint()
    span = layer_hidden[start_idx:end_idx]   # (span_len, hidden_dim)

    if len(span) == 0:
        # Fallback: use the last token of the full sequence
        span = layer_hidden[[-1]]

    if strategy == "first":
        vec = span[0]
    elif strategy == "mean":
        vec = span.mean(dim=0)
    elif strategy == "last":
        vec = span[-1]
    else:
        raise ValueError(f"Unknown pooling strategy: '{strategy}'. "
                         f"Choose from ['first', 'mean', 'last'].")

    return vec.float().cpu().numpy()


# ── Single-sample Extraction ──────────────────────────────────────────────────

def forward_hidden_states_only(model, inputs):
    """
    Prefer the transformer backbone over the LM head.

    For hidden-state extraction we do not need vocabulary logits. Calling the
    backbone avoids materializing a large `(seq_len, vocab_size)` tensor and is
    much less memory-hungry for later tokenwise analysis as well.
    """
    breakpoint()
    forward_kwargs = {
        **inputs,
        "output_hidden_states": True,
        "use_cache": False,
        "return_dict": True,
    }

    breakpoint()
    backbone = getattr(model, "model", None)
    if backbone is not None and backbone is not model:
        return backbone(**forward_kwargs)
    breakpoint()

    base_model = getattr(model, "base_model", None)
    if base_model is not None and base_model is not model:
        return base_model(**forward_kwargs)
    breakpoint()
    return model(**forward_kwargs)


def extract_answer_span_hidden_states(
    outputs,
    start_idx: int,
    end_idx: int,
) -> List[torch.Tensor]:
    """
    Return answer-span hidden states for every transformer layer on CPU.
    Shape per layer: (answer_len, hidden_dim)
    """
    breakpoint()
    layer_spans = []
    for layer_hs in outputs.hidden_states[1:]:
        breakpoint()
        span = layer_hs[0, start_idx:end_idx, :]
        if span.shape[0] == 0:
            breakpoint()
            span = layer_hs[0, -1:, :]
        layer_spans.append(span.detach().cpu())
        breakpoint()
    return layer_spans

# def extract_hidden_states_single(
#     model, tokenizer, prompt: str, answer: str, strategy: str = "first",
# ) -> np.ndarray:
#     """
#     Run a forward pass and return hidden states at the answer token position
#     for every transformer layer.

#     Returns:
#         np.ndarray of shape (num_layers, hidden_dim)
#         Layer 0 = first transformer block output (embedding layer excluded).
#     """
#     # full_text = prompt + " " + answer
#     # inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
#     # breakpoint()
    
#     # with torch.no_grad():
#     #     outputs = model(**inputs, output_hidden_states=True)

#     # # outputs.hidden_states: tuple of length (num_layers + 1)
#     # #   index 0 → embedding layer output  (excluded from our analysis)
#     # #   index 1 … L → transformer block outputs
#     # all_hidden_states = outputs.hidden_states[1:]   # skip embedding layer

#     # breakpoint()
    
#     # start_idx, end_idx = find_answer_token_span(tokenizer, prompt, answer)

#     # breakpoint()

#     # layer_vectors = []
#     # for layer_hs in all_hidden_states:
#     #     hs_2d = layer_hs[0]   # (seq_len, hidden_dim) — batch dim removed
#     #     vec   = pool_hidden_states(hs_2d, start_idx, end_idx, strategy)
#     #     layer_vectors.append(vec)

#     # return np.stack(layer_vectors)   # (num_layers, hidden_dim)
#     full_text = prompt + " " + answer
#     device = next(model.parameters()).device
#     inputs = tokenizer(full_text, return_tensors="pt").to(device)
    
#     with torch.no_grad():
#         outputs = model(**inputs, output_hidden_states=True)
        
#     all_hidden_states = outputs.hidden_states[1:]   # skip embedding layer
    
#     start_idx, end_idx = find_answer_token_span(tokenizer, inputs.input_ids, answer)

#     layer_vectors = []
#     for layer_hs in all_hidden_states:
#         hs_2d = layer_hs[0]   # (seq_len, hidden_dim) — batch dim removed
#         vec   = pool_hidden_states(hs_2d, start_idx, end_idx, strategy)
#         layer_vectors.append(vec)

#     return np.stack(layer_vectors)   # (num_layers, hidden_dim)

def extract_hidden_states_single(model, tokenizer, prompt, answer, strategy="first"):
    if not answer.strip():
        raise ValueError("Empty answer after stripping.")

    # breakpoint()
    """
    (Pdb) prompt
    "Context: Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer. In the film, 
    two teenage girls cause their respective parents much concern when they start to become interested in boys. The parents' bickering about which girl is the worse influence causes more problems than it solves. 
    Shirley Temple Black (April 23, 1928 – February 10, 2014) was an American actress, singer, dancer, businesswoman, and diplomat who was Hollywood's number one box-office draw as a child actress from 1935 to 1938. As an adult, she was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States.
    \nQuestion: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?\nAnswer:"
    (Pdb) answer
    'Chief of Protocol of the United States'
    """
    if strategy == "prompt_last":
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]
        
        start_idx = prompt_len - 1
        end_idx = prompt_len
        
        with torch.no_grad():
            outputs = forward_hidden_states_only(model, inputs)
            
        layer_spans = extract_answer_span_hidden_states(outputs, start_idx, end_idx)
        layer_vectors = []
        for span in layer_spans:
            vec = pool_hidden_states(span, 0, span.shape[0], strategy="first")
            layer_vectors.append(vec)
            
        del outputs
        del inputs
        return np.stack(layer_vectors)
    
    inputs, start_idx, end_idx = tokenize_with_answer_span(
        model,
        tokenizer,
        prompt,
        answer,
    )
    # breakpoint()
    """
    (Pdb) len(prompt)
    809
    (Pdb) len(inputs[0])
    188
    (Pdb) inputs[0]
    Encoding(num_tokens=188, attributes=[ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])
    """

    with torch.no_grad():
        breakpoint()
        outputs = forward_hidden_states_only(model, inputs)

    breakpoint()
    layer_spans = extract_answer_span_hidden_states(outputs, start_idx, end_idx)

    layer_vectors = []
    for span in layer_spans:
        breakpoint()
        vec = pool_hidden_states(span, 0, span.shape[0], strategy)
        layer_vectors.append(vec)

    breakpoint()
    del outputs
    del inputs

    return np.stack(layer_vectors)


def extract_tokenwise_hidden_states_single(
    model,
    tokenizer,
    prompt: str,
    answer: str,
) -> Dict:
    """
    Keep token-level answer-span hidden states for future logit-lens / tuned-lens
    style analyses on a smaller subset of cases.
    """
    breakpoint()
    if not answer.strip():
        raise ValueError("Empty answer after stripping.")

    breakpoint()
    inputs, start_idx, end_idx = tokenize_with_answer_span(
        model,
        tokenizer,
        prompt,
        answer,
    )

    with torch.no_grad():
        breakpoint()
        outputs = forward_hidden_states_only(model, inputs)

    breakpoint()
    layer_spans = extract_answer_span_hidden_states(outputs, start_idx, end_idx)

    tokenwise_hidden_states = torch.stack(
        [span.to(dtype=torch.float16) for span in layer_spans],
        dim=0,
    ).contiguous()

    record = {
        "full_input_ids": inputs.input_ids[0].detach().cpu(),
        "answer_input_ids": inputs.input_ids[0, start_idx:end_idx].detach().cpu(),
        "answer_start": int(start_idx),
        "answer_end": int(end_idx),
        "hidden_states": tokenwise_hidden_states,
    }

    del outputs
    del inputs

    return record

# ── Batch Extraction ──────────────────────────────────────────────────────────

def extract_all_hidden_states(
    model,
    tokenizer,
    cases: List[Dict],
    strategy: str = "first",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract hidden states for all cases.

    For both labels we use the w/ context prompt and the answer produced in the
    w/ context condition. This keeps the comparison aligned with the main
    research question: whether the model internally distinguishes between
    following the provided evidence (Case 1) and ignoring it (Case 3) when the
    evidence is present.

    Returns:
        hidden_states : np.ndarray  (N, num_layers, hidden_dim)
        labels        : np.ndarray  (N,)  0 = non-hallucination, 1 = hallucination
    """
    all_hs = []
    all_labels = []

    breakpoint()
    model.eval()

    for item in tqdm(cases, desc=f"Extracting hidden states [strategy={strategy}]"):
        breakpoint()
        label = item["label"]

        prompt = item["prompt_w_context"]
        answer = item["ans_w_context"]

        if not answer.strip():
            raise ValueError(f"Empty answer for question: {item['question']}")

        try:
            hs = extract_hidden_states_single(model, tokenizer, prompt, answer, strategy)
            breakpoint()
            all_hs.append(hs)
            all_labels.append(label)
        except Exception as e:
            raise RuntimeError(
                f"Hidden-state extraction failed for question: {item['question']}"
            ) from e

    breakpoint()
    hidden_states = np.stack(all_hs)          # (N, num_layers, hidden_dim)
    labels = np.array(all_labels)       # (N,)

    breakpoint()
    print(f"\nExtraction complete: {len(all_hs)} samples, "
          f"shape={hidden_states.shape}, "
          f"label dist={dict(zip(*np.unique(labels, return_counts=True)))}")

    return hidden_states, labels


def extract_tokenwise_hidden_states(
    model,
    tokenizer,
    cases: List[Dict],
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """
    Extract token-level answer-span hidden states for a smaller subset of cases.
    This is intended for later mechanistic analyses such as logit lens or tuned
    lens, not for the large pooled probe experiment.
    """
    selected_cases = cases if max_samples is None else cases[:max_samples]
    records = []

    breakpoint()
    model.eval()

    for idx, item in enumerate(
        tqdm(selected_cases, desc="Extracting tokenwise hidden states")
    ):
        breakpoint()
        prompt = item["prompt_w_context"]
        answer = item["ans_w_context"]
        record = extract_tokenwise_hidden_states_single(model, tokenizer, prompt, answer)
        record.update({
            "case_index": int(idx),
            "question": item["question"],
            "gold_answer": item["gold_answer"],
            "model_answer": answer,
            "label": int(item["label"]),
            "case": int(item["case"]),
            "answer_type": item["answer_type"],
        })
        records.append(record)

    return records


# ── Save / Load ───────────────────────────────────────────────────────────────

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
    labels: np.ndarray,
    strategy: str,
    metadata: Optional[Dict] = None,
) -> Dict:
    label_vals, label_counts = np.unique(labels, return_counts=True)
    meta = {
        "cache_version": HIDDEN_STATE_CACHE_VERSION,
        "strategy": strategy,
        "num_samples": int(hidden_states.shape[0]),
        "num_layers": int(hidden_states.shape[1]),
        "hidden_dim": int(hidden_states.shape[2]),
        "label_distribution": {
            str(int(label)): int(count)
            for label, count in zip(label_vals, label_counts)
        },
    }
    if metadata:
        meta.update(metadata)
    return meta

def save_hidden_states(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    strategy: str,
    out_dir: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> None:
    """Save hidden states and labels as .npy files."""
    save_dir = out_dir if out_dir is not None else config.HIDDEN_STATE_DIR
    os.makedirs(save_dir, exist_ok=True)

    hs_path, lbl_path, meta_path = _hidden_state_paths(save_dir, strategy)

    np.save(hs_path,  hidden_states)
    np.save(lbl_path, labels)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            _build_hidden_state_metadata(hidden_states, labels, strategy, metadata),
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    print(f"Saved hidden states → {hs_path}  shape={hidden_states.shape}")
    print(f"Saved labels        → {lbl_path}  shape={labels.shape}")
    print(f"Saved metadata      → {meta_path}")


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

    for path in (hs_path, lbl_path, meta_path):
        if not os.path.exists(path):
            return False, f"missing cache file: {path}"

    try:
        meta = load_hidden_state_metadata(strategy, hs_dir=load_dir)
    except Exception as e:
        return False, f"failed to read hidden-state metadata: {e}"

    if meta.get("cache_version") != HIDDEN_STATE_CACHE_VERSION:
        return False, (
            f"hidden-state cache version mismatch "
            f"({meta.get('cache_version')} != {HIDDEN_STATE_CACHE_VERSION})"
        )
    if meta.get("strategy") != strategy:
        return False, f"strategy mismatch in cache ({meta.get('strategy')} != {strategy})"

    if expected_metadata:
        for key, expected_value in expected_metadata.items():
            if meta.get(key) != expected_value:
                return False, f"metadata mismatch for '{key}' ({meta.get(key)} != {expected_value})"

    return True, "ok"


def load_hidden_states(strategy: str, hs_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load previously saved hidden states and labels."""
    load_dir = hs_dir if hs_dir is not None else config.HIDDEN_STATE_DIR
    hs_path, lbl_path, _ = _hidden_state_paths(load_dir, strategy)

    hidden_states = np.load(hs_path)
    labels        = np.load(lbl_path)

    print(f"Loaded hidden states: {hs_path}  shape={hidden_states.shape}")
    print(f"Loaded labels:        {lbl_path}  shape={labels.shape}")

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
        key = str(record["label"])
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

    print(f"Saved tokenwise cache → {payload_path}  samples={len(records)}")
    print(f"Saved tokenwise meta  → {meta_path}")


def load_tokenwise_hidden_states(
    name: str = "tokenwise_w_context",
    hs_dir: Optional[str] = None,
) -> List[Dict]:
    load_dir = hs_dir if hs_dir is not None else config.HIDDEN_STATE_DIR
    payload_path, _ = _tokenwise_paths(load_dir, name)
    records = torch.load(payload_path, map_location="cpu")
    print(f"Loaded tokenwise cache: {payload_path}  samples={len(records)}")
    return records
