"""
extract_hidden_state.py

Extract per-layer hidden states at the answer token position for each case.

For each sample we run a forward pass with teacher forcing and collect the
hidden state at the answer token position across ALL transformer layers.
This gives us a matrix of shape (num_layers, hidden_dim) per sample,
which is the input to the linear probe.

Token position strategies (config.TOKEN_POSITIONS):
  "first" : hidden state of the first answer token  (Phase 2 default)
  "mean"  : mean over all answer tokens
  "last"  : hidden state of the last answer token
  "all"   : concatenation of all answer token hidden states
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

import config


# ── Token Position Utilities ──────────────────────────────────────────────────

def find_answer_token_span(
    # tokenizer, prompt: str, answer: str,
    tokenizer, full_input_ids: torch.Tensor, answer: str,
) -> Tuple[int, int]:
    """
    Return (start_idx, end_idx) of the answer tokens in the full sequence.
    end_idx is exclusive (Python-slice style).
    """
    # full_text = prompt + " " + answer
    # prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    # full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

    # start_idx = len(prompt_ids)
    # end_idx = len(full_ids)           # exclusive

    # # Safety clamp
    # start_idx = min(start_idx, end_idx)
    # if start_idx == end_idx:            # degenerate: answer tokenized to nothing
    #     start_idx = max(0, end_idx - 1)
    # # breakpoint()  # Debugging: inspect tokenization issues if this happens

    # return start_idx, end_idx
    answer_ids = tokenizer(
        answer, add_special_tokens=False
    ).input_ids
    end_idx = full_input_ids.shape[1]
    start_idx = end_idx - len(answer_ids)
    start_idx = max(0, start_idx)
    return start_idx, end_idx


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

def extract_hidden_states_single(
    model, tokenizer, prompt: str, answer: str, strategy: str = "first",
) -> np.ndarray:
    """
    Run a forward pass and return hidden states at the answer token position
    for every transformer layer.

    Returns:
        np.ndarray of shape (num_layers, hidden_dim)
        Layer 0 = first transformer block output (embedding layer excluded).
    """
    # full_text = prompt + " " + answer
    # inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    # breakpoint()
    
    # with torch.no_grad():
    #     outputs = model(**inputs, output_hidden_states=True)

    # # outputs.hidden_states: tuple of length (num_layers + 1)
    # #   index 0 → embedding layer output  (excluded from our analysis)
    # #   index 1 … L → transformer block outputs
    # all_hidden_states = outputs.hidden_states[1:]   # skip embedding layer

    # breakpoint()
    
    # start_idx, end_idx = find_answer_token_span(tokenizer, prompt, answer)

    # breakpoint()

    # layer_vectors = []
    # for layer_hs in all_hidden_states:
    #     hs_2d = layer_hs[0]   # (seq_len, hidden_dim) — batch dim removed
    #     vec   = pool_hidden_states(hs_2d, start_idx, end_idx, strategy)
    #     layer_vectors.append(vec)

    # return np.stack(layer_vectors)   # (num_layers, hidden_dim)
    full_text = prompt + " " + answer
    device = next(model.parameters()).device
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
    all_hidden_states = outputs.hidden_states[1:]   # skip embedding layer
    
    start_idx, end_idx = find_answer_token_span(tokenizer, inputs.input_ids, answer)

    layer_vectors = []
    for layer_hs in all_hidden_states:
        hs_2d = layer_hs[0]   # (seq_len, hidden_dim) — batch dim removed
        vec   = pool_hidden_states(hs_2d, start_idx, end_idx, strategy)
        layer_vectors.append(vec)

    return np.stack(layer_vectors)   # (num_layers, hidden_dim)

# ── Batch Extraction ──────────────────────────────────────────────────────────

def extract_all_hidden_states(
    model,
    tokenizer,
    cases: List[Dict],
    strategy: str = "first",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract hidden states for all cases.

    For label 0 (non-hallucination / Case 1):
        Uses the w/ context prompt and the w/ context answer
        (the setting in which the model answered correctly).

    For label 1 (hallucination / Case 3):
        Uses the w/o context prompt and the w/o context answer
        (the setting in which the hallucination occurs).

    Returns:
        hidden_states : np.ndarray  (N, num_layers, hidden_dim)
        labels        : np.ndarray  (N,)  0 = non-hallucination, 1 = hallucination
    """
    all_hs = []
    all_labels = []

    model.eval()

    for item in tqdm(cases, desc=f"Extracting hidden states [strategy={strategy}]"):
        label = item["label"]

        # # Select the prompt/answer pair that reflects the label's condition
        # if label == 0:
        #     prompt = item["prompt_w_context"]
        #     answer = item["ans_w_context"]
        # else:
        #     prompt = item["prompt_wo_context"]
        #     answer = item["ans_wo_context"]
        prompt = item["prompt_w_context"]
        answer = item["ans_w_context"]

        if not answer.strip():
            continue   # Skip degenerate empty answers

        try:
            hs = extract_hidden_states_single(model, tokenizer, prompt, answer, strategy)
            all_hs.append(hs)
            all_labels.append(label)
        except Exception as e:
            print(f"[skip] {item['question'][:60]}... | error: {e}")
            continue

    hidden_states = np.stack(all_hs)          # (N, num_layers, hidden_dim)
    labels        = np.array(all_labels)       # (N,)

    print(f"\nExtraction complete: {len(all_hs)} samples, "
          f"shape={hidden_states.shape}, "
          f"label dist={dict(zip(*np.unique(labels, return_counts=True)))}")

    return hidden_states, labels


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_hidden_states(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    strategy: str,
    out_dir: Optional[str] = None,
) -> None:
    """Save hidden states and labels as .npy files."""
    save_dir = out_dir if out_dir is not None else config.HIDDEN_STATE_DIR
    os.makedirs(save_dir, exist_ok=True)

    hs_path  = os.path.join(save_dir, f"hs_{strategy}.npy")
    lbl_path = os.path.join(save_dir, f"labels_{strategy}.npy")

    np.save(hs_path,  hidden_states)
    np.save(lbl_path, labels)

    print(f"Saved hidden states → {hs_path}  shape={hidden_states.shape}")
    print(f"Saved labels        → {lbl_path}  shape={labels.shape}")


def load_hidden_states(strategy: str, hs_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load previously saved hidden states and labels."""
    load_dir = hs_dir if hs_dir is not None else config.HIDDEN_STATE_DIR
    hs_path  = os.path.join(load_dir, f"hs_{strategy}.npy")
    lbl_path = os.path.join(load_dir, f"labels_{strategy}.npy")

    hidden_states = np.load(hs_path)
    labels        = np.load(lbl_path)

    print(f"Loaded hidden states: {hs_path}  shape={hidden_states.shape}")
    print(f"Loaded labels:        {lbl_path}  shape={labels.shape}")

    return hidden_states, labels