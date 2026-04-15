"""
uncertainty_metrics.py

Uncertainty metric extractors for hallucination analysis.

Functions:
    extract_logit_entropy    : entropy of softmax distribution at answer token position
    extract_logit_margin     : top1 - top2 probability difference at answer position
    extract_attention_entropy: entropy of attention weights (layer-wise) at answer position
    extract_logit_lens       : apply lm_head to each layer's hidden state -> token distribution
    extract_tuned_lens       : learned affine transform per layer before lm_head

Description:
extract_logit_entropy(model, tokenizer, cases)
  -> 답 생성 직전 softmax 분포의 entropy

extract_logit_margin(model, tokenizer, cases)
  -> top1 - top2 확률 차이

extract_attention_entropy(model, tokenizer, cases)
  -> context 토큰에 대한 attention entropy (레이어별)

extract_logit_lens(model, tokenizer, cases)
  -> 각 레이어 hidden state -> lm_head 통과 -> 예측 토큰 분포
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm

import config

logger = logging.getLogger(__name__)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _get_answer_pos(tokenizer, seq_len: int, answer: str) -> int:
    """Return the index of the first answer token in the full sequence."""
    answer_ids = tokenizer(answer.strip(), add_special_tokens=False)["input_ids"]
    pos = seq_len - len(answer_ids)
    return max(0, min(pos, seq_len - 1))


def _plot_metric_distribution(
    values0: List[float],
    values1: List[float],
    metric_name: str,
    strategy: str,
    figure_dir: str,
    filename: str,
) -> None:
    """Histogram + box-plot comparison between label 0 and label 1."""
    os.makedirs(figure_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    ax = axes[0]
    if values0:
        ax.hist(values0, bins=40, color="steelblue", alpha=0.6,
                density=True, label="Non-Hall (0)")
    if values1:
        ax.hist(values1, bins=40, color="tomato", alpha=0.6,
                density=True, label="Hall (1)")
    ax.set_xlabel(metric_name, fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"{metric_name} Distribution [{strategy}]", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Box plot
    ax2 = axes[1]
    data   = [v for v in [values0, values1] if v]
    labels = [l for l, v in zip(["Non-Hall (0)", "Hall (1)"], [values0, values1]) if v]
    colors = ["steelblue", "tomato"][: len(data)]
    if data:
        bp = ax2.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    ax2.set_ylabel(metric_name, fontsize=11)
    ax2.set_title(f"{metric_name} Box Plot [{strategy}]", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(figure_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close()


# ── extract_logit_entropy ──────────────────────────────────────────────────────

def extract_logit_entropy(
    model,
    tokenizer,
    hidden_states: np.ndarray,   # (N, num_layers, hidden_dim) — interface consistency
    direction: np.ndarray,        # (hidden_dim,) — probe direction
    cases: List[Dict],
    strategy: str,
    figure_dir: Optional[str] = None,
) -> Dict:
    """
    답 생성 직전 softmax 분포의 entropy를 계산.

    For each case, compute the entropy of the next-token probability distribution
    at the position just before the first answer token.

    Returns:
        {
            "label0": [entropy, ...],
            "label1": [entropy, ...],
            "mean_label0": float,
            "mean_label1": float,
        }
    """
    model.eval()
    device = next(model.parameters()).device
    entropies_by_label: Dict[int, List[float]] = {0: [], 1: []}

    for item in tqdm(cases, desc="[logit_entropy]"):
        label = item.get("label")
        if label not in (0, 1):
            continue

        prompt    = item["prompt_w_context"]
        answer    = item["ans_w_context"]
        full_text = prompt + " " + answer

        inputs  = tokenizer(full_text, return_tensors="pt").to(device)
        seq_len = inputs.input_ids.shape[1]
        ans_pos = _get_answer_pos(tokenizer, seq_len, answer)
        pred_pos = max(0, ans_pos - 1)  # predict ans_pos from pred_pos

        try:
            with torch.no_grad():
                logits = model(**inputs, return_dict=True).logits  # (1, seq, vocab)
                probs   = torch.softmax(logits[0, pred_pos, :].float(), dim=-1)
                entropy = -float((probs * torch.log(probs + 1e-10)).sum().cpu())
                entropies_by_label[label].append(entropy)
        except Exception as e:
            logger.warning(f"[logit_entropy] sample skipped: {e}")
            continue

    mean0 = float(np.mean(entropies_by_label[0])) if entropies_by_label[0] else float("nan")
    mean1 = float(np.mean(entropies_by_label[1])) if entropies_by_label[1] else float("nan")

    logger.info(
        "[logit_entropy | strategy=%s] label=0 mean=%.4f  label=1 mean=%.4f  diff=%+.4f",
        strategy, mean0, mean1, mean1 - mean0,
    )
    print(
        f"[logit_entropy | {strategy}] "
        f"Non-Hall mean={mean0:.4f}  Hall mean={mean1:.4f}  diff={mean1 - mean0:+.4f}"
    )

    result = {
        "label0": entropies_by_label[0],
        "label1": entropies_by_label[1],
        "mean_label0": mean0,
        "mean_label1": mean1,
    }

    if figure_dir is not None:
        _plot_metric_distribution(
            entropies_by_label[0], entropies_by_label[1],
            metric_name="Logit Entropy",
            strategy=strategy,
            figure_dir=figure_dir,
            filename=f"logit_entropy_{strategy}.png",
        )

    return result


# ── extract_logit_margin ───────────────────────────────────────────────────────

def extract_logit_margin(
    model,
    tokenizer,
    hidden_states: np.ndarray,
    direction: np.ndarray,
    cases: List[Dict],
    strategy: str,
    figure_dir: Optional[str] = None,
) -> Dict:
    """
    top1 - top2 확률 차이를 계산.

    Larger margin = more confident prediction.
    Lower margin for hallucination -> model is less certain.

    Returns:
        {
            "label0": [margin, ...],
            "label1": [margin, ...],
            "mean_label0": float,
            "mean_label1": float,
        }
    """
    model.eval()
    device = next(model.parameters()).device
    margins_by_label: Dict[int, List[float]] = {0: [], 1: []}

    for item in tqdm(cases, desc="[logit_margin]"):
        label = item.get("label")
        if label not in (0, 1):
            continue

        prompt    = item["prompt_w_context"]
        answer    = item["ans_w_context"]
        full_text = prompt + " " + answer

        inputs  = tokenizer(full_text, return_tensors="pt").to(device)
        seq_len = inputs.input_ids.shape[1]
        ans_pos = _get_answer_pos(tokenizer, seq_len, answer)
        pred_pos = max(0, ans_pos - 1)

        try:
            with torch.no_grad():
                logits = model(**inputs, return_dict=True).logits
                probs   = torch.softmax(logits[0, pred_pos, :].float(), dim=-1)
                top2    = torch.topk(probs, 2).values
                margin  = float((top2[0] - top2[1]).cpu())
                margins_by_label[label].append(margin)
        except Exception as e:
            logger.warning(f"[logit_margin] sample skipped: {e}")
            continue

    mean0 = float(np.mean(margins_by_label[0])) if margins_by_label[0] else float("nan")
    mean1 = float(np.mean(margins_by_label[1])) if margins_by_label[1] else float("nan")

    logger.info(
        "[logit_margin | strategy=%s] label=0 mean=%.4f  label=1 mean=%.4f  diff=%+.4f",
        strategy, mean0, mean1, mean1 - mean0,
    )
    print(
        f"[logit_margin | {strategy}] "
        f"Non-Hall mean={mean0:.4f}  Hall mean={mean1:.4f}  diff={mean1 - mean0:+.4f}"
    )

    result = {
        "label0": margins_by_label[0],
        "label1": margins_by_label[1],
        "mean_label0": mean0,
        "mean_label1": mean1,
    }

    if figure_dir is not None:
        _plot_metric_distribution(
            margins_by_label[0], margins_by_label[1],
            metric_name="Logit Margin (top1 − top2)",
            strategy=strategy,
            figure_dir=figure_dir,
            filename=f"logit_margin_{strategy}.png",
        )

    return result


# ── extract_attention_entropy ──────────────────────────────────────────────────

def extract_attention_entropy(
    model,
    tokenizer,
    hidden_states: np.ndarray,
    direction: np.ndarray,
    cases: List[Dict],
    strategy: str,
    figure_dir: Optional[str] = None,
) -> Dict:
    """
    context 토큰에 대한 attention entropy (레이어별)를 계산.

    For each case, compute the entropy of attention weights at the first answer
    token position, averaged across heads, per layer.

    Returns:
        {
            "label0_layer_entropy": [mean_entropy_per_layer, ...],   # len = num_layers
            "label1_layer_entropy": [mean_entropy_per_layer, ...],
        }
    """
    if hasattr(model, "set_attn_implementation"):
        try:
            model.set_attn_implementation("eager")
        except Exception:
            pass

    model.eval()
    device    = next(model.parameters()).device
    num_layers = model.config.num_hidden_layers

    entropy_by_label: Dict[int, List[List[float]]] = {
        0: [[] for _ in range(num_layers)],
        1: [[] for _ in range(num_layers)],
    }

    for item in tqdm(cases, desc="[attention_entropy]"):
        label = item.get("label")
        if label not in (0, 1):
            continue

        prompt    = item["prompt_w_context"]
        answer    = item["ans_w_context"]
        full_text = prompt + " " + answer

        inputs  = tokenizer(full_text, return_tensors="pt").to(device)
        seq_len = inputs.input_ids.shape[1]
        ans_pos = _get_answer_pos(tokenizer, seq_len, answer)

        try:
            with torch.no_grad():
                outputs = model(
                    **inputs,
                    output_attentions=True,
                    use_cache=False,
                    return_dict=True,
                )
        except Exception as e:
            logger.warning(f"[attention_entropy] sample skipped: {e}")
            continue

        if outputs.attentions is None:
            logger.warning("[attention_entropy] attentions is None — skipping sample")
            continue

        for li, attn in enumerate(outputs.attentions):
            # attn: (1, num_heads, seq, seq)
            attn_row = attn[0, :, ans_pos, :].float().cpu()   # (heads, seq)
            # Clamp near-zero to avoid log(0)
            attn_row = torch.clamp(attn_row, min=1e-10)
            attn_row = attn_row / attn_row.sum(dim=-1, keepdim=True)
            h = -float((attn_row * torch.log(attn_row)).sum(dim=-1).mean().cpu())
            entropy_by_label[label][li].append(h)

    mean0 = [np.mean(v) if v else float("nan") for v in entropy_by_label[0]]
    mean1 = [np.mean(v) if v else float("nan") for v in entropy_by_label[1]]

    # Peak layer (max entropy = most spread attention)
    best0 = int(np.nanargmax(mean0)) if any(not np.isnan(m) for m in mean0) else -1
    best1 = int(np.nanargmax(mean1)) if any(not np.isnan(m) for m in mean1) else -1

    logger.info(
        "[attention_entropy | strategy=%s] label=0 peak_layer=%d (%.4f)  "
        "label=1 peak_layer=%d (%.4f)",
        strategy,
        best0, mean0[best0] if best0 >= 0 else float("nan"),
        best1, mean1[best1] if best1 >= 0 else float("nan"),
    )
    print(
        f"[attention_entropy | {strategy}] "
        f"Non-Hall peak layer={best0} (entropy={mean0[best0]:.4f})  "
        f"Hall peak layer={best1} (entropy={mean1[best1]:.4f})"
    )

    result = {
        "label0_layer_entropy": mean0,
        "label1_layer_entropy": mean1,
    }

    if figure_dir is not None:
        os.makedirs(figure_dir, exist_ok=True)
        layers = list(range(num_layers))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(layers, mean0, color="steelblue", linewidth=2,
                marker="o", markersize=3, label="Non-Hall (0)")
        ax.plot(layers, mean1, color="tomato", linewidth=2,
                marker="o", markersize=3, label="Hall (1)")
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Attention Entropy (nats)", fontsize=12)
        ax.set_title(f"Layer-wise Attention Entropy at Answer Position [{strategy}]",
                     fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        plt.tight_layout()
        path = os.path.join(figure_dir, f"attention_entropy_{strategy}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
        plt.close()

    return result


# ── extract_logit_lens ─────────────────────────────────────────────────────────

def extract_logit_lens(
    model,
    tokenizer,
    hidden_states: np.ndarray,   # (N, num_layers, hidden_dim)
    direction: np.ndarray,        # (hidden_dim,)
    cases: List[Dict],
    strategy: str,
    figure_dir: Optional[str] = None,
) -> Dict:
    """
    각 레이어 hidden state -> lm_head 통과 -> 예측 토큰 분포.

    Uses precomputed hidden_states (shape N × num_layers × hidden_dim).
    Applies model.model.norm then model.lm_head at each layer.
    Reports entropy per layer, averaged by label.

    Returns:
        {
            "label0_layer_entropy": [mean_entropy_per_layer, ...],
            "label1_layer_entropy": [mean_entropy_per_layer, ...],
        }
    """
    if not hasattr(model, "lm_head"):
        print("[logit_lens] ERROR: model has no lm_head — skipping.")
        return {}

    model.eval()
    device     = next(model.parameters()).device
    num_layers = hidden_states.shape[1]

    labels_list = [item.get("label", -1) for item in cases]

    entropy_by_label: Dict[int, List[List[float]]] = {
        0: [[] for _ in range(num_layers)],
        1: [[] for _ in range(num_layers)],
    }

    for i, item in enumerate(tqdm(cases, desc="[logit_lens]")):
        label = item.get("label")
        if label not in (0, 1):
            continue
        if i >= len(hidden_states):
            continue

        for li in range(num_layers):
            hs = torch.tensor(hidden_states[i, li], dtype=torch.float32).unsqueeze(0).to(device)
            try:
                with torch.no_grad():
                    if hasattr(model, "model") and hasattr(model.model, "norm"):
                        hs = model.model.norm(hs)
                    logits  = model.lm_head(hs)          # (1, vocab_size)
                    probs   = torch.softmax(logits[0].float(), dim=-1)
                    entropy = -float((probs * torch.log(probs + 1e-10)).sum().cpu())
                    entropy_by_label[label][li].append(entropy)
            except Exception:
                continue

    mean0 = [np.mean(v) if v else float("nan") for v in entropy_by_label[0]]
    mean1 = [np.mean(v) if v else float("nan") for v in entropy_by_label[1]]

    best0 = int(np.nanargmin(mean0)) if any(not np.isnan(m) for m in mean0) else -1
    best1 = int(np.nanargmin(mean1)) if any(not np.isnan(m) for m in mean1) else -1

    logger.info(
        "[logit_lens | strategy=%s] label=0 min_entropy_layer=%d (%.4f)  "
        "label=1 min_entropy_layer=%d (%.4f)",
        strategy,
        best0, mean0[best0] if best0 >= 0 else float("nan"),
        best1, mean1[best1] if best1 >= 0 else float("nan"),
    )
    print(
        f"[logit_lens | {strategy}] "
        f"Non-Hall most-confident layer={best0}  Hall most-confident layer={best1}"
    )

    result = {
        "label0_layer_entropy": mean0,
        "label1_layer_entropy": mean1,
    }

    if figure_dir is not None:
        os.makedirs(figure_dir, exist_ok=True)
        layers = list(range(num_layers))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(layers, mean0, color="steelblue", linewidth=2,
                marker="o", markersize=3, label="Non-Hall (0)")
        ax.plot(layers, mean1, color="tomato", linewidth=2,
                marker="o", markersize=3, label="Hall (1)")
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Logit-Lens Entropy (nats)", fontsize=12)
        ax.set_title(f"Logit Lens: Layer-wise Entropy [{strategy}]", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        plt.tight_layout()
        path = os.path.join(figure_dir, f"logit_lens_{strategy}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
        plt.close()

    return result


# ── extract_tuned_lens ─────────────────────────────────────────────────────────

def extract_tuned_lens(
    model,
    tokenizer,
    hidden_states: np.ndarray,   # (N, num_layers, hidden_dim)
    direction: np.ndarray,        # (hidden_dim,)
    cases: List[Dict],
    strategy: str,
    figure_dir: Optional[str] = None,
) -> Dict:
    """
    Learned affine transform per layer before lm_head.

    For each layer l, fit a linear map  T_l: h_l -> h_final  using the
    precomputed hidden states (ridge regression), then apply lm_head to the
    mapped representation and report entropy per layer.

    This is a lightweight "self-supervised tuned lens" that needs no external
    checkpoint — it trains the translators in-memory on the provided data.

    Returns:
        {
            "label0_layer_entropy": [mean_entropy_per_layer, ...],
            "label1_layer_entropy": [mean_entropy_per_layer, ...],
        }
    """
    from sklearn.linear_model import Ridge

    if not hasattr(model, "lm_head"):
        print("[tuned_lens] ERROR: model has no lm_head — skipping.")
        return {}

    model.eval()
    device     = next(model.parameters()).device
    num_layers = hidden_states.shape[1]

    # Target: hidden state at the final layer (index -1)
    H_final = hidden_states[:, -1, :]   # (N, hidden_dim)

    labels_list = np.array([item.get("label", -1) for item in cases])

    entropy_by_label: Dict[int, List[List[float]]] = {
        0: [[] for _ in range(num_layers)],
        1: [[] for _ in range(num_layers)],
    }

    print(f"[tuned_lens | {strategy}] Fitting per-layer translators ...")
    for li in tqdm(range(num_layers), desc="[tuned_lens] fit"):
        H_l = hidden_states[:, li, :]   # (N, hidden_dim)

        # Fit affine map H_l -> H_final (column-wise ridge regression)
        ridge = Ridge(alpha=1.0, fit_intercept=True)
        ridge.fit(H_l, H_final)

        H_mapped = ridge.predict(H_l)   # (N, hidden_dim)

        for i, item in enumerate(cases):
            label = item.get("label")
            if label not in (0, 1):
                continue
            if i >= len(H_mapped):
                continue

            hs = torch.tensor(H_mapped[i], dtype=torch.float32).unsqueeze(0).to(device)
            try:
                with torch.no_grad():
                    if hasattr(model, "model") and hasattr(model.model, "norm"):
                        hs = model.model.norm(hs)
                    logits  = model.lm_head(hs)
                    probs   = torch.softmax(logits[0].float(), dim=-1)
                    entropy = -float((probs * torch.log(probs + 1e-10)).sum().cpu())
                    entropy_by_label[label][li].append(entropy)
            except Exception:
                continue

    mean0 = [np.mean(v) if v else float("nan") for v in entropy_by_label[0]]
    mean1 = [np.mean(v) if v else float("nan") for v in entropy_by_label[1]]

    best0 = int(np.nanargmin(mean0)) if any(not np.isnan(m) for m in mean0) else -1
    best1 = int(np.nanargmin(mean1)) if any(not np.isnan(m) for m in mean1) else -1

    logger.info(
        "[tuned_lens | strategy=%s] label=0 min_entropy_layer=%d  label=1 min_entropy_layer=%d",
        strategy, best0, best1,
    )
    print(
        f"[tuned_lens | {strategy}] "
        f"Non-Hall most-confident layer={best0}  Hall most-confident layer={best1}"
    )

    result = {
        "label0_layer_entropy": mean0,
        "label1_layer_entropy": mean1,
    }

    if figure_dir is not None:
        os.makedirs(figure_dir, exist_ok=True)
        layers = list(range(num_layers))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(layers, mean0, color="steelblue", linewidth=2,
                marker="o", markersize=3, label="Non-Hall (0)")
        ax.plot(layers, mean1, color="tomato", linewidth=2,
                marker="o", markersize=3, label="Hall (1)")
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Tuned-Lens Entropy (nats)", fontsize=12)
        ax.set_title(f"Tuned Lens: Layer-wise Entropy [{strategy}]", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        plt.tight_layout()
        path = os.path.join(figure_dir, f"tuned_lens_{strategy}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
        plt.close()

    return result
