"""
visualize.py


Extended analysis modules for the Linguisic Hyperplane proving.
Each function is a self-contained analysis that can be called independently.

Phases:
* 
* probe_direction: probe coefficients -> vocabulary direction
* pca_analysis   : PCA top-k directions, label alignment
* cka            : layer-wise CKA between label 0 / label 1 group
* attention      : attention weight ratio to context tokens with visualization
* transformerLens: 
"""

import os
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tqdm import tqdm

import config


# Shared utility

def _ascii_safe(text: str) -> str:
    if text is None:
        return "<None>"
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    return text.encode("ascii", "backslashreplace").decode("ascii") or "<empty>"

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    

# Figure 1: Layer-wise Accuracy / AUROC vis.
def plot_layer_accuracy(
    results_dict: Dict[str, List[Dict]],
    save: bool = True,
    filename: str = "layer_accuracy_curve.png",
    figure_dir: Optional[str] = None,
) -> None:
    save_dir = figure_dir if figure_dir is not None else config.FIGURE_DIR
    os.makedirs(save_dir, exist_ok=True)
    colors = ["steelblue", "tomato", "seagreen", "orange", "mediumpurple"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric in zip(axes, ["accuracy", "auroc"]):
        for (strategy, results), color in zip(results_dict.items(), colors):
            layers = [r["layer"] for r in results]
            values = [r[metric]  for r in results]
            ax.plot(layers, values, label=strategy, color=color,
                    linewidth=2, marker="o", markersize=3)

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
        if metric == "auroc":
            ax.axhline(0.65, color="red", linestyle=":", linewidth=1, label="threshold (0.65)")

        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f"Layer-wise {metric.upper()} — Hallucination Probe", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    plt.tight_layout()

    if save:
        path = os.path.join(save_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")

    plt.show()
    plt.close()


# Figure 2: t-SNE Visualization vis.
def plot_tsne(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    layer_indices: Optional[List[int]] = None,
    strategy: str = "first",
    save: bool = True,
    figure_dir: Optional[str] = None,
) -> None:
    save_dir = figure_dir if figure_dir is not None else config.FIGURE_DIR
    os.makedirs(save_dir, exist_ok=True)

    num_layers = hidden_states.shape[1]
    if layer_indices is None:
        layer_indices = [0, num_layers // 4, num_layers // 2, num_layers - 1]

    fig, axes = plt.subplots(1, len(layer_indices),
                             figsize=(5 * len(layer_indices), 4))
    if len(layer_indices) == 1:
        axes = [axes]

    for ax, layer_idx in zip(axes, layer_indices):
        X = hidden_states[:, layer_idx, :]
        pca_dim = min(50, X.shape[1], X.shape[0] - 1)
        X_pca = PCA(n_components=pca_dim,
                    random_state=config.RANDOM_SEED).fit_transform(X)
        X_2d = TSNE(n_components=2, random_state=config.RANDOM_SEED,
                    perplexity=min(30, len(X) // 3)).fit_transform(X_pca)

        for label_val, label_name, color in [
            (0, "Non-Hallucination", "steelblue"),
            (1, "Hallucination",     "tomato"),
        ]:
            mask = labels == label_val
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=color, label=label_name, alpha=0.6, s=20, edgecolors="none")

        ax.set_title(f"Layer {layer_idx}", fontsize=12)
        ax.legend(fontsize=9)
        ax.axis("off")

    fig.suptitle(f"t-SNE Hidden States [{strategy}] — Hallucination vs Non-Hallucination",
                 fontsize=13)
    plt.tight_layout()

    if save:
        path = os.path.join(save_dir, f"tsne_{strategy}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")

    plt.show()
    plt.close()


# Figure 3: Token Position Comparison vis.
def plot_token_position_comparison(
    results_by_position: Dict[str, List[Dict]],
    save: bool = True,
    figure_dir: Optional[str] = None,
) -> None:
    plot_layer_accuracy(
        results_by_position,
        save=save,
        filename="token_position_comparison.png",
        figure_dir=figure_dir,
    )

# Figure 4: Probe AUROC at positions
def _probe_auroc_at_positions(
    model, tokenizer,
    cases: List[Dict],
    labels: np.ndarray,
    token_positions: str,
    layer_index: int,
) -> float:
    """
    Extract hidden state from each layer index from given token positions.
    By using the Logistic Regression probe trained, plot AUROC curve at each token position.
    """
    device = next(model.parameters()).device
    model.eval()
    
    all_vecs = []
    all_labels = []
    
    for item, label in zip(cases, labels):
        prompt = item["prompt_w_context"]
        answer = item["ans_w_context"]

        # Token offset for answer tokens (assuming answer tokens are at the end of the sequence)
        context_start = len(tokenizer("Context: ", add_special_tokens=False)["input_ids"])
        context_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        context_end = context_start + context_len
        
        # Question offset
        ctx_text = item["context"]
        prefix_context = f"Context: {ctx_text}\n"
        q_start = len(tokenizer(prefix_context + "Question: ", add_special_tokens=False)["input_ids"])
        q_len = len(tokenizer(item["question"], add_special_tokens=False)["input_ids"])
        q_end = q_start + q_len
        
        full_text = prompt + " " + answer
        inputs = tokenizer(full_text, return_tensors="pt").to(device)
        seq_len = inputs.input_ids.shape[1]
        
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        answer_start = seq_len - len(answer_ids) # 확인해보고 max(0, seq_len - len(answer_ids))로 수정할 수도 있음
        answer_end = seq_len
        
        # NEED TO CHECK: Depends on what model we use, BOS token may different
        bos_offset = 1 if inputs.input_ids[0, 0].item() in tokenizer.all_special_ids else 0
        
        pos_map = {
            "context_start": context_start + bos_offset,
            "context_end": context_end + bos_offset - 1,
            "question_start": q_start + bos_offset,
            "question_end": q_end + bos_offset - 1,
            "answer_colon": answer_start + q_end, # "Answer:(colon) "
            "answer_start": answer_start,
            "answer_end": answer_end - 1,
        }
        tok_pos = pos_map.get(token_positions, answer_start)
        tok_pos = max(0, min(tok_pos, seq_len - 1))  # Ensure within bounds
        
        try:
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_state = outputs.hidden_states[layer_index][0, tok_pos, :].cpu().numpy()
                all_vecs.append(hidden_state)
                all_labels.append(label)
        except Exception as e:
            print(f"[WARN] Failed to extract hidden state for position '{token_positions}' in sample: {e}")
            continue
        
    if len(all_vecs) == 0:
        print(f"[ERROR] No valid hidden states extracted for position '{token_positions}'. Skipping AUROC calculation.")
        return float('nan')
    
    X = np.array(all_vecs)
    y = np.array(all_labels)
    
    if len(np.unique(y)) < 2:
        print(f"[ERROR] Only one class present in labels for position '{token_positions}'. Cannot compute AUROC.")
        return float('nan')
    
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.PROBE_TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y
    )
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    clf = LogisticRegression(
        C=1.0,
        max_iter=config.PROBE_MAX_ITER,
        random_state=config.RANDOM_SEED,
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    
    return float(roc_auc_score(y_test, proba))

# Figure 5: AUROC curve for multi position
def plot_multi_position_auroc(
    auroc_by_position: Dict[str, List[float]],
    layer_indices: Optional[List[int]] = None,
    save: bool = True,
    filename: str = "multi_position_auroc.png",
    figure_dir: Optional[str] = None,
) -> None:
    """
    여러 token position의 layer-wise AUROC를 한 그래프에 overlay.
    heatmap의 각 열(position)을 하나의 선으로 그려서 피크 위치 비교.
    """
    save_dir = figure_dir if figure_dir is not None else config.FIGURE_DIR
    os.makedirs(save_dir, exist_ok=True)

    colors = ["steelblue", "tomato", "seagreen", "orange", "mediumpurple", "brown", "pink"]

    fig, ax = plt.subplots(figsize=(12, 5))

    for (pos_name, aurocs), color in zip(auroc_by_position.items(), colors):
        layers = layer_indices if layer_indices is not None else list(range(len(aurocs)))
        ax.plot(layers, aurocs, label=pos_name, color=color,
                linewidth=2, marker="o", markersize=3)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("Layer-wise AUROC by Token Position — Hallucination Probe", fontsize=13)
    ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    plt.tight_layout()

    if save:
        path = os.path.join(save_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")

    plt.show()
    plt.close()


# Figure 6: Samples (same number of hallucination / non-hallucination cases) Probe ACC at positions
def plot_sampled_position_accuracy(
    results_by_position: Dict[str, List[Dict]],
    save: bool = True,
    filename: str = "sampled_position_accuracy.png",
    figure_dir: Optional[str] = None,
) -> None:
    """
    Hallucination / Non-Hallucination 라벨을 동일한 수로 샘플링 한 결과에 대해서
    각 token position에서 probe accuracy를 비교하는 그래프.

    results_by_position: {position_name: [{layer, accuracy, auroc}, ...]}
    """
    save_dir = figure_dir if figure_dir is not None else config.FIGURE_DIR
    os.makedirs(save_dir, exist_ok=True)

    colors = ["steelblue", "tomato", "seagreen", "orange", "mediumpurple", "brown", "pink"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric in zip(axes, ["accuracy", "auroc"]):
        for (pos_name, results), color in zip(results_by_position.items(), colors):
            layers = [r["layer"] for r in results]
            values = [r[metric] for r in results]
            ax.plot(layers, values, label=pos_name, color=color,
                    linewidth=2, marker="o", markersize=3)

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(
            f"Sampled Layer-wise {metric.upper()} by Position\n"
            "(Balanced Hallucination / Non-Hallucination)",
            fontsize=12,
        )
        ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    plt.tight_layout()

    if save:
        path = os.path.join(save_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")

    plt.show()
    plt.close()


# Figure 7: Plot token layer heatmap
def plot_token_layer_heatmap(
    model, tokenizer,
    cases: List[Dict],
    labels: np.ndarray,
    layer_stride: int = 2,
    token_positions: Optional[List[str]] = None,
    save: bool = True,
    filename: str = "token_layer_heatmap.png",
    figure_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Token Position × Layer AUROC Heatmap.

    X축: 토큰 위치 (ctx_first, ctx_last, q_first, q_last, answer_colon, ans_first, ans_last)
    Y축: 레이어 (0 ~ L-1, stride로 샘플링)
    값:  probe AUROC

    이 heatmap 하나로 "어느 토큰 위치 × 어느 레이어에서 환각 신호가 가장 강한가"를
    시각적으로 확인할 수 있음.

    Returns:
        auroc_matrix: (num_layers_sampled, num_positions) ndarray
    """
    save_dir = figure_dir if figure_dir is not None else config.FIGURE_DIR
    os.makedirs(save_dir, exist_ok=True)

    if token_positions is None:
        token_positions = [
            "context_start", "context_end",
            "question_start", "question_end",
            "answer_colon", "answer_start", "answer_end",
        ]

    num_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    sampled_layers = list(range(0, num_layers, layer_stride))

    auroc_matrix = np.full((len(sampled_layers), len(token_positions)), np.nan)

    for li, layer_idx in enumerate(tqdm(sampled_layers, desc="Token×Layer AUROC")):
        for pi, pos_name in enumerate(token_positions):
            auroc = _probe_auroc_at_positions(
                model, tokenizer, cases, labels, pos_name, layer_idx
            )
            auroc_matrix[li, pi] = auroc

    # Plot heatmap
    fig_h = max(6, len(sampled_layers) * 0.35 + 2)
    fig_w = len(token_positions) * 1.6 + 2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(auroc_matrix, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1.0)
    plt.colorbar(im, ax=ax, label="AUROC")

    ax.set_xticks(range(len(token_positions)))
    ax.set_xticklabels(token_positions, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(sampled_layers)))
    ax.set_yticklabels([str(l) for l in sampled_layers], fontsize=8)
    ax.set_xlabel("Token Position", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(
        "Token Position × Layer AUROC Heatmap\n(Hallucination Probe)",
        fontsize=13,
    )

    plt.tight_layout()

    if save:
        path = os.path.join(save_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")

    plt.show()
    plt.close()

    return auroc_matrix


# Phase 8: Attention to context
def analyze_attention_to_context(
    model, tokenizer,
    cases: List[Dict],
    figure_dir: str,
    max_samples: Optional[int] = None,
    layer_indices: Optional[List[int]] = None,
) -> None:
    """
    Analyze attention weight ratio to context tokens.

    For each sample, at the answer token position:
      ratio = sum(attn[:, answer_pos, context_range]) /
              sum(attn[:, answer_pos, :])

    Compares label 0 (non-hallucination) vs label 1 (hallucination).
    Hypothesis: label 0 attends more to context when generating the answer.

    Uses output_attentions=True. Averaged across attention heads per layer.
    """
    os.makedirs(figure_dir, exist_ok=True)

    if hasattr(model, "set_attn_implementation"):
        try:
            model.set_attn_implementation("eager")
            print("[Phase 8] Attention backend set to eager for attention extraction.")
        except Exception as e:
            print(f"[WARN] Failed to switch attention backend to eager: {e}")

    model.eval()
    device = next(model.parameters()).device
    num_layers = model.config.num_hidden_layers

    if layer_indices is None:
        layer_indices = list(range(num_layers))

    cases_filtered = [c for c in cases if c.get("label") in [0, 1]]
    rng = np.random.RandomState(config.RANDOM_SEED)
    if max_samples is not None and len(cases_filtered) > max_samples:
        idxs = rng.choice(len(cases_filtered), max_samples, replace=False)
        cases_filtered = [cases_filtered[i] for i in idxs]

    ratios_by_label = {0: [[] for _ in range(num_layers)],
                       1: [[] for _ in range(num_layers)]}

    for item in tqdm(cases_filtered, desc="Attention analysis"):
        label  = item["label"]
        prompt = item["prompt_w_context"]
        answer = item["ans_w_context"]

        ctx_start = len(tokenizer("Context: ", add_special_tokens=False).input_ids)
        ctx_len   = len(tokenizer(item["context"], add_special_tokens=False).input_ids)
        ctx_end   = ctx_start + ctx_len

        full_text = prompt + " " + answer
        inputs = tokenizer(full_text, return_tensors="pt").to(device)
        seq_len = inputs.input_ids.shape[1]

        answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
        answer_pos = seq_len - len(answer_ids)
        answer_pos = max(0, min(answer_pos, seq_len - 1))

        try:
            with torch.no_grad():
                outputs = model(
                    **inputs,
                    output_attentions=True,
                    use_cache=False,
                    return_dict=True,
                )
        except Exception:
            continue

        if outputs.attentions is None:
            raise RuntimeError(
                "Attention tensors were not returned. "
                "The model may still be using an attention backend that does not expose attentions."
            )

        for li, attn in enumerate(outputs.attentions):
            attn_np = attn[0].float().cpu().numpy()  # (heads, seq, seq)
            attn_row = attn_np[:, answer_pos, :]     # (heads, seq)
            total = attn_row.sum(axis=1)             # (heads,)
            ctx_end_clip = min(ctx_end, seq_len)
            to_ctx = attn_row[:, ctx_start:ctx_end_clip].sum(axis=1)
            ratio = (to_ctx / (total + 1e-9)).mean()
            ratios_by_label[label][li].append(float(ratio))

    layers = list(range(num_layers))
    mean0 = [np.mean(ratios_by_label[0][li]) if ratios_by_label[0][li] else 0.0 for li in layers]
    mean1 = [np.mean(ratios_by_label[1][li]) if ratios_by_label[1][li] else 0.0 for li in layers]
    std0  = [np.std(ratios_by_label[0][li])  if ratios_by_label[0][li] else 0.0 for li in layers]
    std1  = [np.std(ratios_by_label[1][li])  if ratios_by_label[1][li] else 0.0 for li in layers]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(layers, mean0, color="steelblue", linewidth=2,
            marker="o", markersize=3, label="Non-Hallucination (label 0)")
    ax.fill_between(layers,
                    [m - s for m, s in zip(mean0, std0)],
                    [m + s for m, s in zip(mean0, std0)],
                    color="steelblue", alpha=0.2)
    ax.plot(layers, mean1, color="tomato", linewidth=2,
            marker="o", markersize=3, label="Hallucination (label 1)")
    ax.fill_between(layers,
                    [m - s for m, s in zip(mean1, std1)],
                    [m + s for m, s in zip(mean1, std1)],
                    color="tomato", alpha=0.2)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Attention ratio to context", fontsize=12)
    ax.set_title(
        f"Attention to Context — Label 0 vs 1\n"
        f"(answer token position, {len(cases_filtered)} samples)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(figure_dir, "attention_to_context.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close()