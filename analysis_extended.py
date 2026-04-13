"""
analysis_extended.py

Extended analysis modules for the Linguisic Hyperplane proving.
Each function is a self-contained analysis that can be called independently.

Phases:
* probe_direction: probe coefficients -> vocabulary direction
* pca_analysis   : PCA top-k directions, label alignment
* cka            : layer-wise CKA between label 0 / label 1 group
* attention      : attention weight ratio to context tokens with visualization
* transformerLens: 
"""

import os
from typing import List, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import config

def _ascii_safe_for_plot(text: str) -> str:
    """
    Matplotlib's default DejaVu font cannot render every tokenizer-decoded
    symbol. Convert labels to an ASCII-safe representation for plotting while
    keeping the original text for console logs.
    """
    if text is None:
        return "<None>"
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    safe = text.encode("ascii", "backslashreplace").decode("ascii")
    return safe if safe else "<empty>"

def _load_probe_coef(probe_dir: str, strategy: str) -> Optional[np.ndarray]:
    """
    Re-train probe on ALL data (no split) for a single layer to extract coef.
    Returns coef vector (hidden_dim,) for the best AUROC layer.
    """
    import json
    result_path = os.path.join(probe_dir, f"probe_{strategy}.json")
    if not os.path.exists(result_path):
        print(f"[skip] Probe results not found: {result_path}")
        return None, None
    with open(result_path) as f:
        results = json.load(f)
    best = max(results, key=lambda r: r["auroc"])
    return best

def analyze_probe_direction(
    hidden_states: np.ndarray, # (num_samples, seq_len, hidden_dim)
    labels: np.ndarray,        # (num_samples,)
    model, tokenizer,
    probe_dir: str, strategy: str, figure_dir: str, top_k: int = 30
) -> None:
    """
    Probe direction analysis:
    
    1. Retrain LogisticRegression on each layer with ALL data
    2. Extract probe coefficient vector -> (hidden_dim,)
    3. Project onto lm_head.weight -> (vocab_size,) scores
    4. Find top-k tokens with highest positive scores, visualize with bar plot.
    
    Interpretation:
     Positive Direction; Tokens the model tends to output when hallucination. 
     Negative Direction; Tokens the model tends to output when NOT hallucination.
    """
    os.makedirs(figure_dir, exist_ok=True)
    
    breakpoint()
    import json
    result_path = os.path.join(probe_dir, f"probe_{strategy}.json")
    
    with open(result_path) as f:
        results = json.load(f)
    best_layer = max(results, key=lambda r: r["auroc"])["layer"]
    print(f"Best probe layer for strategy '{strategy}': {best_layer}")
    
    X = hidden_states[:, best_layer, :]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    breakpoint()
    clf = LogisticRegression(
        C=1.0,
        max_iter=config.PROBE_MAX_ITER,
        random_state=config.RANDOM_SEED,
        solver="lbfgs",
    )
    breakpoint()
    clf.fit(X_scaled, labels)
    breakpoint()
    
    # Extract probe coefficient vector
    direction = clf.coef_[0]  # (hidden_dim,)
    print(f"  Probe direction norm: {np.linalg.norm(direction):.4f}")
    
    # Project onto lm_head.weight
    try:
        lm_head_weight = model.lm_head.weight.detach().cpu().numpy()  # (vocab_size, hidden_dim)
    except AttributeError:
        print("[ERROR] Model does not have lm_head.weight. Skipping probe direction analysis.")
        return
    
    breakpoint()
    scores = lm_head_weight @ direction  # (vocab_size,)
    
    # Top-k tokens with highest positive scores
    hallu_ids = scores.argsort()[-top_k:][::-1]
    non_hallu_ids = scores.argsort()[:top_k]
    
    hallu_tokens = [(tokenizer.decode([idx]).strip(), float(scores[idx])) for idx in hallu_ids]
    non_hallu_tokens = [(tokenizer.decode([idx]).strip(), float(scores[idx])) for idx in non_hallu_ids]
    breakpoint()
    print(f"     Top-{top_k} tokens aligned with hallucination direction (+):")
    for token, score in hallu_tokens:
        print(f"       {token:15s} | score: {score:.4f}")
    print(f"     Top-{top_k} tokens aligned with non-hallucination direction (-):")
    for token, score in non_hallu_tokens:
        print(f"       {token:15s} | score: {score:.4f}")
        
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, tokens, title, color in [
        (axes[0], hallu_tokens, f"Top-{top_k} Tokens Aligned with Hallucination Direction (+)", "tomato"),
        (axes[1], non_hallu_tokens, f"Top-{top_k} Tokens Aligned with Non-Hallucination Direction (-)", "steelblue"),
    ]:
        toks = [_ascii_safe_for_plot(t) for t, _ in tokens[:20]]
        vals = [abs(s) for _, s in tokens[:20]]
        ax.barh(range(len(toks)), vals[::-1], color=color, alpha=0.8)
        ax.set_yticks(range(len(toks)))
        ax.set_yticklabels(toks[::-1], fontsize=9)
        ax.set_title(f"Layer {best_layer} [{strategy}]\n{title}", fontsize=11)
        ax.set_xlabel("Score magnitude", fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle(f"Probe Direction -> Vocabulary Projection\n"
                 f"Strategy={strategy}, Best Layer={best_layer}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(figure_dir, f"probe_direction_{strategy}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved to {path}")
    plt.close()
    
def analyze_pca(
    hidden_states: np.ndarray,   # (N, num_layers, hidden_dim)
    labels: np.ndarray,          # (N,)
    strategy: str,
    figure_dir: str,
    layer_indices: Optional[List[int]] = None,
) -> None:
    """
    PCA top-2 directions with label alignment.

    For each selected layer:
      - PCA(n=2) on ALL samples
      - Scatter plot colored by label (more interpretable than t-SNE)
      - Compute alignment score: how much PC1 correlates with labels
        (point-biserial correlation)

    Why PCA over t-SNE:
      - Linear -> directly interpretable as "variance directions"
      - PC1 alignment with label = evidence that the primary variance
        axis in hidden space correlates with hallucination state
    """
    from scipy.stats import pointbiserialr
    os.makedirs(figure_dir, exist_ok=True)

    num_layers = hidden_states.shape[1]
    if layer_indices is None:
        # Sample key layers: early, quarter, mid, three-quarter, final
        layer_indices = sorted(set([
            0, num_layers // 4, num_layers // 2,
            num_layers * 3 // 4, num_layers - 1
        ]))

    fig, axes = plt.subplots(2, len(layer_indices),
                             figsize=(5 * len(layer_indices), 9))
    if len(layer_indices) == 1:
        axes = axes.reshape(2, 1)

    alignment_scores = []

    for col, layer_idx in enumerate(layer_indices):
        X = hidden_states[:, layer_idx, :]
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        pca = PCA(n_components=2, random_state=config.RANDOM_SEED)
        X_2d = pca.fit_transform(X_s)

        # Scatter
        ax = axes[0, col]
        for lv, ln, color in [(0, "Non-Hall", "steelblue"), (1, "Hall", "tomato")]:
            mask = labels == lv
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=color, label=ln, alpha=0.5, s=10, edgecolors="none")
        ax.set_title(f"Layer {layer_idx}", fontsize=11)
        ax.legend(fontsize=8, markerscale=2)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=9)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=9)
        ax.grid(True, alpha=0.2)

        # PC1 alignment with labels
        r, p = pointbiserialr(labels, X_2d[:, 0])
        alignment_scores.append((layer_idx, r, p))

        # PC1 histogram by label
        ax2 = axes[1, col]
        for lv, ln, color in [(0, "Non-Hall", "steelblue"), (1, "Hall", "tomato")]:
            mask = labels == lv
            ax2.hist(X_2d[mask, 0], bins=40, color=color, alpha=0.6,
                     label=ln, density=True)
        ax2.set_xlabel("PC1 score", fontsize=9)
        ax2.set_ylabel("Density", fontsize=9)
        ax2.set_title(f"PC1 dist | r={r:.3f} p={p:.2e}", fontsize=10)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.2)

    fig.suptitle(f"PCA Analysis [{strategy}] — Label Alignment", fontsize=13)
    plt.tight_layout()
    path = os.path.join(figure_dir, f"pca_analysis_{strategy}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {path}")
    plt.close()

    # Print alignment table
    print(f"\n  PC1 <-> Label alignment (point-biserial r):")
    print(f"  {'Layer':>6}  {'r':>8}  {'p':>10}")
    print("  " + "-" * 30)
    for layer_idx, r, p in alignment_scores:
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        print(f"  {layer_idx:>6}  {r:>8.4f}  {p:>10.2e} {sig}")

    # Plot alignment curve
    fig2, ax = plt.subplots(figsize=(8, 4))
    # Compute for ALL layers
    all_rs = []
    for li in range(num_layers):
        X = hidden_states[:, li, :]
        pca = PCA(n_components=1, random_state=config.RANDOM_SEED)
        pc1 = pca.fit_transform(StandardScaler().fit_transform(X))[:, 0]
        r, _ = pointbiserialr(labels, pc1)
        all_rs.append(abs(r))

    ax.plot(range(num_layers), all_rs, color="steelblue", linewidth=2,
            marker="o", markersize=3)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("|PC1–Label correlation|", fontsize=12)
    ax.set_title(f"PC1 ↔ Label Alignment Curve [{strategy}]", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(figure_dir, f"pca_alignment_curve_{strategy}.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"Saved -> {path2}")
    plt.close()

    
def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA between two representation matrices.
    X, Y: (N, d) — same N, different d allowed.

    CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)

    Value in [0, 1]. 1 = identical geometry, 0 = orthogonal.
    """
    def _hsic(K, L):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return np.trace(K @ H @ L @ H) / (n - 1) ** 2

    K = X @ X.T
    L = Y @ Y.T
    hsic_kl = _hsic(K, L)
    hsic_kk = _hsic(K, K)
    hsic_ll = _hsic(L, L)
    if hsic_kk == 0 or hsic_ll == 0:
        return 0.0
    return float(hsic_kl / np.sqrt(hsic_kk * hsic_ll))

def analyze_cka(
    hidden_states: np.ndarray, # (num_samples, seq_len, hidden_dim)
    labels: np.ndarray,        # (num_samples,)
    figure_dir: str, strategy: str, max_per_class: Optional[int] = None
) -> None:
    """
    Layer-wise CKA between label-0 and label-1 groups.

    For each layer:
      - Take min(max_per_class, n_class) samples from each label
      - Compute linear CKA between the two groups' representation matrices
      - Low CKA -> the two groups live in different representational geometry
        -> evidence of different internal states

    This is complementary to AUROC:
      AUROC   = "can a linear boundary separate them?"
      CKA gap = "how geometrically different are the two groups?"
    """
    os.makedirs(figure_dir, exist_ok=True)
    num_layers = hidden_states.shape[1]

    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]

    if len(idx0) == 0 or len(idx1) == 0:
        raise ValueError("CKA requires at least one sample from each label.")

    if max_per_class is None:
        n = min(len(idx0), len(idx1))
    else:
        n = min(max_per_class, len(idx0), len(idx1))

    if n <= 0:
        raise ValueError("CKA sample size must be positive.")

    rng = np.random.RandomState(config.RANDOM_SEED)
    idx0_s = rng.choice(idx0, n, replace=False)
    idx1_s = rng.choice(idx1, n, replace=False)
    print(f"\n[Phase 7] CKA: using {n} samples per class")

    cka_scores = []
    for li in range(num_layers):
        X0 = hidden_states[idx0_s, li, :].astype(np.float32)
        X1 = hidden_states[idx1_s, li, :].astype(np.float32)

        # Standardize each group independently
        X0 = StandardScaler().fit_transform(X0)
        X1 = StandardScaler().fit_transform(X1)

        cka = _linear_cka(X0, X1)
        cka_scores.append(cka)
        if li % 8 == 0:
            print(f"  Layer {li:>2}: CKA = {cka:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(num_layers), cka_scores, color="seagreen",
            linewidth=2, marker="o", markersize=3)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="identical (1.0)")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Linear CKA", fontsize=12)
    ax.set_title(f"Layer-wise CKA: Label-0 vs Label-1 [{strategy}]\n"
                 f"(lower = more geometrically different)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    plt.tight_layout()
    path = os.path.join(figure_dir, f"cka_{strategy}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {path}")
    plt.close()

    # Print summary
    best_diff_layer = int(np.argmin(cka_scores))
    print(f"\n  Most geometrically different layer: {best_diff_layer} "
          f"(CKA={cka_scores[best_diff_layer]:.4f})")

def analyze_attention_to_context(
    model, tokenizer,
    cases: List[Dict],
    figure_dir: str, max_samples: Optional[int] = None, layer_indices: Optional[List[int]] = None
) -> None:
    """
    Analyze attention weight ratio to context tokens.
    
    For each case, compute the average attention weight to context tokens vs non-context tokens.
    Visualize the distribution of this ratio for label 0 vs label 1 with box plots or violin plots.
    """
    """
    Phase 8: Attention weight ratio to context tokens.

    For each sample, at the answer token position:
      ratio = sum(attn[:, answer_pos, context_range]) /
              sum(attn[:, answer_pos, :])

    Compares label 0 (non-hallucination) vs label 1 (hallucination).
    Hypothesis: label 0 attends more to context when generating the answer.

    Uses output_attentions=True — requires model to be loaded.
    Averaged across attention heads per layer.
    """
    import torch
    from tqdm import tqdm as _tqdm

    os.makedirs(figure_dir, exist_ok=True)

    if hasattr(model, "set_attn_implementation"):
        try:
            model.set_attn_implementation("eager")
            print("[Phase 8] Attention backend set to eager for attention extraction.")
        except Exception as e:
            print(f"[WARN] Failed to switch attention backend to eager: {e}")
    breakpoint()
    model.eval()
    breakpoint()
    device = next(model.parameters()).device
    num_layers = model.config.num_hidden_layers

    if layer_indices is None:
        layer_indices = list(range(num_layers))

    # Subsample
    cases13 = [c for c in cases if c["label"] in [0, 1]]
    rng = np.random.RandomState(config.RANDOM_SEED)
    if len(cases13) > max_samples:
        idxs = rng.choice(len(cases13), max_samples, replace=False)
        cases13 = [cases13[i] for i in idxs]

    ratios_by_label = {0: [[] for _ in range(num_layers)],
                       1: [[] for _ in range(num_layers)]}

    for item in _tqdm(cases13, desc="Attention analysis"):
        label  = item["label"]
        prompt = item["prompt_w_context"]
        answer = item["ans_w_context"]

        # Find context token range in prompt
        ctx_start = len(tokenizer("Context: ", add_special_tokens=False).input_ids)
        ctx_text  = item["context"]
        ctx_len   = len(tokenizer(ctx_text, add_special_tokens=False).input_ids)
        ctx_end   = ctx_start + ctx_len

        full_text = prompt + " " + answer
        inputs = tokenizer(full_text, return_tensors="pt").to(device)
        seq_len = inputs.input_ids.shape[1]

        # Answer token position (last token)
        answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
        answer_pos = seq_len - len(answer_ids)  # first answer token
        answer_pos = max(0, min(answer_pos, seq_len - 1))

        try:
            with torch.no_grad():
                outputs = model(
                    **inputs,
                    output_attentions=True,
                    use_cache=False,
                    return_dict=True,
                )
        except Exception as e:
            continue

        # attentions: tuple of (1, num_heads, seq_len, seq_len) per layer
        if outputs.attentions is None:
            raise RuntimeError(
                "Attention tensors were not returned. "
                "The model may still be using an attention backend that does not expose attentions."
            )

        for li, attn in enumerate(outputs.attentions):
            attn_np = attn[0].float().cpu().numpy()  # (heads, seq, seq)
            # attention FROM answer_pos TO context range
            attn_row = attn_np[:, answer_pos, :]  # (heads, seq)
            total = attn_row.sum(axis=1)  # (heads,)
            ctx_end_clip = min(ctx_end, seq_len)
            to_ctx = attn_row[:, ctx_start:ctx_end_clip].sum(axis=1)  # (heads,)
            ratio = (to_ctx / (total + 1e-9)).mean()
            ratios_by_label[label][li].append(float(ratio))

    # Compute mean ± std per layer per label
    layers = list(range(num_layers))
    mean0 = [np.mean(ratios_by_label[0][li]) if ratios_by_label[0][li] else 0
              for li in layers]
    mean1 = [np.mean(ratios_by_label[1][li]) if ratios_by_label[1][li] else 0
              for li in layers]
    std0  = [np.std(ratios_by_label[0][li])  if ratios_by_label[0][li] else 0
              for li in layers]
    std1  = [np.std(ratios_by_label[1][li])  if ratios_by_label[1][li] else 0
              for li in layers]

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
    ax.set_title(f"Attention to Context — Label 0 vs 1\n"
                 f"(answer token position, {max_samples} samples)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(figure_dir, "attention_to_context.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {path}")
    plt.close()
    
