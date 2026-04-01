"""
layer_analysis.py

Visualization tools for the Linguistic Hyperplane experiments.

Figures generated:
  1. Layer-wise Accuracy / AUROC curves  (for each token position strategy)
  2. t-SNE plots of hidden states at selected layers
  3. Token position comparison curves    (first vs mean vs last)
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import config


# ── Figure 1: Layer-wise Accuracy / AUROC ────────────────────────────────────

def plot_layer_accuracy(
    results_dict: Dict[str, List[Dict]],
    save: bool = True,
    filename: str = "layer_accuracy_curve.png",
) -> None:
    """
    Plot layer-wise accuracy and AUROC for one or more token strategies.

    Args:
        results_dict : {"strategy_name": [probe result dicts], ...}
                       e.g. {"first": [...], "mean": [...], "last": [...]}
    """
    os.makedirs(config.FIGURE_DIR, exist_ok=True)
    colors = ["steelblue", "tomato", "seagreen", "orange", "mediumpurple"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric in zip(axes, ["accuracy", "auroc"]):
        for (strategy, results), color in zip(results_dict.items(), colors):
            layers = [r["layer"]  for r in results]
            values = [r[metric]   for r in results]
            ax.plot(
                layers, values,
                label=strategy, color=color,
                linewidth=2, marker="o", markersize=3,
            )

        # Chance level baseline
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")

        # Go / no-go threshold for AUROC
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
        path = os.path.join(config.FIGURE_DIR, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")

    plt.show()
    plt.close()


# ── Figure 2: t-SNE Visualization ────────────────────────────────────────────

def plot_tsne(
    hidden_states: np.ndarray,        # (N, num_layers, hidden_dim)
    labels: np.ndarray,               # (N,)
    layer_indices: Optional[List[int]] = None,
    strategy: str = "first",
    save: bool = True,
) -> None:
    """
    Plot t-SNE projections of hidden states at selected layers.

    Visualizes whether hallucination (red) and non-hallucination (blue)
    samples cluster separately at different layers — directly testing
    whether the Linguistic Hyperplane exists.

    Args:
        layer_indices : which layers to visualize. Defaults to
                        [0, L//4, L//2, L-1] (early, mid-early, mid, final).
    """
    os.makedirs(config.FIGURE_DIR, exist_ok=True)

    num_layers = hidden_states.shape[1]
    if layer_indices is None:
        layer_indices = [0, num_layers // 4, num_layers // 2, num_layers - 1]

    fig, axes = plt.subplots(
        1, len(layer_indices),
        figsize=(5 * len(layer_indices), 4),
    )
    if len(layer_indices) == 1:
        axes = [axes]

    for ax, layer_idx in zip(axes, layer_indices):
        X = hidden_states[:, layer_idx, :]   # (N, hidden_dim)

        # PCA first for speed (t-SNE on 4096-dim is slow)
        pca_dim = min(50, X.shape[1], X.shape[0] - 1)
        X_pca   = PCA(n_components=pca_dim, random_state=config.RANDOM_SEED).fit_transform(X)

        X_2d = TSNE(
            n_components=2,
            random_state=config.RANDOM_SEED,
            perplexity=min(30, len(X) // 3),
        ).fit_transform(X_pca)

        for label_val, label_name, color in [
            (0, "Non-Hallucination", "steelblue"),
            (1, "Hallucination",     "tomato"),
        ]:
            mask = labels == label_val
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=color, label=label_name,
                alpha=0.6, s=20, edgecolors="none",
            )

        ax.set_title(f"Layer {layer_idx}", fontsize=12)
        ax.legend(fontsize=9)
        ax.axis("off")

    fig.suptitle(
        f"t-SNE Hidden States [{strategy}] — Hallucination vs Non-Hallucination",
        fontsize=13,
    )
    plt.tight_layout()

    if save:
        path = os.path.join(config.FIGURE_DIR, f"tsne_{strategy}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")

    plt.show()
    plt.close()


# ── Figure 3: Token Position Comparison ──────────────────────────────────────

def plot_token_position_comparison(
    results_by_position: Dict[str, List[Dict]],
    save: bool = True,
) -> None:
    """
    Compare layer-wise AUROC across different token position strategies.
    Used in Phase 4 to determine which token position is most discriminative.

    Args:
        results_by_position : {"first": [...], "mean": [...], "last": [...]}
    """
    plot_layer_accuracy(
        results_by_position,
        save=save,
        filename="token_position_comparison.png",
    )