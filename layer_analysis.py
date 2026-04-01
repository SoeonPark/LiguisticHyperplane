import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import config


def plot_layer_accuracy(results_dict: dict, save: bool = True):
    """
    Args:
        results_dict: { strategy_name: [probe results list], ... }
            -> e.g., {"first": [...], "mean": [...], "last": [...]}
    """
    os.makedirs(config.FIGURE_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["steelblue", "tomato", "seagreen", "orange"]

    for ax, metric in zip(axes, ["accuracy", "auroc"]):
        for (strategy, results), color in zip(results_dict.items(), colors):
            layers  = [r["layer"]  for r in results]
            values  = [r[metric]   for r in results]
            ax.plot(layers, values, label=strategy, color=color, linewidth=2, marker="o", markersize=3)

        # chance level
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f"Layer-wise {metric.upper()} — Hallucination Probe", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    plt.tight_layout()
    path = os.path.join(config.FIGURE_DIR, "layer_accuracy_curve.png")
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()

def plot_tsne(hidden_states: np.ndarray, labels: np.ndarray,
              layer_indices: list = None, strategy: str = "first",
              save: bool = True):

    os.makedirs(config.FIGURE_DIR, exist_ok=True)

    num_layers = hidden_states.shape[1]
    if layer_indices is None:
        layer_indices = [0, num_layers // 4, num_layers // 2, num_layers - 1]

    fig, axes = plt.subplots(1, len(layer_indices), figsize=(5 * len(layer_indices), 4))
    if len(layer_indices) == 1:
        axes = [axes]

    colors = np.where(labels == 0, "steelblue", "tomato")

    for ax, layer_idx in zip(axes, layer_indices):
        X = hidden_states[:, layer_idx, :]   # (N, hidden_dim)

        pca_dim = min(50, X.shape[1], X.shape[0] - 1)
        X_pca   = PCA(n_components=pca_dim, random_state=config.RANDOM_SEED).fit_transform(X)

        X_2d = TSNE(
            n_components=2,
            random_state=config.RANDOM_SEED,
            perplexity=min(30, len(X) // 3),
        ).fit_transform(X_pca)

        for label_val, label_name, color in [(0, "Non-Hallucination", "steelblue"), (1, "Hallucination", "tomato")]:
            mask = labels == label_val
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=color, label=label_name, alpha=0.6, s=20, edgecolors="none")

        ax.set_title(f"Layer {layer_idx}", fontsize=12)
        ax.legend(fontsize=9)
        ax.axis("off")

    fig.suptitle(f"t-SNE of Hidden States [{strategy}] — Hallucination vs Non-Hallucination",
                 fontsize=13)
    plt.tight_layout()

    path = os.path.join(config.FIGURE_DIR, f"tsne_{strategy}.png")
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()


def plot_token_position_comparison(results_by_position: dict, save: bool = True):
    """
    Args:
        results_by_position: {"first": results, "mean": results, "last": results}
    """
    plot_layer_accuracy(results_by_position, save=save)