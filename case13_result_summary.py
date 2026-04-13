"""
case13_result_summary.py

Aggregate Case 1 vs Case 3 additional experiment outputs into a CSV summary
and a lightweight comparison plot.
"""

from __future__ import annotations

import csv
import glob
import os
from typing import Dict, List

import matplotlib.pyplot as plt

from extended_experiment_utils import load_json


def _collect_sampled_probe_rows(probe_dir: str) -> List[Dict]:
    rows: List[Dict] = []
    for path in sorted(glob.glob(os.path.join(probe_dir, "sampled_probe_*.json"))):
        payload = load_json(path)
        best = max(payload["results"], key=lambda item: item["auroc"])
        rows.append(
            {
                "experiment_type": "sampled_probe",
                "strategy": payload["strategy"],
                "model_type": "",
                "sampling_scope": payload["sampling_scope"],
                "sampling_method": payload["sampling_method"],
                "best_layer": int(best["layer"]),
                "best_epoch": "",
                "accuracy": float(best["accuracy"]),
                "auroc": float(best["auroc"]),
                "result_path": path,
            }
        )
    return rows


def _collect_sequence_rows(sequence_dir: str) -> List[Dict]:
    rows: List[Dict] = []
    for path in sorted(glob.glob(os.path.join(sequence_dir, "sequence_*.json"))):
        payload = load_json(path)
        test_metrics = payload["test_metrics"]
        rows.append(
            {
                "experiment_type": "sequence_model",
                "strategy": payload["strategy"],
                "model_type": payload["model_type"],
                "sampling_scope": payload["sampling_scope"],
                "sampling_method": payload["sampling_method"],
                "best_layer": "",
                "best_epoch": int(payload["best_epoch"]),
                "accuracy": float(test_metrics["accuracy"]),
                "auroc": float(test_metrics["auroc"]),
                "result_path": path,
            }
        )
    return rows


def save_summary_csv(rows: List[Dict], summary_dir: str) -> str:
    os.makedirs(summary_dir, exist_ok=True)
    path = os.path.join(summary_dir, "case13_results_summary.csv")
    fieldnames = [
        "experiment_type",
        "strategy",
        "model_type",
        "sampling_scope",
        "sampling_method",
        "best_layer",
        "best_epoch",
        "accuracy",
        "auroc",
        "result_path",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary CSV -> {path}")
    return path


def save_summary_plot(rows: List[Dict], summary_dir: str) -> str | None:
    if not rows:
        print("[skip] No result files found for summary plot.")
        return None

    probe_rows = [row for row in rows if row["experiment_type"] == "sampled_probe"]
    sequence_rows = [row for row in rows if row["experiment_type"] == "sequence_model"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    if probe_rows:
        labels = [f"{row['strategy']}\n{row['sampling_scope']}" for row in probe_rows]
        values = [row["auroc"] for row in probe_rows]
        axes[0].bar(labels, values, color="steelblue", alpha=0.85)
        axes[0].set_title("Best AUROC by Strategy\nSampled Layer Probe")
        axes[0].set_ylabel("AUROC")
        axes[0].set_ylim(0.0, 1.0)
        axes[0].grid(True, axis="y", alpha=0.3)
    else:
        axes[0].set_title("No sampled probe results")
        axes[0].axis("off")

    if sequence_rows:
        labels = [
            f"{row['strategy']}\n{row['model_type']}\n{row['sampling_scope']}"
            for row in sequence_rows
        ]
        values = [row["auroc"] for row in sequence_rows]
        axes[1].bar(labels, values, color="tomato", alpha=0.85)
        axes[1].set_title("Test AUROC by Strategy/Model\nLayer Sequence Classifier")
        axes[1].set_ylabel("AUROC")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].grid(True, axis="y", alpha=0.3)
    else:
        axes[1].set_title("No sequence-model results")
        axes[1].axis("off")

    plt.tight_layout()
    path = os.path.join(summary_dir, "case13_results_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved summary plot -> {path}")
    return path


def build_case13_summary(probe_dir: str, sequence_dir: str, summary_dir: str) -> Dict[str, str | None]:
    rows = _collect_sampled_probe_rows(probe_dir)
    rows.extend(_collect_sequence_rows(sequence_dir))
    csv_path = save_summary_csv(rows, summary_dir)
    plot_path = save_summary_plot(rows, summary_dir)
    return {
        "csv_path": csv_path,
        "plot_path": plot_path,
    }
