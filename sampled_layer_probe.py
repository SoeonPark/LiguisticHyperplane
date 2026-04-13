"""
sampled_layer_probe.py

Layer-wise Logistic Regression probe with optional label-balanced sampling on
the training split. This keeps the original per-layer probe experiment intact
while adding a directly comparable sampled variant.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import config
from extended_experiment_utils import (
    apply_label_balancing,
    label_distribution,
    load_json,
    save_json,
    split_train_test_indices,
    subset_cases,
)

SAMPLED_PROBE_CACHE_VERSION = 1


def train_sampled_probe_per_layer(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    cases: Optional[List[Dict]] = None,
    sampling_method: str = "undersample",
    sampling_scope: str = "train",
) -> Dict[str, Any]:
    n_samples, num_layers, hidden_dim = hidden_states.shape
    print(
        f"\nSampled probe training: {n_samples} samples | "
        f"{num_layers} layers | hidden_dim={hidden_dim}"
    )
    print(
        f"Full label distribution: "
        f"non-hallucination(0)={int((labels == 0).sum())}  "
        f"hallucination(1)={int((labels == 1).sum())}"
    )

    if sampling_scope not in {"train", "dataset"}:
        raise ValueError(
            f"Unknown sampling_scope='{sampling_scope}'. "
            "Choose from ['train', 'dataset']."
        )

    dataset_idx = np.arange(len(labels))
    if sampling_scope == "dataset":
        dataset_idx = apply_label_balancing(
            labels,
            dataset_idx,
            sampling_method=sampling_method,
            random_seed=config.RANDOM_SEED,
        )

    working_hidden_states = hidden_states[dataset_idx]
    working_labels = labels[dataset_idx]
    working_cases = subset_cases(cases, dataset_idx)

    train_idx, test_idx = split_train_test_indices(
        working_labels,
        cases=working_cases,
        test_size=config.PROBE_TEST_SIZE,
        random_seed=config.RANDOM_SEED,
    )
    balanced_train_idx = train_idx
    if sampling_scope == "train":
        balanced_train_idx = apply_label_balancing(
            working_labels,
            train_idx,
            sampling_method=sampling_method,
            random_seed=config.RANDOM_SEED,
        )

    split_summary = {
        "original_full": label_distribution(labels),
        "dataset_after_sampling": label_distribution(working_labels),
        "train_before_sampling": label_distribution(working_labels, train_idx),
        "train_after_sampling": label_distribution(working_labels, balanced_train_idx),
        "test": label_distribution(working_labels, test_idx),
        "dataset_size_after_sampling": int(len(working_labels)),
        "train_size_before_sampling": int(len(train_idx)),
        "train_size_after_sampling": int(len(balanced_train_idx)),
        "test_size": int(len(test_idx)),
    }

    results: List[Dict[str, Any]] = []

    for layer_idx in tqdm(range(num_layers), desc="Training sampled per-layer probes"):
        x = working_hidden_states[:, layer_idx, :]
        x_train = x[balanced_train_idx]
        x_test = x[test_idx]
        y_train = working_labels[balanced_train_idx]
        y_test = working_labels[test_idx]

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        clf = LogisticRegression(
            C=1.0,
            max_iter=config.PROBE_MAX_ITER,
            random_state=config.RANDOM_SEED,
            solver="lbfgs",
        )
        clf.fit(x_train, y_train)

        train_acc = accuracy_score(y_train, clf.predict(x_train))
        test_acc = accuracy_score(y_test, clf.predict(x_test))
        proba = clf.predict_proba(x_test)[:, 1]
        auroc = roc_auc_score(y_test, proba)

        results.append(
            {
                "layer": layer_idx,
                "accuracy": round(float(test_acc), 4),
                "auroc": round(float(auroc), 4),
                "train_acc": round(float(train_acc), 4),
            }
        )

    return {
        "cache_version": SAMPLED_PROBE_CACHE_VERSION,
        "sampling_method": sampling_method,
        "sampling_scope": sampling_scope,
        "split_summary": split_summary,
        "results": results,
        "probe_test_size": config.PROBE_TEST_SIZE,
        "probe_max_iter": config.PROBE_MAX_ITER,
        "random_seed": config.RANDOM_SEED,
        "num_samples": int(len(working_labels)),
        "num_layers": int(num_layers),
        "hidden_dim": int(hidden_dim),
    }


def _result_path(save_dir: str, strategy: str, sampling_method: str, sampling_scope: str) -> str:
    return os.path.join(
        save_dir,
        f"sampled_probe_{strategy}_{sampling_scope}_{sampling_method}.json",
    )


def save_sampled_probe_results(
    payload: Dict[str, Any],
    strategy: str,
    sampling_method: str,
    sampling_scope: str,
    probe_dir: str,
) -> str:
    path = _result_path(probe_dir, strategy, sampling_method, sampling_scope)
    to_save = dict(payload)
    to_save["strategy"] = strategy
    to_save["sampling_scope"] = sampling_scope
    to_save["sampling_method"] = sampling_method
    save_json(to_save, path)
    print(f"Saved sampled probe results -> {path}")
    return path


def load_sampled_probe_results(
    strategy: str,
    sampling_method: str,
    sampling_scope: str,
    probe_dir: str,
) -> Dict[str, Any]:
    return load_json(_result_path(probe_dir, strategy, sampling_method, sampling_scope))


def sampled_probe_cache_is_current(
    strategy: str,
    sampling_method: str,
    sampling_scope: str,
    probe_dir: str,
    expected_metadata: Dict[str, Any],
) -> Tuple[bool, str]:
    path = _result_path(probe_dir, strategy, sampling_method, sampling_scope)
    if not os.path.exists(path):
        return False, f"missing sampled probe cache: {path}"

    try:
        payload = load_json(path)
    except Exception as exc:
        return False, f"failed to read sampled probe cache: {exc}"

    if payload.get("cache_version") != SAMPLED_PROBE_CACHE_VERSION:
        return False, (
            "sampled probe cache version mismatch "
            f"({payload.get('cache_version')} != {SAMPLED_PROBE_CACHE_VERSION})"
        )

    for key, expected_value in expected_metadata.items():
        if payload.get(key) != expected_value:
            return False, f"metadata mismatch for '{key}' ({payload.get(key)} != {expected_value})"
    return True, "ok"


def print_summary(payload: Dict[str, Any]) -> None:
    results = payload["results"]
    split_summary = payload["split_summary"]
    best = max(results, key=lambda item: item["auroc"])

    print("\nSampling summary")
    print(f"  method                 : {payload['sampling_method']}")
    print(f"  scope                  : {payload['sampling_scope']}")
    print(f"  original full          : {split_summary['original_full']}")
    print(f"  dataset after sampling : {split_summary['dataset_after_sampling']}")
    print(f"  train before sampling  : {split_summary['train_before_sampling']}")
    print(f"  train after sampling   : {split_summary['train_after_sampling']}")
    print(f"  test                   : {split_summary['test']}")

    print(f"\n{'Layer':>6}  {'Test Acc':>9}  {'AUROC':>7}  {'Train Acc':>10}")
    print("-" * 45)
    for row in results:
        marker = "  <- BEST" if row["layer"] == best["layer"] else ""
        print(
            f"{row['layer']:>6}  "
            f"{row['accuracy']:>9.4f}  "
            f"{row['auroc']:>7.4f}  "
            f"{row['train_acc']:>10.4f}"
            f"{marker}"
        )

    print(
        f"\nBest layer : {best['layer']}  "
        f"AUROC={best['auroc']:.4f}  "
        f"Acc={best['accuracy']:.4f}"
    )
