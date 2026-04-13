"""
extended_experiment_utils.py

Shared helpers for additional Case 1 vs Case 3 experiments:
1. label-balanced sampling probes
2. across-layer sequence models
"""

from __future__ import annotations

import gc
import json
import os
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split

import config

ALL_STRATEGIES = ["first", "mean", "last"]


def make_model_short(model_name: str) -> str:
    short = model_name.split("/")[-1].lower()
    short = short.replace("meta-llama-", "llama").replace("meta-", "")
    short = short.replace("mistralai-", "").replace("microsoft-", "")
    return short


def build_paths(
    model_name: str,
    subset: str,
    data_split: str,
    max_samples: int | None,
    answer_types: List[str],
    tag: str,
) -> dict:
    short = make_model_short(model_name)
    split_s = "val" if data_split == "validation" else data_split
    sample_s = f"n{max_samples}" if max_samples is not None else "nall"
    type_s = "-".join(sorted(answer_types))
    gen_s = f"gen{config.MAX_NEW_TOKENS}"
    exp_name = f"{short}__{subset}_{split_s}__{sample_s}__types_{type_s}__{gen_s}"
    if tag:
        exp_name += f"__{tag}"
    base = os.path.join("outputs", exp_name)
    return {
        "base": base,
        "cases": os.path.join(base, "cases.json"),
        "cases_all": os.path.join(base, "cases_all.json"),
        "hs_dir": os.path.join(base, "hidden_states"),
        "probe_sampled_dir": os.path.join(base, "probe_results_sampled"),
        "sequence_dir": os.path.join(base, "sequence_results"),
        "summary_dir": os.path.join(base, "summary"),
        "log_dir": os.path.join(base, "logs"),
        "exp_name": exp_name,
    }


def strategies_to_run(strategy_arg: str) -> List[str]:
    return ALL_STRATEGIES if strategy_arg == "all" else [strategy_arg]


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def save_json(payload: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def load_json(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def label_distribution(
    labels: Sequence[int] | np.ndarray,
    indices: Optional[Sequence[int]] = None,
) -> Dict[str, int]:
    arr = np.asarray(labels)
    if indices is not None:
        arr = arr[np.asarray(indices, dtype=int)]
    vals, counts = np.unique(arr, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(vals, counts)}


def subset_cases(cases: Optional[List[Dict]], indices: Sequence[int]) -> Optional[List[Dict]]:
    if cases is None:
        return None
    idx = np.asarray(indices, dtype=int).tolist()
    return [cases[i] for i in idx]


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _can_stratify(values: Optional[Sequence]) -> bool:
    if values is None:
        return False
    counts = Counter(values)
    return len(counts) > 1 and min(counts.values()) >= 2


def build_strata(
    labels: Sequence[int] | np.ndarray,
    cases: Optional[List[Dict]] = None,
) -> Optional[np.ndarray]:
    label_arr = np.asarray(labels)
    label_only = np.asarray(label_arr, dtype=int)

    if cases is None:
        return label_only if _can_stratify(label_only.tolist()) else None

    answer_type_strata = np.asarray(
        [
            f"{cases[i].get('answer_type', 'unknown')}_{int(label_arr[i])}"
            for i in range(len(label_arr))
        ],
        dtype=object,
    )
    if _can_stratify(answer_type_strata.tolist()):
        return answer_type_strata
    if _can_stratify(label_only.tolist()):
        return label_only
    return None


def split_train_test_indices(
    labels: Sequence[int] | np.ndarray,
    cases: Optional[List[Dict]] = None,
    test_size: float = config.PROBE_TEST_SIZE,
    random_seed: int = config.RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(labels))
    strata = build_strata(labels, cases)
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_seed,
        stratify=strata,
    )
    return np.asarray(train_idx), np.asarray(test_idx)


def split_train_val_test_indices(
    labels: Sequence[int] | np.ndarray,
    cases: Optional[List[Dict]] = None,
    test_size: float = config.PROBE_TEST_SIZE,
    val_size: float = 0.1,
    random_seed: int = config.RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if val_size < 0.0 or val_size >= 1.0:
        raise ValueError(f"val_size must be in [0, 1), got {val_size}")

    idx = np.arange(len(labels))
    strata = build_strata(labels, cases)
    train_val_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_seed,
        stratify=strata,
    )

    if val_size == 0.0:
        return np.asarray(train_val_idx), np.array([], dtype=int), np.asarray(test_idx)

    relative_val_size = val_size / (1.0 - test_size)
    train_val_strata = None if strata is None else strata[train_val_idx]

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val_size,
        random_state=random_seed,
        stratify=train_val_strata,
    )
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def apply_label_balancing(
    labels: Sequence[int] | np.ndarray,
    indices: Sequence[int],
    sampling_method: str = "undersample",
    random_seed: int = config.RANDOM_SEED,
) -> np.ndarray:
    idx = np.asarray(indices, dtype=int)
    if sampling_method == "none":
        return idx
    if sampling_method != "undersample":
        raise ValueError(
            f"Unknown sampling_method='{sampling_method}'. "
            "Choose from ['none', 'undersample']."
        )

    label_arr = np.asarray(labels)
    grouped: Dict[int, np.ndarray] = {}
    for label in np.unique(label_arr[idx]):
        grouped[int(label)] = idx[label_arr[idx] == label]

    if set(grouped.keys()) != {0, 1}:
        raise ValueError(
            f"Balanced sampling requires both labels in the split, got {sorted(grouped.keys())}"
        )

    target = min(len(grouped[0]), len(grouped[1]))
    rng = np.random.default_rng(random_seed)

    sampled_parts = []
    for label in (0, 1):
        sampled = rng.choice(grouped[label], size=target, replace=False)
        sampled_parts.append(sampled)

    return np.sort(np.concatenate(sampled_parts))
