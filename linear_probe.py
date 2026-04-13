"""
linear_probe.py

Train a Logistic Regression (linear probe) per layer and evaluate whether
hallucination vs non-hallucination hidden states are linearly separable.

This is the core R1 verification experiment:
  IF AUROC >> 0.5 at some layer -> Linguistic Hyperplane exists.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import config

# Set by main.py based on --balanced flag
BALANCED: bool = False
PROBE_CACHE_VERSION = 1


# ── Core Probe Training ───────────────────────────────────────────────────────

def train_probe_per_layer(
    hidden_states: np.ndarray,   # (N, num_layers, hidden_dim)
    labels: np.ndarray,          # (N,)
    cases: Optional[List[Dict]] = None,
) -> List[Dict[str, Any]]:
    """
    Train one Logistic Regression probe per layer and record metrics.

    The same train/test split is reused across all layers so that
    layer-wise comparisons are fair.

    Returns:
        List of dicts, one per layer:
        {
            "layer"    : int,
            "accuracy" : float,   # test accuracy
            "auroc"    : float,   # test AUROC
            "train_acc": float,   # train accuracy (overfitting check)
        }
    """
    breakpoint()
    N, num_layers, hidden_dim = hidden_states.shape
    print(f"\nProbe training: {N} samples | {num_layers} layers | hidden_dim={hidden_dim}")
    print(f"Label distribution: "
          f"non-hallucination(0)={int((labels == 0).sum())}  "
          f"hallucination(1)={int((labels == 1).sum())}")

    # Fixed train/test split — same across all layers
    idx = np.arange(N)
    # strata = [f"{cases[i]['answer_type']}_{labels[i]}" for i in range(N)]
    if cases is not None:
        strata = [f"{cases[i]['answer_type']}_{labels[i]}" for i in range(N)]
    else:
        strata = labels
    
    breakpoint()
    train_idx, test_idx = train_test_split(
        idx,
        test_size=config.PROBE_TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=strata,
    )
    breakpoint()

    results = []

    for layer_idx in tqdm(range(num_layers), desc="Training per-layer probes"):
        X = hidden_states[:, layer_idx, :]   # (N, hidden_dim)
        breakpoint()
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # Standardize per layer independently
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
        # breakpoint()
        
        if layer_idx == 0:
            print(f"\n[Debug] Layer {layer_idx} | "
                  f"X_train mean={X_train.mean():.4f}, std={X_train.std():.4f} | "
                  f"X_test mean={X_test.mean():.4f}, std={X_test.std():.4f}")

        # clf: Logistic Regression Classifier with optional class balancing
        clf = LogisticRegression( 
            C=1.0,
            max_iter=config.PROBE_MAX_ITER,
            random_state=config.RANDOM_SEED,
            solver="lbfgs",
            class_weight="balanced" if BALANCED else None,
        )
        breakpoint()
        clf.fit(X_train, y_train)
        breakpoint()

        if layer_idx == 0:
            print(f"\n[Debug] Layer {layer_idx} | "
                  f"Class distribution in train: {np.bincount(y_train)} | "
                  f"Class distribution in test: {np.bincount(y_test)}")
            
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc  = accuracy_score(y_test,  clf.predict(X_test))
        proba = clf.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, proba)

        results.append({
            "layer":     layer_idx,
            "accuracy":  round(float(test_acc),  4),
            "auroc":     round(float(auroc),     4),
            "train_acc": round(float(train_acc), 4),
        })

    return results


# ── Save / Load ───────────────────────────────────────────────────────────────

def _probe_paths(save_dir: str, strategy: str) -> Tuple[str, str]:
    result_path = os.path.join(save_dir, f"probe_{strategy}.json")
    meta_path = os.path.join(save_dir, f"probe_{strategy}.meta.json")
    return result_path, meta_path


def _build_probe_metadata(
    results: List[Dict],
    strategy: str,
    metadata: Optional[Dict] = None,
) -> Dict:
    meta = {
        "cache_version": PROBE_CACHE_VERSION,
        "strategy": strategy,
        "num_layers": int(len(results)),
    }
    if metadata:
        meta.update(metadata)
    return meta

def save_probe_results(
    results: List[Dict],
    strategy: str,
    probe_dir: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> None:
    save_dir = probe_dir if probe_dir is not None else config.PROBE_RESULT_DIR
    os.makedirs(save_dir, exist_ok=True)
    path, meta_path = _probe_paths(save_dir, strategy)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            _build_probe_metadata(results, strategy, metadata),
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    print(f"Saved probe results -> {path}")
    print(f"Saved probe metadata -> {meta_path}")


def load_probe_results(strategy: str, probe_dir: Optional[str] = None) -> List[Dict]:
    load_dir = probe_dir if probe_dir is not None else config.PROBE_RESULT_DIR
    path, _ = _probe_paths(load_dir, strategy)
    with open(path) as f:
        return json.load(f)


def load_probe_metadata(strategy: str, probe_dir: Optional[str] = None) -> Dict:
    load_dir = probe_dir if probe_dir is not None else config.PROBE_RESULT_DIR
    _, meta_path = _probe_paths(load_dir, strategy)
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def probe_cache_is_current(
    strategy: str,
    probe_dir: Optional[str] = None,
    expected_metadata: Optional[Dict] = None,
) -> Tuple[bool, str]:
    load_dir = probe_dir if probe_dir is not None else config.PROBE_RESULT_DIR
    path, meta_path = _probe_paths(load_dir, strategy)

    for file_path in (path, meta_path):
        if not os.path.exists(file_path):
            return False, f"missing probe cache file: {file_path}"

    try:
        meta = load_probe_metadata(strategy, probe_dir=load_dir)
    except Exception as e:
        return False, f"failed to read probe metadata: {e}"

    if meta.get("cache_version") != PROBE_CACHE_VERSION:
        return False, (
            f"probe cache version mismatch "
            f"({meta.get('cache_version')} != {PROBE_CACHE_VERSION})"
        )
    if meta.get("strategy") != strategy:
        return False, f"strategy mismatch in probe cache ({meta.get('strategy')} != {strategy})"

    if expected_metadata:
        for key, expected_value in expected_metadata.items():
            if meta.get(key) != expected_value:
                return False, f"metadata mismatch for '{key}' ({meta.get(key)} != {expected_value})"

    return True, "ok"


# ── Summary Printer ───────────────────────────────────────────────────────────

def print_summary(results: List[Dict]) -> None:
    """Print layer-wise metrics and highlight the best layer by AUROC."""
    best = max(results, key=lambda r: r["auroc"])

    print(f"\n{'Layer':>6}  {'Test Acc':>9}  {'AUROC':>7}  {'Train Acc':>10}")
    print("-" * 45)
    for r in results:
        marker = "  <- BEST" if r["layer"] == best["layer"] else ""
        print(
            f"{r['layer']:>6}  "
            f"{r['accuracy']:>9.4f}  "
            f"{r['auroc']:>7.4f}  "
            f"{r['train_acc']:>10.4f}"
            f"{marker}"
        )
    print(
        f"\nBest layer : {best['layer']}  "
        f"AUROC={best['auroc']:.4f}  "
        f"Acc={best['accuracy']:.4f}"
    )

    # Go / No-go decision
    if best["auroc"] >= 0.65:
        print("\n[GO] AUROC >= 0.65 -> Linguistic Hyperplane likely exists. Proceed.")
    else:
        print("\n[NO-GO] AUROC < 0.65 -> Linear separability not confirmed.")
