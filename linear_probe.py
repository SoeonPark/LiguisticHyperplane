import os
import json
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import config

from Typing import Dict, List, Tuple, Optional, Any

def train_probe_per_layer(hidden_states: Dict[str, torch.Tensor], labels: List[int]) -> Dict[str, Dict[str, Any]]:
    N, num_layers, hidden_size = hidden_states.shape # In common for all cases
    print(f"Total Data: {N} samples, {num_layers} layers, Hidden size: {hidden_size}")
    print(f"Hidden states shape: {hidden_states.shape}, Labels length: {len(labels)}")

    for layer_name in hidden_states.keys():
        layer_data = hidden_states[layer_name]
        print(f"Layer: {layer_name}, Data shape: {layer_data.shape}")

    idx = np.arange(N)
    train_idx, test_idx = train_test_split(
        idx,
        test_size=config.PROBE_TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=labels,
    )

    results = []

    for layer_idx in tqdm(range(num_layers), desc="Training probes"):
        X = hidden_states[:, layer_idx, :]   # (N, hidden_dim)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # Linear Probe
        clf = LogisticRegression(
            max_iter=config.PROBE_MAX_ITER,
            random_state=config.RANDOM_SEED,
            C=1.0,
        )
        clf.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc  = accuracy_score(y_test,  clf.predict(X_test))

        proba     = clf.predict_proba(X_test)[:, 1]
        auroc     = roc_auc_score(y_test, proba)

        results.append({
            "layer":     layer_idx,
            "accuracy":  round(test_acc,  4),
            "auroc":     round(auroc,     4),
            "train_acc": round(train_acc, 4),
        })

    return results

def save_probe_results(results: list, strategy: str):
    os.makedirs(config.PROBE_RESULT_DIR, exist_ok=True)
    path = os.path.join(config.PROBE_RESULT_DIR, f"probe_{strategy}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved probe results: {path}")


def load_probe_results(strategy: str) -> list:
    path = os.path.join(config.PROBE_RESULT_DIR, f"probe_{strategy}.json")
    with open(path) as f:
        return json.load(f)


def print_summary(results: list):
    best = max(results, key=lambda x: x["auroc"])
    print(f"\n{'Num Layer':>6}  {'Accuracy':>9}  {'AUROC':>7}  {'Train Acc':>10}")
    print("-" * 40)
    for r in results:
        marker = " ◀ best" if r["layer"] == best["layer"] else ""
        print(f"{r['layer']:>6}  {r['accuracy']:>9.4f}  {r['auroc']:>7.4f}  {r['train_acc']:>10.4f}{marker}")
    print(f"\nBest Layer: {best['layer']}  AUROC={best['auroc']:.4f}  Acc={best['accuracy']:.4f}")