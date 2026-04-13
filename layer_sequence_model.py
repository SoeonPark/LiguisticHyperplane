"""
layer_sequence_model.py

Evaluate whether cross-layer structure helps discriminate Case 1 vs Case 3 by
feeding all layer hidden states for a pooled answer token position into a
sequence classifier.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

import config
from extended_experiment_utils import (
    apply_label_balancing,
    label_distribution,
    load_json,
    save_json,
    set_random_seed,
    split_train_val_test_indices,
    subset_cases,
)

SEQUENCE_RESULT_CACHE_VERSION = 1


class IndexedTensorDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor, indices: np.ndarray):
        self.features = features
        self.labels = labels
        self.indices = torch.as_tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int):
        sample_idx = self.indices[idx]
        return self.features[sample_idx], self.labels[sample_idx]


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.input_norm = nn.LayerNorm(model_dim)
        self.encoder = nn.LSTM(
            input_size=model_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(self.input_proj(x))
        _, (hidden, _) = self.encoder(x)
        pooled = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        return self.classifier(self.dropout(pooled))


class SmallTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(
                f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.input_proj = nn.Linear(input_dim, model_dim)
        self.input_norm = nn.LayerNorm(model_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 2),
        )
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        x = self.input_norm(self.input_proj(x))
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        return self.classifier(self.dropout(pooled))


def build_sequence_model(
    model_type: str,
    input_dim: int,
    seq_len: int,
    model_config: Dict[str, Any],
) -> nn.Module:
    if model_type == "bilstm":
        return BiLSTMClassifier(
            input_dim=input_dim,
            model_dim=int(model_config["model_dim"]),
            hidden_dim=int(model_config["lstm_hidden_dim"]),
            num_layers=int(model_config["lstm_layers"]),
            dropout=float(model_config["dropout"]),
        )
    if model_type == "transformer":
        return SmallTransformerClassifier(
            input_dim=input_dim,
            seq_len=seq_len,
            model_dim=int(model_config["model_dim"]),
            num_heads=int(model_config["transformer_heads"]),
            num_layers=int(model_config["transformer_layers"]),
            ff_dim=int(model_config["transformer_ff_dim"]),
            dropout=float(model_config["dropout"]),
        )
    raise ValueError(f"Unknown model_type='{model_type}'")


def _build_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = IndexedTensorDataset(features, labels, indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _compute_metrics_from_logits(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()[:, 1]
    preds = logits.argmax(axis=-1)
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "auroc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan"),
    }
    return metrics


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_examples = 0
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(batch_x)
            loss = nn.functional.cross_entropy(logits, batch_y)

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size = int(batch_y.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_examples += batch_size
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(batch_y.detach().cpu().numpy())

    stacked_logits = np.concatenate(all_logits, axis=0)
    stacked_labels = np.concatenate(all_labels, axis=0)
    metrics = _compute_metrics_from_logits(stacked_logits, stacked_labels)
    metrics["loss"] = total_loss / max(total_examples, 1)
    return metrics


def train_layer_sequence_model(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    cases: Optional[List[Dict]],
    model_type: str,
    model_config: Dict[str, Any],
) -> Dict[str, Any]:
    set_random_seed(int(model_config["random_seed"]))

    sampling_scope = str(model_config["sampling_scope"])
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
            sampling_method=str(model_config["sampling_method"]),
            random_seed=int(model_config["random_seed"]),
        )

    working_hidden_states = hidden_states[dataset_idx]
    working_labels = labels[dataset_idx]
    working_cases = subset_cases(cases, dataset_idx)

    train_idx, val_idx, test_idx = split_train_val_test_indices(
        working_labels,
        cases=working_cases,
        test_size=float(model_config["test_size"]),
        val_size=float(model_config["val_size"]),
        random_seed=int(model_config["random_seed"]),
    )
    balanced_train_idx = train_idx
    if sampling_scope == "train":
        balanced_train_idx = apply_label_balancing(
            working_labels,
            train_idx,
            sampling_method=str(model_config["sampling_method"]),
            random_seed=int(model_config["random_seed"]),
        )

    split_summary = {
        "original_full": label_distribution(labels),
        "dataset_after_sampling": label_distribution(working_labels),
        "train_before_sampling": label_distribution(working_labels, train_idx),
        "train_after_sampling": label_distribution(working_labels, balanced_train_idx),
        "val": label_distribution(working_labels, val_idx),
        "test": label_distribution(working_labels, test_idx),
        "dataset_size_after_sampling": int(len(working_labels)),
        "train_size_before_sampling": int(len(train_idx)),
        "train_size_after_sampling": int(len(balanced_train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
    }

    features = torch.from_numpy(working_hidden_states.astype(np.float32, copy=False))
    targets = torch.from_numpy(working_labels.astype(np.int64, copy=False))

    batch_size = int(model_config["batch_size"])
    train_loader = _build_loader(features, targets, balanced_train_idx, batch_size, shuffle=True)
    val_loader = _build_loader(features, targets, val_idx, batch_size, shuffle=False)
    test_loader = _build_loader(features, targets, test_idx, batch_size, shuffle=False)

    num_layers = int(working_hidden_states.shape[1])
    hidden_dim = int(working_hidden_states.shape[2])
    model = build_sequence_model(
        model_type=model_type,
        input_dim=hidden_dim,
        seq_len=num_layers,
        model_config=model_config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )

    history: List[Dict[str, Any]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_auroc = float("-inf")
    best_val_metrics: Optional[Dict[str, float]] = None
    patience = int(model_config["patience"])
    patience_counter = 0

    for epoch in range(1, int(model_config["epochs"]) + 1):
        train_metrics = _run_epoch(model, train_loader, device, optimizer=optimizer)
        val_metrics = _run_epoch(model, val_loader, device) if len(val_idx) else train_metrics

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(float(train_metrics["loss"]), 6),
                "train_accuracy": round(float(train_metrics["accuracy"]), 6),
                "train_auroc": round(float(train_metrics["auroc"]), 6),
                "val_loss": round(float(val_metrics["loss"]), 6),
                "val_accuracy": round(float(val_metrics["accuracy"]), 6),
                "val_auroc": round(float(val_metrics["auroc"]), 6),
            }
        )

        score = float(val_metrics["auroc"])
        if score > best_val_auroc:
            best_val_auroc = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_val_metrics = val_metrics
            patience_counter = 0
        else:
            patience_counter += 1
            if len(val_idx) and patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    test_metrics = _run_epoch(model, test_loader, device)

    return {
        "cache_version": SEQUENCE_RESULT_CACHE_VERSION,
        "model_type": model_type,
        "sampling_method": model_config["sampling_method"],
        "sampling_scope": sampling_scope,
        "input_shape": {
            "num_samples": int(working_hidden_states.shape[0]),
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
        },
        "split_summary": split_summary,
        "config": dict(model_config),
        "best_epoch": int(best_epoch),
        "best_val_metrics": None
        if best_val_metrics is None
        else {
            key: round(float(value), 6)
            for key, value in best_val_metrics.items()
        },
        "test_metrics": {
            key: round(float(value), 6)
            for key, value in test_metrics.items()
        },
        "history": history,
    }


def _result_path(
    save_dir: str,
    strategy: str,
    model_type: str,
    sampling_method: str,
    sampling_scope: str,
) -> str:
    return os.path.join(
        save_dir,
        f"sequence_{model_type}_{strategy}_{sampling_scope}_{sampling_method}.json",
    )


def save_sequence_results(
    payload: Dict[str, Any],
    strategy: str,
    model_type: str,
    sampling_method: str,
    sampling_scope: str,
    save_dir: str,
) -> str:
    path = _result_path(save_dir, strategy, model_type, sampling_method, sampling_scope)
    to_save = dict(payload)
    to_save["strategy"] = strategy
    to_save["model_type"] = model_type
    to_save["sampling_method"] = sampling_method
    to_save["sampling_scope"] = sampling_scope
    save_json(to_save, path)
    print(f"Saved sequence-model results -> {path}")
    return path


def load_sequence_results(
    strategy: str,
    model_type: str,
    sampling_method: str,
    sampling_scope: str,
    save_dir: str,
) -> Dict[str, Any]:
    return load_json(_result_path(save_dir, strategy, model_type, sampling_method, sampling_scope))


def sequence_result_cache_is_current(
    strategy: str,
    model_type: str,
    sampling_method: str,
    sampling_scope: str,
    save_dir: str,
    expected_metadata: Dict[str, Any],
) -> Tuple[bool, str]:
    path = _result_path(save_dir, strategy, model_type, sampling_method, sampling_scope)
    if not os.path.exists(path):
        return False, f"missing sequence-model cache: {path}"

    try:
        payload = load_json(path)
    except Exception as exc:
        return False, f"failed to read sequence-model cache: {exc}"

    if payload.get("cache_version") != SEQUENCE_RESULT_CACHE_VERSION:
        return False, (
            "sequence-model cache version mismatch "
            f"({payload.get('cache_version')} != {SEQUENCE_RESULT_CACHE_VERSION})"
        )

    for key, expected_value in expected_metadata.items():
        if payload.get(key) != expected_value:
            return False, f"metadata mismatch for '{key}' ({payload.get(key)} != {expected_value})"
    return True, "ok"


def print_sequence_summary(payload: Dict[str, Any]) -> None:
    test_metrics = payload["test_metrics"]
    val_metrics = payload.get("best_val_metrics")
    split_summary = payload["split_summary"]

    print("\nSequence-model summary")
    print(f"  model type             : {payload['model_type']}")
    print(f"  sampling method        : {payload['sampling_method']}")
    print(f"  sampling scope         : {payload['sampling_scope']}")
    print(f"  best epoch             : {payload['best_epoch']}")
    print(f"  original full          : {split_summary['original_full']}")
    print(f"  dataset after sampling : {split_summary['dataset_after_sampling']}")
    print(f"  train before sampling  : {split_summary['train_before_sampling']}")
    print(f"  train after sampling   : {split_summary['train_after_sampling']}")
    print(f"  val                    : {split_summary['val']}")
    print(f"  test                   : {split_summary['test']}")
    if val_metrics is not None:
        print(
            f"  best val               : "
            f"loss={val_metrics['loss']:.4f}  "
            f"acc={val_metrics['accuracy']:.4f}  "
            f"auroc={val_metrics['auroc']:.4f}"
        )
    print(
        f"  test                   : "
        f"loss={test_metrics['loss']:.4f}  "
        f"acc={test_metrics['accuracy']:.4f}  "
        f"auroc={test_metrics['auroc']:.4f}"
    )
