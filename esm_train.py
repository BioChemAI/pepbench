#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train and evaluate a frozen ESM encoder with an MLP prediction head.

The script uses a fixed learning rate and preserves the same feature
extraction, validation, refit, and test procedure across datasets.

The script keeps the original frozen-feature workflow:
1. ESM extracts all peptide features into memory before MLP training.
2. The same in-memory ESM features are reused across random seeds.
3. Only the MLP prediction head is optimized; ESM parameters are never updated.

Selection protocol:
- seeds: 42, 43, 44 by default;
- classification: select the best epoch by validation F1;
- regression: select the best epoch by validation RMSE;
- the test set is not used for epoch selection;
- refit on train + validation for each seed,
  using that seed's selected epoch count, then evaluate the untouched test set;
- save three seed-specific final MLP checkpoints and full JSON/CSV records.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import platform
import random
import re
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from pephub.results import write_metrics_report

from pephub.dataset import PepDataset
from pephub.featurizer import PeptideFeaturizer
from pephub.splitter import split_dataset, split_dataset_by_similarity


DEFAULT_LEARNING_RATE = 1e-3
MODEL_NAME = "esm_mlp"
FEATURE_TYPE = "esm"


# =============================================================================
# MLP heads copied from the original ESM + MLP implementation
# =============================================================================


class MLPClassifier(nn.Module):
    """Binary MLP classifier that returns logits."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dims = [int(value) for value in hidden_dims]
        self.dropout = float(dropout)
        self.activation = str(activation)

        layers: List[nn.Module] = []
        previous_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(make_activation(self.activation))
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features).squeeze(-1)


class MLPRegressor(nn.Module):
    """MLP regressor that returns one scalar per sample."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dims = [int(value) for value in hidden_dims]
        self.dropout = float(dropout)
        self.activation = str(activation)

        layers: List[nn.Module] = []
        previous_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(make_activation(self.activation))
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features).squeeze(-1)


def make_activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def build_head(
    task_type: str,
    input_dim: int,
    hidden_dims: Sequence[int],
    dropout: float,
    activation: str,
    device: torch.device,
) -> nn.Module:
    if task_type == "classification":
        model: nn.Module = MLPClassifier(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
        )
    elif task_type == "regression":
        model = MLPRegressor(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
        )
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    return model.to(device)


# =============================================================================
# General serialization and persistence helpers
# =============================================================================


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return cleaned or "unnamed"


def compact_json(value: Any) -> str:
    return json.dumps(
        to_serializable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(item) for item in value]
    return value


def stable_id(prefix: str, payload: Mapping[str, Any], length: int = 24) -> str:
    normalized = compact_json(payload)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def atomic_json_dump(data: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=str(output_path.parent),
        text=True,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(to_serializable(data), handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)


def atomic_torch_save(data: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        torch.save(data, temp_path)
        os.replace(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)


def torch_load_compatible(path: Path, map_location: Any) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def atomic_dataframe_to_csv(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=str(output_path.parent),
        text=True,
    )
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        dataframe.to_csv(temp_path, index=False)
        os.replace(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)


def load_or_initialize_results(output_path: Path) -> Dict[str, Any]:
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        data.setdefault("schema_version", 1)
        data.setdefault("run_configs", {})
        data.setdefault("search_records", {})
        data.setdefault("learning_rate_summaries", {})
        data.setdefault("best_results", {})
        data.setdefault("dataset_failures", [])
        return data

    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
        },
        "run_configs": {},
        "search_records": {},
        "learning_rate_summaries": {},
        "best_results": {},
        "dataset_failures": [],
    }


def search_record_rows(results: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record_id, record in results.get("search_records", {}).items():
        rows.append(
            {
                "record_id": record_id,
                "dataset": record.get("dataset"),
                "task_type": record.get("task_type"),
                "split_method": record.get("split_method"),
                "similarity_threshold": record.get("similarity_threshold"),
                "model": MODEL_NAME,
                "random_seed": record.get("random_seed"),
                "learning_rate_head": record.get("learning_rate_head"),
                "best_epoch": record.get("best_epoch"),
                "validation_metrics_json": compact_json(
                    record.get("validation_metrics", {})
                )
                if record.get("validation_metrics")
                else "",
                "head_parameters_json": compact_json(
                    record.get("head_parameters", {})
                ),
                "training_parameters_json": compact_json(
                    record.get("training_parameters", {})
                ),
                "esm_parameters_json": compact_json(
                    record.get("esm_parameters", {})
                ),
                "split_sizes_json": compact_json(record.get("split_sizes", {})),
                "status": record.get("status"),
                "error_json": compact_json(record.get("error"))
                if record.get("error")
                else "",
                "device": record.get("device"),
                "gpu_name": record.get("gpu_name"),
                "started_at": record.get("started_at"),
                "finished_at": record.get("finished_at"),
            }
        )
    rows.sort(
        key=lambda row: (
            str(row.get("split_method")),
            str(row.get("dataset")),
            float(row.get("learning_rate_head") or 0.0),
            int(row.get("random_seed") or 0),
        )
    )
    return rows


def learning_rate_summary_rows(results: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for summary_id, summary in results.get("learning_rate_summaries", {}).items():
        rows.append(
            {
                "summary_id": summary_id,
                "dataset": summary.get("dataset"),
                "task_type": summary.get("task_type"),
                "split_method": summary.get("split_method"),
                "similarity_threshold": summary.get("similarity_threshold"),
                "model": MODEL_NAME,
                "learning_rate_head": summary.get("learning_rate_head"),
                "successful_seed_count": summary.get("successful_seed_count"),
                "expected_seeds_json": compact_json(summary.get("expected_seeds", [])),
                "validation_summary_json": compact_json(
                    summary.get("validation_summary", {})
                ),
                "selection_metric": summary.get("selection_metric"),
                "selection_direction": summary.get("selection_direction"),
                "status": summary.get("status"),
                "updated_at": summary.get("updated_at"),
            }
        )
    rows.sort(
        key=lambda row: (
            str(row.get("split_method")),
            str(row.get("dataset")),
            float(row.get("learning_rate_head") or 0.0),
        )
    )
    return rows


def best_summary_rows(results: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for best_id, best in results.get("best_results", {}).items():
        rows.append(
            {
                "best_result_id": best_id,
                "dataset": best.get("dataset"),
                "task_type": best.get("task_type"),
                "split_method": best.get("split_method"),
                "similarity_threshold": best.get("similarity_threshold"),
                "model": MODEL_NAME,
                "best_learning_rate_head": best.get("best_learning_rate_head"),
                "validation_summary_json": compact_json(
                    best.get("validation_summary", {})
                ),
                "test_summary_json": compact_json(best.get("test_summary", {})),
                "test_runs_json": compact_json(best.get("test_runs", [])),
                "head_parameters_json": compact_json(best.get("head_parameters", {})),
                "training_parameters_json": compact_json(
                    best.get("training_parameters", {})
                ),
                "esm_parameters_json": compact_json(best.get("esm_parameters", {})),
                "best_metrics_path": best.get("best_metrics_path"),
                "status": best.get("status"),
                "updated_at": best.get("updated_at"),
            }
        )
    rows.sort(
        key=lambda row: (
            str(row.get("split_method")),
            str(row.get("dataset")),
        )
    )
    return rows


def persist_results(
    results: Dict[str, Any],
    output_json: Path,
    search_csv: Path,
    lr_summary_csv: Path,
    best_summary_csv: Path,
) -> None:
    results["updated_at"] = utc_now()
    atomic_json_dump(results, output_json)
    atomic_dataframe_to_csv(pd.DataFrame(search_record_rows(results)), search_csv)
    atomic_dataframe_to_csv(
        pd.DataFrame(learning_rate_summary_rows(results)), lr_summary_csv
    )
    atomic_dataframe_to_csv(pd.DataFrame(best_summary_rows(results)), best_summary_csv)


# =============================================================================
# Reproducibility and GPU helpers
# =============================================================================


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def capture_rng_state(train_generator: torch.Generator) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "train_generator": train_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(
    state: Mapping[str, Any],
    train_generator: torch.Generator,
) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if "train_generator" in state:
        train_generator.set_state(state["train_generator"])
    if torch.cuda.is_available() and "torch_cuda_all" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def resolve_cuda_device(device_text: str) -> torch.device:
    if not device_text.lower().startswith("cuda"):
        raise ValueError("--device must be a CUDA device such as cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available to PyTorch. This script does not fall back to CPU."
        )
    device = torch.device(device_text)
    try:
        torch.cuda.get_device_properties(device)
    except Exception as exc:
        raise RuntimeError(f"Cannot access CUDA device {device_text}: {exc}") from exc
    torch.cuda.set_device(device)
    return device


def gpu_name(device: torch.device) -> str:
    return torch.cuda.get_device_name(device)


def cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: tensor.detach().cpu().clone()
        for key, tensor in model.state_dict().items()
    }


def release_memory(*objects: Any) -> None:
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Dataset loading, task detection, splitting, and frozen ESM extraction
# =============================================================================


def collect_csv_files(
    data_dir: Path,
    datasets: Optional[Sequence[str]],
) -> List[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    if datasets:
        paths = [data_dir / f"{name}.csv" for name in datasets]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Requested datasets not found: {missing}")
        return paths
    paths = sorted(data_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV datasets found in: {data_dir}")
    return paths


def determine_task_type(dataset_name: str) -> str:
    return "regression" if "reg" in dataset_name.lower() else "classification"


def validate_dataframe(
    dataframe: pd.DataFrame,
    dataset_name: str,
    task_type: str,
) -> pd.DataFrame:
    required = {"peps", "label"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise KeyError(
            f"Dataset {dataset_name} is missing columns: {sorted(missing)}"
        )
    data = dataframe.copy().reset_index(drop=True)
    if data.empty:
        raise ValueError(f"Dataset {dataset_name} is empty")
    if data["peps"].isna().any():
        raise ValueError(f"Dataset {dataset_name} contains missing sequences")
    labels = pd.to_numeric(data["label"], errors="raise").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(labels)):
        raise ValueError(f"Dataset {dataset_name} contains non-finite labels")
    if task_type == "classification":
        unique = set(float(value) for value in np.unique(labels))
        if not unique.issubset({0.0, 1.0}):
            raise ValueError(
                f"Classification labels must be 0/1 or 0.0/1.0; got {sorted(unique)}"
            )
    data["label"] = labels.astype(np.float32)
    data["peps"] = data["peps"].astype(str)
    data["_sample_id"] = np.arange(len(data), dtype=np.int64)
    return data


def load_dataset_with_project_loader(
    data_dir: Path,
    dataset_name: str,
) -> pd.DataFrame:
    loader = PepDataset(data_dir=str(data_dir))
    return loader.load_dataset(dataset_name)


def split_dataset_for_seed(
    data: pd.DataFrame,
    task_type: str,
    split_method: str,
    test_size: float,
    val_size: float,
    seed: int,
    similarity_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if split_method == "similarity":
        train_data, test_data, val_data = split_dataset_by_similarity(
            data,
            test_size=test_size,
            val_size=val_size,
            random_state=seed,
            similarity_threshold=similarity_threshold,
        )
    elif split_method == "random":
        train_data, test_data, val_data = split_dataset(
            data,
            test_size=test_size,
            val_size=val_size,
            random_state=seed,
            stratify=(task_type == "classification"),
        )
    else:
        raise ValueError(f"Unsupported split method: {split_method}")

    for split_name, frame in (
        ("train", train_data),
        ("validation", val_data),
        ("test", test_data),
    ):
        if frame is None or len(frame) == 0:
            raise ValueError(f"{split_name} split is empty")
        if "_sample_id" not in frame.columns:
            raise RuntimeError(
                "The splitter did not preserve the '_sample_id' column. "
                "The learning-rate search requires row IDs so one shared in-memory "
                "ESM feature matrix can be indexed safely."
            )

    return (
        train_data.reset_index(drop=True).copy(),
        test_data.reset_index(drop=True).copy(),
        val_data.reset_index(drop=True).copy(),
    )


def configure_frozen_featurizer(featurizer: Any) -> None:
    """Best-effort enforcement that any exposed ESM module is frozen and in eval mode."""
    for attribute_name in ("esm_model", "model"):
        module = getattr(featurizer, attribute_name, None)
        if isinstance(module, nn.Module):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False


def normalize_feature_batch(raw_features: Any, expected_size: int) -> np.ndarray:
    if isinstance(raw_features, torch.Tensor):
        array = raw_features.detach().float().cpu().numpy()
    elif isinstance(raw_features, list):
        if len(raw_features) != expected_size:
            raise ValueError(
                f"Feature list length {len(raw_features)} does not match batch size "
                f"{expected_size}"
            )
        array = np.stack(
            [np.asarray(item, dtype=np.float32).reshape(-1) for item in raw_features],
            axis=0,
        )
    else:
        array = np.asarray(raw_features, dtype=np.float32)

    if array.ndim == 1:
        if expected_size != 1:
            raise ValueError(
                f"Received one feature vector for a batch of {expected_size} sequences"
            )
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {array.shape}")
    if array.shape[0] != expected_size:
        raise ValueError(
            f"Feature rows {array.shape[0]} do not match batch size {expected_size}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("ESM features contain NaN or infinite values")
    return np.ascontiguousarray(array, dtype=np.float32)


def extract_all_features(
    sequences: Sequence[str],
    featurizer: Any,
    feature_batch_size: int,
) -> torch.Tensor:
    """Extract the frozen ESM representation once and keep it in CPU memory."""
    if feature_batch_size <= 0:
        raise ValueError("feature_batch_size must be positive")
    configure_frozen_featurizer(featurizer)

    all_batches: List[torch.Tensor] = []
    total = len(sequences)
    number_of_batches = (total + feature_batch_size - 1) // feature_batch_size
    print(
        f"Extracting frozen ESM features once: samples={total}, "
        f"feature_batch_size={feature_batch_size}, batches={number_of_batches}"
    )

    for batch_index, start in enumerate(range(0, total, feature_batch_size), start=1):
        end = min(start + feature_batch_size, total)
        batch_sequences = list(sequences[start:end])
        print(
            f"  ESM feature batch {batch_index}/{number_of_batches}: "
            f"samples {start + 1}-{end}/{total}"
        )
        # PeptideFeaturizer handles the frozen ESM forward pass internally.
        raw_features = featurizer.transform(batch_sequences)
        array = normalize_feature_batch(raw_features, len(batch_sequences))
        all_batches.append(torch.from_numpy(array))

    feature_tensor = torch.cat(all_batches, dim=0).contiguous()
    if feature_tensor.shape[0] != total:
        raise RuntimeError("Feature extraction returned the wrong number of samples")
    print(f"Frozen ESM feature extraction complete: shape={tuple(feature_tensor.shape)}")
    return feature_tensor


def prepare_split_contexts(
    data: pd.DataFrame,
    task_type: str,
    split_method: str,
    seeds: Sequence[int],
    test_size: float,
    val_size: float,
    similarity_threshold: float,
) -> Dict[int, Dict[str, Any]]:
    contexts: Dict[int, Dict[str, Any]] = {}
    for seed in seeds:
        train_data, test_data, val_data = split_dataset_for_seed(
            data=data,
            task_type=task_type,
            split_method=split_method,
            test_size=test_size,
            val_size=val_size,
            seed=int(seed),
            similarity_threshold=similarity_threshold,
        )
        contexts[int(seed)] = {
            "train_ids": train_data["_sample_id"].to_numpy(dtype=np.int64),
            "validation_ids": val_data["_sample_id"].to_numpy(dtype=np.int64),
            "test_ids": test_data["_sample_id"].to_numpy(dtype=np.int64),
            "split_sizes": {
                "train": int(len(train_data)),
                "validation": int(len(val_data)),
                "test": int(len(test_data)),
            },
        }
        print(
            f"Prepared split seed={seed}: train={len(train_data)}, "
            f"validation={len(val_data)}, test={len(test_data)}"
        )
    return contexts


# =============================================================================
# Metrics and DataLoader creation
# =============================================================================


def compute_classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    labels_int = np.asarray(labels, dtype=np.float32).reshape(-1).astype(np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    predictions = (probabilities >= 0.5).astype(np.int64)
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(labels_int, predictions)),
        "precision": float(
            precision_score(labels_int, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels_int, predictions, zero_division=0)),
        "f1": float(f1_score(labels_int, predictions, zero_division=0)),
        "roc_auc": None,
        "auprc": None,
    }
    if len(np.unique(labels_int)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels_int, probabilities))
        metrics["auprc"] = float(
            average_precision_score(labels_int, probabilities)
        )
    return metrics


def compute_regression_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    mse = float(mean_squared_error(labels, predictions))
    metrics: Dict[str, float] = {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(labels, predictions)),
    }
    if len(labels) >= 2:
        r2 = float(r2_score(labels, predictions))
        if np.isfinite(r2):
            metrics["r2"] = r2
    return metrics


def summarize_metrics(
    metric_dicts: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, float]]:
    names: set[str] = set()
    for metrics in metric_dicts:
        names.update(metrics.keys())

    summary: Dict[str, Dict[str, float]] = {}
    for name in sorted(names):
        values: List[float] = []
        for metrics in metric_dicts:
            raw_value = metrics.get(name)
            if raw_value is None:
                continue
            try:
                numeric = float(raw_value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                values.append(numeric)
        if values:
            summary[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "n": int(len(values)),
            }
    return summary


def make_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> Tuple[DataLoader, torch.Generator]:
    indices = torch.as_tensor(sample_ids, dtype=torch.long)
    selected_features = features.index_select(0, indices)
    selected_labels = labels.index_select(0, indices)
    dataset = TensorDataset(selected_features, selected_labels)

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        generator=generator if shuffle else None,
    )
    return loader, generator


def create_criterion(
    task_type: str,
    train_labels: torch.Tensor,
    use_class_weight: bool,
    device: torch.device,
) -> nn.Module:
    if task_type == "regression":
        return nn.MSELoss()
    if not use_class_weight:
        return nn.BCEWithLogitsLoss()

    positive_count = int((train_labels == 1.0).sum().item())
    negative_count = int((train_labels == 0.0).sum().item())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Cannot compute binary class weight when one class is absent")
    pos_weight = torch.tensor(
        negative_count / positive_count,
        dtype=torch.float32,
        device=device,
    )
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch_features, batch_labels in loader:
        batch_features = batch_features.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True).view(-1)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch_features).view(-1)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        batch_size = int(batch_labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
    if total_samples == 0:
        raise RuntimeError("Training DataLoader produced no samples")
    return total_loss / total_samples


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    task_type: str,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    outputs_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []

    with torch.no_grad():
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True).view(-1)
            logits_or_predictions = model(batch_features).view(-1)
            loss = criterion(logits_or_predictions, batch_labels)

            batch_size = int(batch_labels.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            labels_all.append(batch_labels.detach().float().cpu().numpy())
            if task_type == "classification":
                probabilities = torch.sigmoid(logits_or_predictions)
                outputs_all.append(probabilities.detach().float().cpu().numpy())
            else:
                outputs_all.append(
                    logits_or_predictions.detach().float().cpu().numpy()
                )

    if total_samples == 0:
        raise RuntimeError("Evaluation DataLoader produced no samples")
    labels = np.concatenate(labels_all)
    outputs = np.concatenate(outputs_all)
    metrics: Dict[str, float] = {"loss": total_loss / total_samples}
    if task_type == "classification":
        metrics.update(compute_classification_metrics(labels, outputs))
    else:
        metrics.update(compute_regression_metrics(labels, outputs))
    return metrics


def selection_metric_name(task_type: str) -> str:
    return "f1" if task_type == "classification" else "rmse"


def selection_direction(task_type: str) -> str:
    return "maximize" if task_type == "classification" else "minimize"


def is_improved(
    task_type: str,
    metrics: Mapping[str, float],
    best_metric: float,
) -> bool:
    current = float(metrics[selection_metric_name(task_type)])
    if task_type == "classification":
        return current > best_metric
    return current < best_metric


# =============================================================================
# Training with validation and final refitting
# =============================================================================


def train_with_validation(
    features: torch.Tensor,
    labels: torch.Tensor,
    context: Mapping[str, Any],
    task_type: str,
    seed: int,
    learning_rate: float,
    input_dim: int,
    hidden_dims: Sequence[int],
    dropout: float,
    activation: str,
    batch_size: int,
    max_epochs: int,
    patience: int,
    weight_decay: float,
    num_workers: int,
    log_interval: int,
    checkpoint_interval: int,
    use_class_weight: bool,
    device: torch.device,
    resume_checkpoint_path: Path,
) -> Tuple[Dict[str, float], int]:
    set_global_seed(seed)
    train_loader, train_generator = make_loader(
        features=features,
        labels=labels,
        sample_ids=np.asarray(context["train_ids"]),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
    )
    val_loader, _ = make_loader(
        features=features,
        labels=labels,
        sample_ids=np.asarray(context["validation_ids"]),
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
    )

    model = build_head(
        task_type=task_type,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
        device=device,
    )
    train_label_tensor = labels.index_select(
        0, torch.as_tensor(context["train_ids"], dtype=torch.long)
    )
    criterion = create_criterion(
        task_type=task_type,
        train_labels=train_label_tensor,
        use_class_weight=use_class_weight,
        device=device,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_metric = -float("inf") if task_type == "classification" else float("inf")
    best_epoch = 0
    best_metrics: Optional[Dict[str, float]] = None
    best_state: Optional[Dict[str, torch.Tensor]] = None
    no_improve_count = 0
    start_epoch = 1

    if resume_checkpoint_path.exists():
        checkpoint = torch_load_compatible(resume_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_metric = float(checkpoint["best_metric"])
        best_epoch = int(checkpoint["best_epoch"])
        best_metrics = checkpoint.get("best_metrics")
        best_state = checkpoint.get("best_model_state_dict")
        no_improve_count = int(checkpoint.get("no_improve_count", 0))
        start_epoch = int(checkpoint["current_epoch"]) + 1
        if checkpoint.get("rng_state"):
            restore_rng_state(checkpoint["rng_state"], train_generator)
        elif checkpoint.get("train_generator_state") is not None:
            # Backward compatibility with an older checkpoint created by this script.
            train_generator.set_state(checkpoint["train_generator_state"])
        print(
            f"Resuming LR={learning_rate:g}, seed={seed} from epoch {start_epoch}"
        )

    for epoch in range(start_epoch, max_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        validation_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            task_type=task_type,
            device=device,
        )
        improved = is_improved(task_type, validation_metrics, best_metric)
        if improved:
            best_metric = float(validation_metrics[selection_metric_name(task_type)])
            best_epoch = epoch
            best_metrics = dict(validation_metrics)
            best_state = cpu_state_dict(model)
            no_improve_count = 0
        else:
            no_improve_count += 1

        if (
            epoch == 1
            or epoch % log_interval == 0
            or no_improve_count >= patience
            or epoch == max_epochs
        ):
            if task_type == "classification":
                print(
                    f"LR={learning_rate:g} | seed={seed} | epoch={epoch}/{max_epochs} "
                    f"| train_loss={train_loss:.6f} | val_f1={validation_metrics['f1']:.6f} "
                    f"| val_loss={validation_metrics['loss']:.6f}"
                )
            else:
                print(
                    f"LR={learning_rate:g} | seed={seed} | epoch={epoch}/{max_epochs} "
                    f"| train_loss={train_loss:.6f} | val_rmse={validation_metrics['rmse']:.6f} "
                    f"| val_loss={validation_metrics['loss']:.6f}"
                )

        should_save = (
            improved
            or epoch % checkpoint_interval == 0
            or no_improve_count >= patience
            or epoch == max_epochs
        )
        if should_save:
            atomic_torch_save(
                {
                    "mode": "learning_rate_search",
                    "learning_rate_head": float(learning_rate),
                    "seed": int(seed),
                    "task_type": task_type,
                    "current_epoch": int(epoch),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": float(best_metric),
                    "best_epoch": int(best_epoch),
                    "best_metrics": best_metrics,
                    "best_model_state_dict": best_state,
                    "no_improve_count": int(no_improve_count),
                    "rng_state": capture_rng_state(train_generator),
                    "saved_at": utc_now(),
                },
                resume_checkpoint_path,
            )

        if no_improve_count >= patience:
            print(
                f"Early stopping: LR={learning_rate:g}, seed={seed}, "
                f"best_epoch={best_epoch}"
            )
            break

    if best_metrics is None or best_state is None or best_epoch <= 0:
        raise RuntimeError("No valid validation epoch was produced")

    resume_checkpoint_path.unlink(missing_ok=True)
    release_memory(model, optimizer, criterion, train_loader, val_loader)
    return best_metrics, best_epoch


def train_fixed_epochs_and_test(
    features: torch.Tensor,
    labels: torch.Tensor,
    context: Mapping[str, Any],
    task_type: str,
    seed: int,
    learning_rate: float,
    final_epochs: int,
    input_dim: int,
    hidden_dims: Sequence[int],
    dropout: float,
    activation: str,
    batch_size: int,
    weight_decay: float,
    num_workers: int,
    log_interval: int,
    checkpoint_interval: int,
    use_class_weight: bool,
    device: torch.device,
    resume_checkpoint_path: Path,
) -> Tuple[nn.Module, Dict[str, float]]:
    if final_epochs <= 0:
        raise ValueError(f"final_epochs must be positive; got {final_epochs}")
    set_global_seed(seed)

    train_val_ids = np.concatenate(
        [
            np.asarray(context["train_ids"], dtype=np.int64),
            np.asarray(context["validation_ids"], dtype=np.int64),
        ]
    )
    train_loader, train_generator = make_loader(
        features=features,
        labels=labels,
        sample_ids=train_val_ids,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
    )
    test_loader, _ = make_loader(
        features=features,
        labels=labels,
        sample_ids=np.asarray(context["test_ids"], dtype=np.int64),
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
    )

    model = build_head(
        task_type=task_type,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
        device=device,
    )
    train_labels = labels.index_select(
        0, torch.as_tensor(train_val_ids, dtype=torch.long)
    )
    criterion = create_criterion(
        task_type=task_type,
        train_labels=train_labels,
        use_class_weight=use_class_weight,
        device=device,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    start_epoch = 1
    if resume_checkpoint_path.exists():
        checkpoint = torch_load_compatible(resume_checkpoint_path, map_location=device)
        checkpoint_target = int(checkpoint.get("target_epochs", final_epochs))
        if checkpoint_target != final_epochs:
            raise RuntimeError(
                f"Stale final-refit checkpoint targets {checkpoint_target} epochs, "
                f"but current target is {final_epochs}: {resume_checkpoint_path}"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["current_epoch"]) + 1
        if checkpoint.get("rng_state"):
            restore_rng_state(checkpoint["rng_state"], train_generator)
        elif checkpoint.get("train_generator_state") is not None:
            # Backward compatibility with an older checkpoint created by this script.
            train_generator.set_state(checkpoint["train_generator_state"])
        print(
            f"Resuming final refit seed={seed} from epoch "
            f"{start_epoch}/{final_epochs}"
        )

    for epoch in range(start_epoch, final_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        if epoch == 1 or epoch % log_interval == 0 or epoch == final_epochs:
            print(
                f"Final refit | LR={learning_rate:g} | seed={seed} | "
                f"epoch={epoch}/{final_epochs} | train_loss={train_loss:.6f}"
            )
        if epoch % checkpoint_interval == 0 or epoch == final_epochs:
            atomic_torch_save(
                {
                    "mode": "final_refit",
                    "learning_rate_head": float(learning_rate),
                    "seed": int(seed),
                    "task_type": task_type,
                    "current_epoch": int(epoch),
                    "target_epochs": int(final_epochs),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "rng_state": capture_rng_state(train_generator),
                    "saved_at": utc_now(),
                },
                resume_checkpoint_path,
            )

    test_metrics = evaluate_model(
        model=model,
        loader=test_loader,
        criterion=criterion,
        task_type=task_type,
        device=device,
    )
    resume_checkpoint_path.unlink(missing_ok=True)
    del optimizer, criterion, train_loader, test_loader
    gc.collect()
    return model, test_metrics


# =============================================================================
# Experiment IDs and selection
# =============================================================================


def common_signature(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "esm_model_name": args.esm_model_name,
        "esm_pooling": args.esm_pooling,
        "hidden_dims": list(args.hidden_dims),
        "dropout": float(args.dropout),
        "activation": args.activation,
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "use_class_weight": bool(args.use_class_weight),
    }


def make_search_record_id(
    args: argparse.Namespace,
    dataset_name: str,
    task_type: str,
    learning_rate: float,
    seed: int,
) -> str:
    return stable_id(
        "search",
        {
            "split_method": args.split_method,
            "similarity_threshold": args.similarity_threshold
            if args.split_method == "similarity"
            else None,
            "dataset": dataset_name,
            "task_type": task_type,
            "learning_rate_head": float(learning_rate),
            "seed": int(seed),
            **common_signature(args),
        },
    )


def make_lr_summary_id(
    args: argparse.Namespace,
    dataset_name: str,
    task_type: str,
    learning_rate: float,
) -> str:
    return stable_id(
        "lr",
        {
            "split_method": args.split_method,
            "similarity_threshold": args.similarity_threshold
            if args.split_method == "similarity"
            else None,
            "dataset": dataset_name,
            "task_type": task_type,
            "learning_rate_head": float(learning_rate),
            **common_signature(args),
        },
    )


def make_best_result_id(
    args: argparse.Namespace,
    dataset_name: str,
    task_type: str,
) -> str:
    return stable_id(
        "best",
        {
            "split_method": args.split_method,
            "similarity_threshold": args.similarity_threshold
            if args.split_method == "similarity"
            else None,
            "dataset": dataset_name,
            "task_type": task_type,
            "learning_rate_grid": [float(value) for value in args.learning_rates],
            **common_signature(args),
        },
    )


def select_best_learning_rate(
    task_type: str,
    summaries: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    complete = [summary for summary in summaries if summary.get("status") == "complete"]
    if not complete:
        raise RuntimeError("No learning-rate candidate completed every requested seed")
    metric_name = selection_metric_name(task_type)

    def key(summary: Mapping[str, Any]) -> Tuple[float, float, float]:
        metric_summary = summary["validation_summary"][metric_name]
        mean = float(metric_summary["mean"])
        std = float(metric_summary["std"])
        learning_rate = float(summary["learning_rate_head"])
        if task_type == "classification":
            return (-mean, std, learning_rate)
        return (mean, std, learning_rate)

    return sorted(complete, key=key)[0]


# =============================================================================
# Search and final-refit orchestration
# =============================================================================


def train_dataset(
    args: argparse.Namespace,
    results: Dict[str, Any],
    output_json: Path,
    search_csv: Path,
    lr_summary_csv: Path,
    best_summary_csv: Path,
    best_model_dir: Path,
    best_metrics_dir: Path,
    resume_dir: Path,
    dataset_name: str,
    task_type: str,
    features: torch.Tensor,
    labels: torch.Tensor,
    contexts: Mapping[int, Mapping[str, Any]],
    device: torch.device,
) -> None:
    input_dim = int(features.shape[1])
    lr_summaries: List[Dict[str, Any]] = []
    head_parameters = {
        "input_dim": input_dim,
        "hidden_dims": list(args.hidden_dims),
        "dropout": float(args.dropout),
        "activation": args.activation,
    }
    training_parameters = {
        "optimizer": "Adam",
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "num_workers": int(args.num_workers),
        "use_class_weight": bool(args.use_class_weight),
        "classification_loss": "BCEWithLogitsLoss",
        "regression_loss": "MSELoss",
    }
    esm_parameters = {
        "frozen": True,
        "feature_type": FEATURE_TYPE,
        "esm_model_name": args.esm_model_name,
        "esm_pooling": args.esm_pooling,
        "feature_batch_size": int(args.feature_batch_size),
        "feature_dimension": input_dim,
        "feature_storage": "in_memory_once_per_dataset",
    }

    for lr_index, learning_rate in enumerate(args.learning_rates, start=1):
        learning_rate = float(learning_rate)
        print(
            "\n" + "=" * 80 +
            f"\nDataset={dataset_name} | LR candidate {lr_index}/{len(args.learning_rates)} "
            f"| learning_rate_head={learning_rate:g}\n" + "=" * 80
        )
        validation_metrics_by_seed: List[Dict[str, float]] = []
        successful_seed_count = 0

        for seed in args.seeds:
            seed = int(seed)
            record_id = make_search_record_id(
                args=args,
                dataset_name=dataset_name,
                task_type=task_type,
                learning_rate=learning_rate,
                seed=seed,
            )
            existing = results["search_records"].get(record_id)
            if existing and existing.get("status") == "success":
                print(f"Seed {seed}: successful record exists; skipping")
                validation_metrics_by_seed.append(existing["validation_metrics"])
                successful_seed_count += 1
                continue

            resume_path = (
                resume_dir
                / sanitize_filename(args.split_method)
                / sanitize_filename(dataset_name)
                / f"search_lr_{learning_rate:.0e}_seed{seed}_{record_id}.pt"
            )
            started_at = utc_now()
            try:
                validation_metrics, best_epoch = train_with_validation(
                    features=features,
                    labels=labels,
                    context=contexts[seed],
                    task_type=task_type,
                    seed=seed,
                    learning_rate=learning_rate,
                    input_dim=input_dim,
                    hidden_dims=args.hidden_dims,
                    dropout=args.dropout,
                    activation=args.activation,
                    batch_size=args.batch_size,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    weight_decay=args.weight_decay,
                    num_workers=args.num_workers,
                    log_interval=args.log_interval,
                    checkpoint_interval=args.checkpoint_interval,
                    use_class_weight=args.use_class_weight,
                    device=device,
                    resume_checkpoint_path=resume_path,
                )
                record = {
                    "record_id": record_id,
                    "status": "success",
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "dataset": dataset_name,
                    "task_type": task_type,
                    "split_method": args.split_method,
                    "similarity_threshold": args.similarity_threshold
                    if args.split_method == "similarity"
                    else None,
                    "random_seed": seed,
                    "learning_rate_head": learning_rate,
                    "best_epoch": int(best_epoch),
                    "validation_metrics": validation_metrics,
                    "selection_metric": selection_metric_name(task_type),
                    "selection_direction": selection_direction(task_type),
                    "head_parameters": head_parameters,
                    "training_parameters": training_parameters,
                    "esm_parameters": esm_parameters,
                    "split_sizes": contexts[seed]["split_sizes"],
                    "device": str(device),
                    "gpu_name": gpu_name(device),
                    "error": None,
                }
                results["search_records"][record_id] = record
                validation_metrics_by_seed.append(validation_metrics)
                successful_seed_count += 1
                print(
                    f"Seed {seed}: best_epoch={best_epoch}, "
                    f"validation_metrics={validation_metrics}"
                )
            except Exception as exc:
                results["search_records"][record_id] = {
                    "record_id": record_id,
                    "status": "failed",
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "dataset": dataset_name,
                    "task_type": task_type,
                    "split_method": args.split_method,
                    "similarity_threshold": args.similarity_threshold
                    if args.split_method == "similarity"
                    else None,
                    "random_seed": seed,
                    "learning_rate_head": learning_rate,
                    "best_epoch": None,
                    "validation_metrics": None,
                    "selection_metric": selection_metric_name(task_type),
                    "selection_direction": selection_direction(task_type),
                    "head_parameters": head_parameters,
                    "training_parameters": training_parameters,
                    "esm_parameters": esm_parameters,
                    "split_sizes": contexts[seed]["split_sizes"],
                    "device": str(device),
                    "gpu_name": gpu_name(device),
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                }
                print(f"Seed {seed}: FAILED - {exc}")

            persist_results(
                results,
                output_json,
                search_csv,
                lr_summary_csv,
                best_summary_csv,
            )
            release_memory()

        summary_id = make_lr_summary_id(
            args=args,
            dataset_name=dataset_name,
            task_type=task_type,
            learning_rate=learning_rate,
        )
        status = (
            "complete"
            if successful_seed_count == len(args.seeds)
            else "incomplete"
        )
        summary = {
            "summary_id": summary_id,
            "status": status,
            "updated_at": utc_now(),
            "dataset": dataset_name,
            "task_type": task_type,
            "split_method": args.split_method,
            "similarity_threshold": args.similarity_threshold
            if args.split_method == "similarity"
            else None,
            "learning_rate_head": learning_rate,
            "expected_seeds": list(args.seeds),
            "successful_seed_count": successful_seed_count,
            "validation_summary": summarize_metrics(validation_metrics_by_seed),
            "selection_metric": selection_metric_name(task_type),
            "selection_direction": selection_direction(task_type),
            "head_parameters": head_parameters,
            "training_parameters": training_parameters,
            "esm_parameters": esm_parameters,
        }
        results["learning_rate_summaries"][summary_id] = summary
        lr_summaries.append(summary)
        persist_results(
            results,
            output_json,
            search_csv,
            lr_summary_csv,
            best_summary_csv,
        )

    best_summary = select_best_learning_rate(task_type, lr_summaries)
    best_learning_rate = float(best_summary["learning_rate_head"])
    print(
        f"\nSelected best learning rate for {dataset_name}: "
        f"{best_learning_rate:g} based on mean validation "
        f"{selection_metric_name(task_type)}"
    )

    best_id = make_best_result_id(args, dataset_name, task_type)
    existing_best = results["best_results"].get(best_id)
    if existing_best and existing_best.get("status") == "success":
        existing_seeds = {
            int(run["random_seed"])
            for run in existing_best.get("test_runs", [])
            if run.get("status") == "success"
        }
        model_files_exist = all(
            run.get("model_path") and Path(run["model_path"]).exists()
            for run in existing_best.get("test_runs", [])
            if run.get("status") == "success"
        )
        if (
            existing_seeds == set(int(seed) for seed in args.seeds)
            and model_files_exist
        ):
            print(f"Final best-result record already complete for {dataset_name}; skipping")
            return

    test_runs: List[Dict[str, Any]] = []
    for seed in args.seeds:
        seed = int(seed)
        selected_record_id = make_search_record_id(
            args=args,
            dataset_name=dataset_name,
            task_type=task_type,
            learning_rate=best_learning_rate,
            seed=seed,
        )
        selected_record = results["search_records"].get(selected_record_id)
        if not selected_record or selected_record.get("status") != "success":
            raise RuntimeError(
                f"Missing successful search record for best LR, seed={seed}"
            )
        final_epochs = int(selected_record["best_epoch"])
        final_resume_path = (
            resume_dir
            / sanitize_filename(args.split_method)
            / sanitize_filename(dataset_name)
            / f"final_seed{seed}_{best_id}.pt"
        )
        try:
            model, test_metrics = train_fixed_epochs_and_test(
                features=features,
                labels=labels,
                context=contexts[seed],
                task_type=task_type,
                seed=seed,
                learning_rate=best_learning_rate,
                final_epochs=final_epochs,
                input_dim=input_dim,
                hidden_dims=args.hidden_dims,
                dropout=args.dropout,
                activation=args.activation,
                batch_size=args.batch_size,
                weight_decay=args.weight_decay,
                num_workers=args.num_workers,
                log_interval=args.log_interval,
                checkpoint_interval=args.checkpoint_interval,
                use_class_weight=args.use_class_weight,
                device=device,
                resume_checkpoint_path=final_resume_path,
            )
            model_path = (
                best_model_dir
                / sanitize_filename(args.split_method)
                / sanitize_filename(dataset_name)
                / f"esm_mlp__seed{seed}.pt"
            )
            atomic_torch_save(
                {
                    "model_name": MODEL_NAME,
                    "freeze_esm": True,
                    "head_state_dict": cpu_state_dict(model),
                    "task_type": task_type,
                    "input_dim": input_dim,
                    "hidden_dims": list(args.hidden_dims),
                    "dropout": float(args.dropout),
                    "activation": args.activation,
                    "best_learning_rate_head": best_learning_rate,
                    "weight_decay": float(args.weight_decay),
                    "use_class_weight": bool(args.use_class_weight),
                    "esm_model_name": args.esm_model_name,
                    "esm_pooling": args.esm_pooling,
                    "random_seed": seed,
                    "split_method": args.split_method,
                    "similarity_threshold": args.similarity_threshold
                    if args.split_method == "similarity"
                    else None,
                    "dataset": dataset_name,
                    "final_epochs": final_epochs,
                    "test_metrics": test_metrics,
                    "saved_at": utc_now(),
                },
                model_path,
            )
            test_runs.append(
                {
                    "status": "success",
                    "random_seed": seed,
                    "final_epochs": final_epochs,
                    "model_path": str(model_path),
                    "test_metrics": test_metrics,
                }
            )
            print(
                f"Final seed {seed}: test_metrics={test_metrics}, "
                f"model={model_path}"
            )
            release_memory(model)
        except Exception as exc:
            test_runs.append(
                {
                    "status": "failed",
                    "random_seed": seed,
                    "final_epochs": final_epochs,
                    "model_path": None,
                    "test_metrics": None,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                }
            )
            print(f"Final refit seed {seed}: FAILED - {exc}")

    successful_test_metrics = [
        run["test_metrics"]
        for run in test_runs
        if run.get("status") == "success" and run.get("test_metrics")
    ]
    final_status = (
        "success"
        if len(successful_test_metrics) == len(args.seeds)
        else "incomplete"
    )
    best_metrics_path = (
        best_metrics_dir
        / sanitize_filename(args.split_method)
        / sanitize_filename(dataset_name)
        / "esm_mlp.json"
    )
    best_result = {
        "best_result_id": best_id,
        "status": final_status,
        "updated_at": utc_now(),
        "dataset": dataset_name,
        "task_type": task_type,
        "split_method": args.split_method,
        "similarity_threshold": args.similarity_threshold
        if args.split_method == "similarity"
        else None,
        "model": MODEL_NAME,
        "best_learning_rate_head": best_learning_rate,
        "selection_metric": selection_metric_name(task_type),
        "selection_direction": selection_direction(task_type),
        "validation_summary": best_summary["validation_summary"],
        "test_summary": summarize_metrics(successful_test_metrics),
        "test_runs": test_runs,
        "head_parameters": head_parameters,
        "training_parameters": training_parameters,
        "esm_parameters": esm_parameters,
        "best_metrics_path": str(best_metrics_path),
    }
    results["best_results"][best_id] = best_result
    write_metrics_report(best_metrics_path, best_result)
    persist_results(
        results,
        output_json,
        search_csv,
        lr_summary_csv,
        best_summary_csv,
    )


# =============================================================================
# Main
# =============================================================================


def resolve_script_relative_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path.resolve()


def resolve_esm_model_name(value: str) -> str:
    candidate = Path(value).expanduser()
    script_relative = Path(__file__).resolve().parent / candidate
    if candidate.is_absolute():
        return str(candidate.resolve())
    if value.startswith(".") or script_relative.exists():
        return str(script_relative.resolve())
    # Otherwise preserve values such as facebook/esm2_t12_35M_UR50D.
    return value


def run_experiments(args: argparse.Namespace) -> None:
    device = resolve_cuda_device(args.device)
    torch.set_float32_matmul_precision("high")

    data_dir = resolve_script_relative_path(args.data_dir)
    output_json = resolve_script_relative_path(args.output_json)
    search_csv = resolve_script_relative_path(args.search_csv)
    lr_summary_csv = resolve_script_relative_path(args.lr_summary_csv)
    best_summary_csv = resolve_script_relative_path(args.best_summary_csv)
    best_model_dir = resolve_script_relative_path(args.best_model_dir)
    best_metrics_dir = resolve_script_relative_path(args.best_metrics_dir)
    resume_dir = resolve_script_relative_path(args.resume_dir)
    args.esm_model_name = resolve_esm_model_name(args.esm_model_name)

    results = load_or_initialize_results(output_json)
    run_config_id = stable_id(
        "run",
        {
            "split_method": args.split_method,
            "similarity_threshold": args.similarity_threshold
            if args.split_method == "similarity"
            else None,
            "seeds": list(args.seeds),
            "learning_rates": list(args.learning_rates),
            **common_signature(args),
        },
    )
    results["run_configs"][run_config_id] = {
        "run_config_id": run_config_id,
        "started_at": utc_now(),
        "split_method": args.split_method,
        "similarity_threshold": args.similarity_threshold
        if args.split_method == "similarity"
        else None,
        "data_dir": str(data_dir),
        "datasets": list(args.datasets) if args.datasets else None,
        "seeds": list(args.seeds),
        "learning_rates": list(args.learning_rates),
        "test_size": float(args.test_size),
        "val_size": float(args.val_size),
        "device": str(device),
        "gpu_name": gpu_name(device),
        **common_signature(args),
    }
    persist_results(
        results,
        output_json,
        search_csv,
        lr_summary_csv,
        best_summary_csv,
    )

    print("=" * 80)
    print("Frozen ESM + MLP Fixed-Parameter Training")
    print(f"Device: {device} ({gpu_name(device)})")
    print(f"Data directory: {data_dir}")
    print(f"ESM model: {args.esm_model_name}")
    print(f"ESM pooling: {args.esm_pooling}")
    print(f"Split method: {args.split_method}")
    print(f"Seeds: {list(args.seeds)}")
    print(f"Learning rate: {args.learning_rates[0]:g}")
    print(f"Hidden dimensions: {list(args.hidden_dims)}")
    print("ESM features are extracted once per dataset and reused in memory.")
    print("=" * 80)

    csv_files = collect_csv_files(data_dir, args.datasets)
    print(f"Found {len(csv_files)} datasets: {[path.stem for path in csv_files]}")

    # One frozen ESM featurizer instance is reused across datasets in this invocation.
    featurizer = PeptideFeaturizer(
        feature_type="esm",
        esm_model_name=args.esm_model_name,
        esm_pooling=args.esm_pooling,
        device=str(device),
    )
    configure_frozen_featurizer(featurizer)

    for dataset_index, csv_file in enumerate(csv_files, start=1):
        dataset_name = csv_file.stem
        task_type = determine_task_type(dataset_name)
        print("\n" + "#" * 80)
        print(
            f"Dataset [{dataset_index}/{len(csv_files)}]: {dataset_name} "
            f"(task={task_type})"
        )
        print("#" * 80)
        try:
            raw_data = load_dataset_with_project_loader(data_dir, dataset_name)
            data = validate_dataframe(raw_data, dataset_name, task_type)
            label_values = data["label"].to_numpy(dtype=np.float32)
            if task_type == "regression":
                label_values = np.log1p(label_values)
            labels = torch.as_tensor(label_values, dtype=torch.float32)

            # Important: one unified ESM extraction before any LR candidate training.
            features = extract_all_features(
                sequences=data["peps"].tolist(),
                featurizer=featurizer,
                feature_batch_size=args.feature_batch_size,
            )
            contexts = prepare_split_contexts(
                data=data,
                task_type=task_type,
                split_method=args.split_method,
                seeds=args.seeds,
                test_size=args.test_size,
                val_size=args.val_size,
                similarity_threshold=args.similarity_threshold,
            )
            train_dataset(
                args=args,
                results=results,
                output_json=output_json,
                search_csv=search_csv,
                lr_summary_csv=lr_summary_csv,
                best_summary_csv=best_summary_csv,
                best_model_dir=best_model_dir,
                best_metrics_dir=best_metrics_dir,
                resume_dir=resume_dir,
                dataset_name=dataset_name,
                task_type=task_type,
                features=features,
                labels=labels,
                contexts=contexts,
                device=device,
            )
            del features, labels, contexts, data, raw_data
            release_memory()
        except Exception as exc:
            failure = {
                "dataset": dataset_name,
                "task_type": task_type,
                "split_method": args.split_method,
                "failed_at": utc_now(),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
            results["dataset_failures"].append(failure)
            print(f"Dataset {dataset_name}: FAILED - {exc}")
            persist_results(
                results,
                output_json,
                search_csv,
                lr_summary_csv,
                best_summary_csv,
            )
            release_memory()

    results["run_configs"][run_config_id]["finished_at"] = utc_now()
    persist_results(
        results,
        output_json,
        search_csv,
        lr_summary_csv,
        best_summary_csv,
    )
    print("\nAll requested ESM training runs finished.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a frozen ESM encoder with a fixed MLP configuration"
    )
    parser.add_argument(
        "--split_method",
        required=True,
        choices=("random", "similarity"),
    )
    parser.add_argument("--similarity_threshold", type=float, default=0.8)
    parser.add_argument("--data_dir", type=str, default="pephub/raw_data")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)

    parser.add_argument(
        "--esm_model_name",
        type=str,
        default="pretrained/esm2_t12_35M_UR50D",
    )
    parser.add_argument("--esm_pooling", type=str, default="mean")
    parser.add_argument("--feature_batch_size", type=int, default=128)

    parser.add_argument(
        "--hidden_dims",
        nargs="+",
        type=int,
        default=[256, 128],
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--activation",
        type=str,
        choices=("relu", "gelu", "tanh"),
        default="relu",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--checkpoint_interval", type=int, default=1)
    parser.add_argument(
        "--use_class_weight",
        action="store_true",
        help="Use positive-class weighting for binary classification. Default: off.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument(
        "--output_json",
        type=str,
        default="outputs/esm/run_state.json",
    )
    parser.add_argument(
        "--search_csv",
        type=str,
        default="outputs/esm/training_runs.csv",
    )
    parser.add_argument(
        "--lr_summary_csv",
        type=str,
        default="outputs/esm/parameter_summary.csv",
    )
    parser.add_argument(
        "--best_summary_csv",
        type=str,
        default="outputs/esm/summary.csv",
    )
    parser.add_argument("--best_model_dir", type=str, default="outputs/models/esm")
    parser.add_argument("--best_metrics_dir", type=str, default="outputs/metrics/esm")
    parser.add_argument(
        "--resume_dir",
        type=str,
        default="outputs/checkpoints/esm",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not (0.0 < args.test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1")
    if not (0.0 < args.val_size < 1.0):
        raise ValueError("val_size must be between 0 and 1")
    if args.test_size + args.val_size >= 1.0:
        raise ValueError("test_size + val_size must be less than 1")
    if not args.seeds:
        raise ValueError("At least one seed is required")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not args.hidden_dims or any(value <= 0 for value in args.hidden_dims):
        raise ValueError("All hidden dimensions must be positive")
    for name in (
        "feature_batch_size",
        "batch_size",
        "max_epochs",
        "patience",
        "log_interval",
        "checkpoint_interval",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if args.weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")
    if not (0.0 <= args.dropout < 1.0):
        raise ValueError("dropout must be in [0, 1)")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    args.learning_rates = [args.learning_rate]
    run_experiments(args)


if __name__ == "__main__":
    main()
