#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train and evaluate peptide LSTM and Transformer models.

The script uses fixed architectures and a consistent training, validation,
refit, and test procedure for both neural model families.

Experiment workflow:

- scan every CSV dataset in the raw-data directory;
- infer classification/regression from the dataset filename;
- run either random splitting or similarity splitting per invocation;
- reuse the same split for each model under one dataset/seed;
- classification: maximize mean validation F1 across seeds;
- regression: minimize mean validation RMSE across seeds;
- merge train + validation and refit the selected configuration;
- evaluate the final refitted model on the untouched test set;
- save one global JSON, CSV tables, best models, and best metrics;
- resume successful combinations and interrupted epoch-level training.

Fixed configurations
--------------------
The LSTM uses embedding size 50, hidden size 256, and two layers. The
Transformer uses model size 64, four heads, two layers, and an FFN ratio of 4.

The existing model implementations output sigmoid probabilities for binary
classification, so this script uses BCELoss for classification and MSELoss for
regression.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
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
from torch.utils.data import DataLoader, Dataset

from pephub.results import write_metrics_report

from pephub.splitter import split_dataset, split_dataset_by_similarity

# =============================================================================
# Self-contained integer encoder and deep-learning models
# =============================================================================

class IntegerEncoder:
    """Encode peptide sequences as integer token IDs.

    Token 0 is reserved for padding. The 20 standard amino acids use IDs 1-20.
    Unknown residues are mapped to padding ID 0, matching the original project
    implementation.
    """

    def __init__(self, max_len: int = 50):
        if int(max_len) <= 0:
            raise ValueError("max_len must be a positive integer.")
        self.max_len = int(max_len)
        self.aa_to_int = {
            "A": 1, "C": 2, "D": 3, "E": 4, "F": 5,
            "G": 6, "H": 7, "I": 8, "K": 9, "L": 10,
            "M": 11, "N": 12, "P": 13, "Q": 14, "R": 15,
            "S": 16, "T": 17, "V": 18, "W": 19, "Y": 20,
        }
        self.padding_idx = 0
        self.vocab_size = 21

    def encode(self, sequences: Sequence[str] | str) -> np.ndarray:
        if isinstance(sequences, str):
            sequences = [sequences]

        encoded_list: List[List[int]] = []
        for sequence in sequences:
            sequence = str(sequence).strip().upper()
            encoded = [
                self.aa_to_int.get(amino_acid, self.padding_idx)
                for amino_acid in sequence
            ]
            if len(encoded) >= self.max_len:
                encoded = encoded[: self.max_len]
            else:
                encoded.extend([self.padding_idx] * (self.max_len - len(encoded)))
            encoded_list.append(encoded)

        return np.asarray(encoded_list, dtype=np.int64)


class LSTMModel(nn.Module):
    """Bidirectional LSTM used by the original project."""

    def __init__(
        self,
        embedding_dim: int = 50,
        hidden_dim: int = 256,
        num_layers: int = 2,
        task: str = "classification",
        dropout: float = 0.2,
        device: Optional[torch.device] = None,
        max_len: Optional[int] = None,
    ):
        super().__init__()
        self.task = task
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.max_len = max_len
        self.vocab_size = 21
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout_rate = float(dropout)

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=0,
        )
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.classifier = nn.Linear(self.hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        model_device = next(self.parameters()).device
        if x.device != model_device:
            x = x.to(model_device)
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        pooled = lstm_out.mean(dim=1)
        output = self.classifier(pooled)
        if self.task == "classification":
            output = torch.sigmoid(output)
        return output


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for batch-first Transformer inputs."""

    def __init__(self, d_model: int, dropout: float, max_len: int = 198):
        super().__init__()
        self.dropout = nn.Dropout(p=float(dropout))

        pe = torch.zeros(int(max_len), int(d_model))
        position = torch.arange(0, int(max_len), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, int(d_model), 2, dtype=torch.float32)
            * (-math.log(10000.0) / int(d_model))
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"Input sequence length {x.size(1)} exceeds positional-encoding "
                f"length {self.pe.size(1)}. Increase --max_len."
            )
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Transformer encoder used by the original project."""

    def __init__(
        self,
        task: Optional[str] = None,
        device: Optional[torch.device] = None,
        max_len: int = 20,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        if int(d_model) % int(nhead) != 0:
            raise ValueError("d_model must be divisible by nhead.")

        self.task = task
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.max_len = int(max_len)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout_rate = float(dropout)
        self.vocab_size = 21

        self.embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(
            self.d_model, self.dropout_rate, max_len=self.max_len
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_rate,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )
        self.fc = nn.Linear(self.d_model, 1)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        model_device = next(self.parameters()).device
        if src.device != model_device:
            src = src.to(model_device)
        padding_mask = src.eq(0)
        embedded = self.embedding(src) * math.sqrt(self.d_model)
        embedded = self.pos_encoder(embedded)
        output = self.transformer_encoder(
            embedded, src_key_padding_mask=padding_mask
        )
        output = output[:, 0, :]
        output = self.fc(output)
        if self.task == "classification":
            output = torch.sigmoid(output)
        return output


# =============================================================================
# Search spaces and fixed architecture settings
# =============================================================================

LSTM_GRID: Dict[str, Sequence[Any]] = {
    "embedding_dim": [50],
    "hidden_dim": [256],
    "num_layers": [2],
}

TRANSFORMER_BASE_GRID: Dict[str, Sequence[Any]] = {
    "d_model": [64],
    "nhead": [4],
    "num_layers": [2],
    "ffn_ratio": [4],
}

SUPPORTED_MODELS = ("lstm", "transformer")
INPUT_TYPE = "integer_sequence"
FEATURE_TYPE = "integer"

LSTM_FIXED_PARAMETERS: Dict[str, Any] = {
    "dropout": 0.2,
    "bidirectional": True,
    "pooling": "mean",
    "vocab_size": 21,
}

TRANSFORMER_FIXED_PARAMETERS: Dict[str, Any] = {
    "dropout": 0.1,
    "activation": "relu",
    "batch_first": True,
    "pooling": "first_token",
    "positional_encoding": "sinusoidal",
    "vocab_size": 21,
}


# =============================================================================
# General serialization and persistence helpers
# =============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return cleaned or "unnamed"


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return [to_serializable(item) for item in obj.tolist()]
    if isinstance(obj, Mapping):
        return {str(key): to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_serializable(value) for value in obj]
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def compact_json(value: Any) -> str:
    return json.dumps(
        to_serializable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_id(prefix: str, payload: Mapping[str, Any], length: int = 24) -> str:
    normalized = compact_json(dict(payload))
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def atomic_json_dump(data: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=str(output_path.parent),
        text=True,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
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
    """Support both older and newer torch.load signatures."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def atomic_dataframe_to_csv(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=str(output_path.parent),
        text=True,
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        dataframe.to_csv(temp_path, index=False)
        os.replace(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)


def build_lstm_candidates() -> List[Dict[str, Any]]:
    return [{name: values[0] for name, values in LSTM_GRID.items()}]


def build_transformer_candidates() -> List[Dict[str, Any]]:
    candidate = {
        name: values[0] for name, values in TRANSFORMER_BASE_GRID.items()
    }
    candidate["dim_feedforward"] = int(
        int(candidate["d_model"]) * int(candidate["ffn_ratio"])
    )
    return [candidate]


def get_parameter_candidates(model_name: str) -> List[Dict[str, Any]]:
    if model_name == "lstm":
        return build_lstm_candidates()
    if model_name == "transformer":
        return build_transformer_candidates()
    raise ValueError(f"Unsupported model: {model_name}")


def build_search_space_metadata() -> Dict[str, Any]:
    return {
        "lstm": {
            "grid": LSTM_GRID,
            "fixed_parameters": LSTM_FIXED_PARAMETERS,
            "candidate_count": len(build_lstm_candidates()),
        },
        "transformer": {
            "grid": TRANSFORMER_BASE_GRID,
            "derived_parameter": "dim_feedforward = d_model * ffn_ratio",
            "constraints": [
                "d_model % nhead == 0",
                "d_model / nhead >= 8",
            ],
            "fixed_parameters": TRANSFORMER_FIXED_PARAMETERS,
            "candidate_count": len(build_transformer_candidates()),
        },
        "selection": {
            "classification": {"metric": "f1", "direction": "maximize"},
            "regression": {"metric": "rmse", "direction": "minimize"},
        },
    }


def load_or_initialize_results(output_path: Path) -> Dict[str, Any]:
    if output_path.exists():
        try:
            with output_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Existing result JSON is invalid: {output_path}") from exc

        data.setdefault("schema_version", 1)
        data.setdefault("run_configs", {})
        data.setdefault("search_records", {})
        data.setdefault("parameter_summaries", {})
        data.setdefault("best_results", {})
        data.setdefault("search_spaces", build_search_space_metadata())
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
        "search_spaces": build_search_space_metadata(),
        "run_configs": {},
        "search_records": {},
        "parameter_summaries": {},
        "best_results": {},
    }


def search_record_rows(results: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    records = results.get("search_records", {})
    for record_id, record in records.items():
        error = record.get("error")
        row = {
            "record_id": record_id,
            "dataset": record.get("dataset"),
            "task_type": record.get("task_type"),
            "split_method": record.get("split_method"),
            "similarity_threshold": record.get("similarity_threshold"),
            "model": record.get("model"),
            "feature_type": record.get("feature_type"),
            "input_type": record.get("input_type"),
            "parameter_id": record.get("parameter_id"),
            "random_seed": record.get("random_seed"),
            "parameters_json": compact_json(record.get("parameters", {})),
            "fixed_parameters_json": compact_json(record.get("fixed_parameters", {})),
            "training_parameters_json": compact_json(
                record.get("training_parameters", {})
            ),
            "max_len": record.get("max_len"),
            "train_size": (record.get("split_sizes") or {}).get("train"),
            "validation_size": (record.get("split_sizes") or {}).get("validation"),
            "test_size": (record.get("split_sizes") or {}).get("test"),
            "best_epoch": record.get("best_epoch"),
            "validation_metrics_json": compact_json(
                record.get("validation_metrics", {})
            )
            if record.get("validation_metrics")
            else "",
            "status": record.get("status"),
            "error_json": compact_json(error) if error else "",
            "device": record.get("device"),
            "gpu_name": record.get("gpu_name"),
            "resume_checkpoint_path": record.get("resume_checkpoint_path"),
            "started_at": record.get("started_at"),
            "finished_at": record.get("finished_at"),
        }
        rows.append(row)
    rows.sort(
        key=lambda item: (
            str(item.get("split_method")),
            str(item.get("dataset")),
            str(item.get("model")),
            str(item.get("parameter_id")),
            int(item.get("random_seed") or 0),
        )
    )
    return rows


def parameter_summary_rows(results: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    summaries = results.get("parameter_summaries", {})
    for summary_id, summary in summaries.items():
        rows.append(
            {
                "parameter_summary_id": summary_id,
                "dataset": summary.get("dataset"),
                "task_type": summary.get("task_type"),
                "split_method": summary.get("split_method"),
                "similarity_threshold": summary.get("similarity_threshold"),
                "model": summary.get("model"),
                "feature_type": summary.get("feature_type"),
                "input_type": summary.get("input_type"),
                "parameters_json": compact_json(summary.get("parameters", {})),
                "fixed_parameters_json": compact_json(
                    summary.get("fixed_parameters", {})
                ),
                "training_parameters_json": compact_json(
                    summary.get("training_parameters", {})
                ),
                "expected_seeds_json": compact_json(
                    summary.get("expected_seeds", [])
                ),
                "successful_seed_count": summary.get("successful_seed_count"),
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
        key=lambda item: (
            str(item.get("split_method")),
            str(item.get("dataset")),
            str(item.get("model")),
            str(item.get("parameter_summary_id")),
        )
    )
    return rows


def best_summary_rows(results: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    best_results = results.get("best_results", {})
    for best_id, best in best_results.items():
        model_paths = [
            item.get("model_path")
            for item in best.get("test_runs", [])
            if item.get("model_path")
        ]
        rows.append(
            {
                "best_result_id": best_id,
                "dataset": best.get("dataset"),
                "task_type": best.get("task_type"),
                "split_method": best.get("split_method"),
                "similarity_threshold": best.get("similarity_threshold"),
                "model": best.get("model"),
                "feature_type": best.get("feature_type"),
                "input_type": best.get("input_type"),
                "best_parameters_json": compact_json(best.get("best_parameters", {})),
                "fixed_parameters_json": compact_json(
                    best.get("fixed_parameters", {})
                ),
                "training_parameters_json": compact_json(
                    best.get("training_parameters", {})
                ),
                "validation_summary_json": compact_json(
                    best.get("validation_summary", {})
                ),
                "test_summary_json": compact_json(best.get("test_summary", {})),
                "model_paths_json": compact_json(model_paths),
                "selection_metric": best.get("selection_metric"),
                "selection_direction": best.get("selection_direction"),
                "status": best.get("status"),
                "best_metrics_path": best.get("best_metrics_path"),
                "updated_at": best.get("updated_at"),
            }
        )
    rows.sort(
        key=lambda item: (
            str(item.get("split_method")),
            str(item.get("dataset")),
            str(item.get("model")),
        )
    )
    return rows


def sync_csv_tables(
    results: Mapping[str, Any],
    search_csv: Path,
    parameter_summary_csv: Path,
    best_summary_csv: Path,
) -> None:
    search_rows = search_record_rows(results)
    parameter_rows = parameter_summary_rows(results)
    best_rows = best_summary_rows(results)

    atomic_dataframe_to_csv(pd.DataFrame(search_rows), search_csv)
    atomic_dataframe_to_csv(pd.DataFrame(parameter_rows), parameter_summary_csv)
    atomic_dataframe_to_csv(pd.DataFrame(best_rows), best_summary_csv)


def persist_results(
    results: Dict[str, Any],
    output_json: Path,
    search_csv: Path,
    parameter_summary_csv: Path,
    best_summary_csv: Path,
) -> None:
    results["updated_at"] = utc_now()
    atomic_json_dump(results, output_json)
    sync_csv_tables(results, search_csv, parameter_summary_csv, best_summary_csv)


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


def capture_rng_state(loader_generator: torch.Generator) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "loader_generator": loader_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any], loader_generator: torch.Generator) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if "loader_generator" in state:
        loader_generator.set_state(state["loader_generator"])
    if torch.cuda.is_available() and "torch_cuda_all" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def resolve_cuda_device(device_text: str) -> torch.device:
    if not device_text.lower().startswith("cuda"):
        raise ValueError(
            "This script is configured for GPU execution. --device must be a CUDA "
            "device such as cuda:0."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available to PyTorch. The user requested GPU-only training, "
            "so the script will not fall back to CPU."
        )
    device = torch.device(device_text)
    try:
        _ = torch.cuda.get_device_properties(device)
    except Exception as exc:
        raise RuntimeError(f"Cannot access requested CUDA device {device_text}: {exc}") from exc
    torch.cuda.set_device(device)
    return device


def gpu_name(device: torch.device) -> str:
    return torch.cuda.get_device_name(device)


def cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def release_cuda_memory(*objects: Any) -> None:
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Dataset preparation and splitting
# =============================================================================

class EncodedPeptideDataset(Dataset):
    """Integer-encoded sequences with float labels for both task types."""

    def __init__(self, dataframe: pd.DataFrame, max_len: int, task_type: str):
        if "peps" not in dataframe.columns or "label" not in dataframe.columns:
            raise KeyError("Dataset must contain 'peps' and 'label' columns.")

        sequences = dataframe["peps"].astype(str).tolist()
        encoder = IntegerEncoder(max_len=max_len)
        encoded = encoder.encode(sequences)

        labels = dataframe["label"].to_numpy(dtype=np.float32)
        if not np.all(np.isfinite(labels)):
            raise ValueError("Labels contain NaN or infinite values.")

        if task_type == "classification":
            unique = set(float(value) for value in np.unique(labels))
            if not unique.issubset({0.0, 1.0}):
                raise ValueError(
                    f"Binary classification labels must be 0/1 or 0.0/1.0; got {sorted(unique)}"
                )

        self.features = torch.as_tensor(encoded, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.aa_to_int = dict(encoder.aa_to_int)
        self.padding_idx = int(encoder.padding_idx)
        self.max_len = int(max_len)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def determine_task_type(dataset_name: str) -> str:
    return "regression" if "reg" in dataset_name.lower() else "classification"


def validate_dataframe(data: pd.DataFrame, dataset_name: str, task_type: str) -> None:
    if "peps" not in data.columns or "label" not in data.columns:
        raise KeyError(
            f"Dataset {dataset_name} must contain 'peps' and 'label' columns."
        )
    if len(data) == 0:
        raise ValueError(f"Dataset {dataset_name} is empty.")
    if data["peps"].isna().any():
        raise ValueError(f"Dataset {dataset_name} contains missing peptide sequences.")
    if data["label"].isna().any():
        raise ValueError(f"Dataset {dataset_name} contains missing labels.")

    labels = data["label"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(labels)):
        raise ValueError(f"Dataset {dataset_name} contains non-finite labels.")

    if task_type == "classification":
        unique = set(float(value) for value in np.unique(labels))
        if not unique.issubset({0.0, 1.0}):
            raise ValueError(
                f"Dataset {dataset_name} is treated as binary classification, but "
                f"labels are {sorted(unique)}."
            )


def split_dataset_for_seed(
    data: pd.DataFrame,
    task_type: str,
    split_method: str,
    test_size: float,
    val_size: float,
    random_seed: int,
    similarity_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if split_method == "similarity":
        train_data, test_data, val_data = split_dataset_by_similarity(
            data,
            test_size=test_size,
            val_size=val_size,
            random_state=random_seed,
            similarity_threshold=similarity_threshold,
        )
    elif split_method == "random":
        train_data, test_data, val_data = split_dataset(
            data,
            test_size=test_size,
            val_size=val_size,
            random_state=random_seed,
            stratify=(task_type == "classification"),
        )
    else:
        raise ValueError(f"Unsupported split method: {split_method}")

    if train_data is None or len(train_data) == 0:
        raise ValueError("Training split is empty.")
    if val_data is None or len(val_data) == 0:
        raise ValueError("Validation split is empty; validation-based search is impossible.")
    if test_data is None or len(test_data) == 0:
        raise ValueError("Test split is empty.")

    return (
        train_data.reset_index(drop=True).copy(),
        test_data.reset_index(drop=True).copy(),
        val_data.reset_index(drop=True).copy(),
    )


def prepare_split_contexts(
    data: pd.DataFrame,
    dataset_name: str,
    task_type: str,
    seeds: Sequence[int],
    split_method: str,
    test_size: float,
    val_size: float,
    similarity_threshold: float,
) -> Dict[int, Dict[str, Any]]:
    contexts: Dict[int, Dict[str, Any]] = {}
    for seed in seeds:
        print("\n" + "-" * 80)
        print(
            f"Preparing split: dataset={dataset_name}, task={task_type}, "
            f"split={split_method}, seed={seed}"
        )
        print("-" * 80)

        train_data, test_data, val_data = split_dataset_for_seed(
            data=data,
            task_type=task_type,
            split_method=split_method,
            test_size=test_size,
            val_size=val_size,
            random_seed=int(seed),
            similarity_threshold=similarity_threshold,
        )

        contexts[int(seed)] = {
            "train_data": train_data,
            "validation_data": val_data,
            "test_data": test_data,
            "split_sizes": {
                "train": int(len(train_data)),
                "validation": int(len(val_data)),
                "test": int(len(test_data)),
            },
        }
        print(
            f"Split sizes: train={len(train_data)}, validation={len(val_data)}, "
            f"test={len(test_data)}"
        )
    return contexts


def make_loader(
    dataframe: pd.DataFrame,
    max_len: int,
    task_type: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    generator: Optional[torch.Generator],
) -> Tuple[DataLoader, EncodedPeptideDataset]:
    dataset = EncodedPeptideDataset(
        dataframe=dataframe,
        max_len=max_len,
        task_type=task_type,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        generator=generator if shuffle else None,
    )
    return loader, dataset


# =============================================================================
# Model construction and metrics
# =============================================================================

def model_fixed_parameters(model_name: str, parameters: Mapping[str, Any]) -> Dict[str, Any]:
    if model_name == "lstm":
        fixed = dict(LSTM_FIXED_PARAMETERS)
        configured_dropout = float(LSTM_FIXED_PARAMETERS["dropout"])
        fixed["effective_lstm_dropout"] = (
            configured_dropout if int(parameters["num_layers"]) > 1 else 0.0
        )
        return fixed
    if model_name == "transformer":
        return dict(TRANSFORMER_FIXED_PARAMETERS)
    raise ValueError(f"Unsupported model: {model_name}")


def build_dl_model(
    model_name: str,
    task_type: str,
    parameters: Mapping[str, Any],
    max_len: int,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    fixed_parameters = model_fixed_parameters(model_name, parameters)

    if model_name == "lstm":
        model = LSTMModel(
            embedding_dim=int(parameters["embedding_dim"]),
            hidden_dim=int(parameters["hidden_dim"]),
            num_layers=int(parameters["num_layers"]),
            task=task_type,
            dropout=float(fixed_parameters["effective_lstm_dropout"]),
            device=device,
            max_len=max_len,
        )
    elif model_name == "transformer":
        model = TransformerModel(
            task=task_type,
            device=device,
            max_len=max_len,
            d_model=int(parameters["d_model"]),
            nhead=int(parameters["nhead"]),
            num_layers=int(parameters["num_layers"]),
            dim_feedforward=int(parameters["dim_feedforward"]),
            dropout=float(fixed_parameters["dropout"]),
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model = model.to(device)
    return model, fixed_parameters


def compute_classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    predictions = (probabilities >= 0.5).astype(np.int64)
    true_int = y_true.astype(np.int64)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(true_int, predictions)),
        "precision": float(
            precision_score(true_int, predictions, zero_division=0)
        ),
        "recall": float(recall_score(true_int, predictions, zero_division=0)),
        "f1": float(f1_score(true_int, predictions, zero_division=0)),
        "roc_auc": None,
        "auprc": None,
    }

    # ROC-AUC and AUPRC are only printed/saved when both classes are present.
    if len(np.unique(true_int)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(true_int, probabilities))
            metrics["auprc"] = float(
                average_precision_score(true_int, probabilities)
            )
        except ValueError:
            pass
    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    mse = float(mean_squared_error(y_true, predictions))
    metrics: Dict[str, float] = {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, predictions)),
    }
    if len(y_true) >= 2:
        r2 = float(r2_score(y_true, predictions))
        if np.isfinite(r2):
            metrics["r2"] = r2
    return metrics


def summarize_metrics(metric_dicts: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    metric_names: set[str] = set()
    for metrics in metric_dicts:
        metric_names.update(metrics.keys())

    summary: Dict[str, Dict[str, float]] = {}
    for metric_name in sorted(metric_names):
        values: List[float] = []
        for metrics in metric_dicts:
            value = metrics.get(metric_name)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                values.append(numeric)
        if values:
            summary[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "n": int(len(values)),
            }
    return summary


# =============================================================================
# Training and epoch-level resume
# =============================================================================

def create_training_components(
    model: nn.Module,
    task_type: str,
    learning_rate: float,
    weight_decay: float,
    scheduler_t_max: int,
) -> Tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    criterion: nn.Module
    if task_type == "classification":
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=scheduler_t_max,
    )
    return criterion, optimizer, scheduler


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

    for sequences, labels in loader:
        sequences = sequences.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).view(-1)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(sequences).view(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("Training loader produced no samples.")
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
        for sequences, labels in loader:
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).view(-1)
            outputs = model(sequences).view(-1)
            loss = criterion(outputs, labels)

            batch_size = int(labels.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            outputs_all.append(outputs.detach().float().cpu().numpy())
            labels_all.append(labels.detach().float().cpu().numpy())

    if total_samples == 0:
        raise RuntimeError("Evaluation loader produced no samples.")

    y_output = np.concatenate(outputs_all)
    y_true = np.concatenate(labels_all)
    metrics: Dict[str, float] = {"loss": total_loss / total_samples}
    if task_type == "classification":
        metrics.update(compute_classification_metrics(y_true, y_output))
    else:
        metrics.update(compute_regression_metrics(y_true, y_output))
    return metrics


def is_improved(
    task_type: str,
    metrics: Mapping[str, float],
    best_metric: float,
) -> bool:
    if task_type == "classification":
        return float(metrics["f1"]) > best_metric
    return float(metrics["rmse"]) < best_metric


def print_epoch_metrics(
    epoch: int,
    max_epochs: int,
    train_loss: float,
    validation_metrics: Mapping[str, float],
    current_lr: float,
    task_type: str,
) -> None:
    if task_type == "classification":
        parts = [
            f"Epoch {epoch}/{max_epochs}",
            f"train_loss={train_loss:.6f}",
            f"val_loss={validation_metrics['loss']:.6f}",
            f"val_f1={validation_metrics['f1']:.6f}",
            f"val_accuracy={validation_metrics['accuracy']:.6f}",
        ]
        if "roc_auc" in validation_metrics:
            parts.append(f"val_roc_auc={validation_metrics['roc_auc']:.6f}")
        if "auprc" in validation_metrics:
            parts.append(f"val_auprc={validation_metrics['auprc']:.6f}")
        parts.append(f"lr={current_lr:.3e}")
    else:
        parts = [
            f"Epoch {epoch}/{max_epochs}",
            f"train_loss={train_loss:.6f}",
            f"val_loss={validation_metrics['loss']:.6f}",
            f"val_rmse={validation_metrics['rmse']:.6f}",
            f"val_mae={validation_metrics['mae']:.6f}",
            f"lr={current_lr:.3e}",
        ]
    print(" | ".join(parts))


def train_with_validation(
    model_name: str,
    parameters: Mapping[str, Any],
    task_type: str,
    context: Mapping[str, Any],
    random_seed: int,
    max_len: int,
    device: torch.device,
    training_parameters: Mapping[str, Any],
    resume_checkpoint_path: Path,
) -> Tuple[Dict[str, float], int, Dict[str, Any]]:
    set_global_seed(random_seed)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(random_seed)

    train_loader, _ = make_loader(
        dataframe=context["train_data"],
        max_len=max_len,
        task_type=task_type,
        batch_size=int(training_parameters["batch_size"]),
        shuffle=True,
        num_workers=int(training_parameters["num_workers"]),
        generator=loader_generator,
    )
    val_loader, _ = make_loader(
        dataframe=context["validation_data"],
        max_len=max_len,
        task_type=task_type,
        batch_size=int(training_parameters["batch_size"]),
        shuffle=False,
        num_workers=int(training_parameters["num_workers"]),
        generator=None,
    )

    model, fixed_parameters = build_dl_model(
        model_name=model_name,
        task_type=task_type,
        parameters=parameters,
        max_len=max_len,
        device=device,
    )
    criterion, optimizer, scheduler = create_training_components(
        model=model,
        task_type=task_type,
        learning_rate=float(training_parameters["learning_rate"]),
        weight_decay=float(training_parameters["weight_decay"]),
        scheduler_t_max=int(training_parameters["scheduler_t_max"]),
    )

    max_epochs = int(training_parameters["max_epochs"])
    patience = int(training_parameters["patience"])
    log_interval = int(training_parameters["log_interval"])

    best_metric = -float("inf") if task_type == "classification" else float("inf")
    best_epoch = 0
    best_metrics: Optional[Dict[str, float]] = None
    best_model_state: Optional[Dict[str, torch.Tensor]] = None
    no_improve_count = 0
    start_epoch = 1

    if resume_checkpoint_path.exists():
        checkpoint = torch_load_compatible(resume_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_metric = float(checkpoint["best_metric"])
        best_epoch = int(checkpoint["best_epoch"])
        best_metrics = checkpoint.get("best_metrics")
        best_model_state = checkpoint.get("best_model_state_dict")
        no_improve_count = int(checkpoint.get("no_improve_count", 0))
        start_epoch = int(checkpoint["current_epoch"]) + 1
        if checkpoint.get("rng_state"):
            restore_rng_state(checkpoint["rng_state"], loader_generator)
        print(
            f"Resuming {model_name} seed={random_seed} from epoch {start_epoch} "
            f"using {resume_checkpoint_path}"
        )

    for epoch in range(start_epoch, max_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        scheduler.step()
        validation_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            task_type=task_type,
            device=device,
        )

        current_selection_metric = (
            float(validation_metrics["f1"])
            if task_type == "classification"
            else float(validation_metrics["rmse"])
        )
        if is_improved(task_type, validation_metrics, best_metric):
            best_metric = current_selection_metric
            best_epoch = epoch
            best_metrics = dict(validation_metrics)
            best_model_state = cpu_state_dict(model)
            no_improve_count = 0
        else:
            no_improve_count += 1

        current_lr = float(optimizer.param_groups[0]["lr"])
        if epoch == 1 or epoch % log_interval == 0 or no_improve_count >= patience:
            print_epoch_metrics(
                epoch=epoch,
                max_epochs=max_epochs,
                train_loss=train_loss,
                validation_metrics=validation_metrics,
                current_lr=current_lr,
                task_type=task_type,
            )

        resume_payload = {
            "mode": "validation_search",
            "model_name": model_name,
            "task_type": task_type,
            "parameters": dict(parameters),
            "fixed_parameters": fixed_parameters,
            "training_parameters": dict(training_parameters),
            "random_seed": int(random_seed),
            "current_epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_metric": float(best_metric),
            "best_epoch": int(best_epoch),
            "best_metrics": best_metrics,
            "best_model_state_dict": best_model_state,
            "no_improve_count": int(no_improve_count),
            "rng_state": capture_rng_state(loader_generator),
            "saved_at": utc_now(),
        }
        atomic_torch_save(resume_payload, resume_checkpoint_path)

        if no_improve_count >= patience:
            print(
                f"Early stopping: model={model_name}, seed={random_seed}, "
                f"epoch={epoch}, best_epoch={best_epoch}"
            )
            break

    if best_metrics is None or best_model_state is None or best_epoch <= 0:
        raise RuntimeError("Training finished without a valid best validation epoch.")

    resume_checkpoint_path.unlink(missing_ok=True)
    release_cuda_memory(model, optimizer, scheduler, criterion, train_loader, val_loader)
    return best_metrics, best_epoch, fixed_parameters


def train_fixed_epochs_and_test(
    model_name: str,
    parameters: Mapping[str, Any],
    task_type: str,
    context: Mapping[str, Any],
    random_seed: int,
    max_len: int,
    device: torch.device,
    training_parameters: Mapping[str, Any],
    final_epochs: int,
    resume_checkpoint_path: Path,
) -> Tuple[nn.Module, Dict[str, float], Dict[str, Any], Dict[str, Any]]:
    if final_epochs <= 0:
        raise ValueError(f"final_epochs must be positive; got {final_epochs}")

    set_global_seed(random_seed)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(random_seed)

    train_val_data = pd.concat(
        [context["train_data"], context["validation_data"]],
        axis=0,
        ignore_index=True,
    )
    train_loader, train_dataset = make_loader(
        dataframe=train_val_data,
        max_len=max_len,
        task_type=task_type,
        batch_size=int(training_parameters["batch_size"]),
        shuffle=True,
        num_workers=int(training_parameters["num_workers"]),
        generator=loader_generator,
    )
    test_loader, test_dataset = make_loader(
        dataframe=context["test_data"],
        max_len=max_len,
        task_type=task_type,
        batch_size=int(training_parameters["batch_size"]),
        shuffle=False,
        num_workers=int(training_parameters["num_workers"]),
        generator=None,
    )

    model, fixed_parameters = build_dl_model(
        model_name=model_name,
        task_type=task_type,
        parameters=parameters,
        max_len=max_len,
        device=device,
    )
    criterion, optimizer, scheduler = create_training_components(
        model=model,
        task_type=task_type,
        learning_rate=float(training_parameters["learning_rate"]),
        weight_decay=float(training_parameters["weight_decay"]),
        scheduler_t_max=int(training_parameters["scheduler_t_max"]),
    )

    start_epoch = 1
    if resume_checkpoint_path.exists():
        checkpoint = torch_load_compatible(resume_checkpoint_path, map_location=device)
        if int(checkpoint.get("target_epochs", final_epochs)) != int(final_epochs):
            raise RuntimeError(
                f"Final-refit resume checkpoint targets {checkpoint.get('target_epochs')} "
                f"epochs, but the current target is {final_epochs}. Remove the stale "
                f"checkpoint: {resume_checkpoint_path}"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["current_epoch"]) + 1
        if checkpoint.get("rng_state"):
            restore_rng_state(checkpoint["rng_state"], loader_generator)
        print(
            f"Resuming final refit {model_name} seed={random_seed} from epoch "
            f"{start_epoch}/{final_epochs}"
        )

    log_interval = int(training_parameters["log_interval"])
    for epoch in range(start_epoch, final_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        scheduler.step()
        if epoch == 1 or epoch % log_interval == 0 or epoch == final_epochs:
            print(
                f"Final refit | model={model_name} | seed={random_seed} | "
                f"epoch={epoch}/{final_epochs} | train_loss={train_loss:.6f} | "
                f"lr={optimizer.param_groups[0]['lr']:.3e}"
            )

        resume_payload = {
            "mode": "final_refit",
            "model_name": model_name,
            "task_type": task_type,
            "parameters": dict(parameters),
            "fixed_parameters": fixed_parameters,
            "training_parameters": dict(training_parameters),
            "random_seed": int(random_seed),
            "current_epoch": int(epoch),
            "target_epochs": int(final_epochs),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "rng_state": capture_rng_state(loader_generator),
            "saved_at": utc_now(),
        }
        atomic_torch_save(resume_payload, resume_checkpoint_path)

    test_metrics = evaluate_model(
        model=model,
        loader=test_loader,
        criterion=criterion,
        task_type=task_type,
        device=device,
    )
    resume_checkpoint_path.unlink(missing_ok=True)

    encoder_metadata = {
        "aa_to_int": train_dataset.aa_to_int,
        "padding_idx": train_dataset.padding_idx,
        "max_len": train_dataset.max_len,
    }
    # Keep the trained model alive for checkpoint saving by the caller.
    del optimizer, scheduler, criterion, train_loader, test_loader, test_dataset
    gc.collect()
    return model, test_metrics, fixed_parameters, encoder_metadata


# =============================================================================
# Experiment IDs
# =============================================================================

def training_signature(training_parameters: Mapping[str, Any], max_len: int) -> Dict[str, Any]:
    return {
        "training_parameters": dict(training_parameters),
        "max_len": int(max_len),
        "input_type": INPUT_TYPE,
    }


def make_parameter_id(
    model_name: str,
    parameters: Mapping[str, Any],
) -> str:
    return stable_id("paramset", {"model": model_name, "parameters": dict(parameters)})


def make_search_record_id(
    split_method: str,
    similarity_threshold: float,
    dataset_name: str,
    task_type: str,
    model_name: str,
    parameters: Mapping[str, Any],
    random_seed: int,
    training_parameters: Mapping[str, Any],
    max_len: int,
) -> str:
    return stable_id(
        "search",
        {
            "split_method": split_method,
            "similarity_threshold": similarity_threshold
            if split_method == "similarity"
            else None,
            "dataset": dataset_name,
            "task_type": task_type,
            "feature_type": FEATURE_TYPE,
            "model": model_name,
            "parameters": dict(parameters),
            "random_seed": int(random_seed),
            **training_signature(training_parameters, max_len),
        },
    )


def make_parameter_summary_id(
    split_method: str,
    similarity_threshold: float,
    dataset_name: str,
    task_type: str,
    model_name: str,
    parameters: Mapping[str, Any],
    training_parameters: Mapping[str, Any],
    max_len: int,
) -> str:
    return stable_id(
        "param",
        {
            "split_method": split_method,
            "similarity_threshold": similarity_threshold
            if split_method == "similarity"
            else None,
            "dataset": dataset_name,
            "task_type": task_type,
            "feature_type": FEATURE_TYPE,
            "model": model_name,
            "parameters": dict(parameters),
            **training_signature(training_parameters, max_len),
        },
    )


def make_best_result_id(
    split_method: str,
    similarity_threshold: float,
    dataset_name: str,
    task_type: str,
    model_name: str,
    training_parameters: Mapping[str, Any],
    max_len: int,
) -> str:
    return stable_id(
        "best",
        {
            "split_method": split_method,
            "similarity_threshold": similarity_threshold
            if split_method == "similarity"
            else None,
            "dataset": dataset_name,
            "task_type": task_type,
            "feature_type": FEATURE_TYPE,
            "model": model_name,
            **training_signature(training_parameters, max_len),
        },
    )


# =============================================================================
# Search for one dataset/model
# =============================================================================

def train_model_configuration(
    results: Dict[str, Any],
    output_json: Path,
    search_csv: Path,
    parameter_summary_csv: Path,
    best_summary_csv: Path,
    resume_dir: Path,
    contexts: Mapping[int, Mapping[str, Any]],
    split_method: str,
    similarity_threshold: float,
    dataset_name: str,
    task_type: str,
    model_name: str,
    seeds: Sequence[int],
    test_size: float,
    val_size: float,
    max_len: int,
    device: torch.device,
    training_parameters: Mapping[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    candidates = get_parameter_candidates(model_name)
    parameter_summaries: List[Dict[str, Any]] = []

    print("\n" + "=" * 80)
    print(
        f"Searching: dataset={dataset_name}, task={task_type}, model={model_name}, "
        f"candidates={len(candidates)}, split={split_method}"
    )
    print("=" * 80)

    for candidate_index, parameters in enumerate(candidates, start=1):
        parameter_id = make_parameter_id(model_name, parameters)
        fixed_parameters = model_fixed_parameters(model_name, parameters)
        print(
            f"\n[{candidate_index}/{len(candidates)}] {model_name} "
            f"parameters={parameters}"
        )

        successful_metrics: List[Dict[str, float]] = []
        seed_record_ids: List[str] = []

        for seed in seeds:
            seed = int(seed)
            record_id = make_search_record_id(
                split_method=split_method,
                similarity_threshold=similarity_threshold,
                dataset_name=dataset_name,
                task_type=task_type,
                model_name=model_name,
                parameters=parameters,
                random_seed=seed,
                training_parameters=training_parameters,
                max_len=max_len,
            )
            seed_record_ids.append(record_id)
            existing = results["search_records"].get(record_id)
            if existing and existing.get("status") == "success":
                print(f"  Seed {seed}: existing successful record found; skipping.")
                successful_metrics.append(existing["validation_metrics"])
                continue

            safe_split = sanitize_filename(split_method)
            safe_dataset = sanitize_filename(dataset_name)
            safe_model = sanitize_filename(model_name)
            resume_checkpoint_path = (
                resume_dir
                / safe_split
                / safe_dataset
                / safe_model
                / f"{record_id}.pt"
            )
            context = contexts[seed]
            started_at = utc_now()

            try:
                validation_metrics, best_epoch, fixed_parameters = train_with_validation(
                    model_name=model_name,
                    parameters=parameters,
                    task_type=task_type,
                    context=context,
                    random_seed=seed,
                    max_len=max_len,
                    device=device,
                    training_parameters=training_parameters,
                    resume_checkpoint_path=resume_checkpoint_path,
                )
                record = {
                    "record_id": record_id,
                    "parameter_id": parameter_id,
                    "status": "success",
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "dataset": dataset_name,
                    "task_type": task_type,
                    "split_method": split_method,
                    "similarity_threshold": similarity_threshold
                    if split_method == "similarity"
                    else None,
                    "test_size": float(test_size),
                    "validation_size": float(val_size),
                    "random_seed": seed,
                    "feature_type": FEATURE_TYPE,
                    "input_type": INPUT_TYPE,
                    "model": model_name,
                    "parameters": dict(parameters),
                    "fixed_parameters": fixed_parameters,
                    "training_parameters": dict(training_parameters),
                    "max_len": int(max_len),
                    "split_sizes": context["split_sizes"],
                    "best_epoch": int(best_epoch),
                    "validation_metrics": validation_metrics,
                    "selection_metric": "f1"
                    if task_type == "classification"
                    else "rmse",
                    "device": str(device),
                    "gpu_name": gpu_name(device),
                    "resume_checkpoint_path": str(resume_checkpoint_path),
                    "error": None,
                }
                results["search_records"][record_id] = record
                successful_metrics.append(validation_metrics)
                print(
                    f"  Seed {seed}: best_epoch={best_epoch}, "
                    f"validation_metrics={validation_metrics}"
                )
            except Exception as exc:
                record = {
                    "record_id": record_id,
                    "parameter_id": parameter_id,
                    "status": "failed",
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "dataset": dataset_name,
                    "task_type": task_type,
                    "split_method": split_method,
                    "similarity_threshold": similarity_threshold
                    if split_method == "similarity"
                    else None,
                    "test_size": float(test_size),
                    "validation_size": float(val_size),
                    "random_seed": seed,
                    "feature_type": FEATURE_TYPE,
                    "input_type": INPUT_TYPE,
                    "model": model_name,
                    "parameters": dict(parameters),
                    "fixed_parameters": fixed_parameters,
                    "training_parameters": dict(training_parameters),
                    "max_len": int(max_len),
                    "split_sizes": context["split_sizes"],
                    "best_epoch": None,
                    "validation_metrics": None,
                    "selection_metric": "f1"
                    if task_type == "classification"
                    else "rmse",
                    "device": str(device),
                    "gpu_name": gpu_name(device),
                    "resume_checkpoint_path": str(resume_checkpoint_path),
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                }
                results["search_records"][record_id] = record
                print(f"  Seed {seed}: FAILED - {exc}")

            # Save after every specific parameter + seed combination.
            persist_results(
                results,
                output_json,
                search_csv,
                parameter_summary_csv,
                best_summary_csv,
            )
            release_cuda_memory()

        parameter_summary_id = make_parameter_summary_id(
            split_method=split_method,
            similarity_threshold=similarity_threshold,
            dataset_name=dataset_name,
            task_type=task_type,
            model_name=model_name,
            parameters=parameters,
            training_parameters=training_parameters,
            max_len=max_len,
        )
        complete = len(successful_metrics) == len(seeds)
        validation_summary = summarize_metrics(successful_metrics)
        parameter_summary = {
            "parameter_summary_id": parameter_summary_id,
            "parameter_id": parameter_id,
            "status": "complete" if complete else "incomplete",
            "dataset": dataset_name,
            "task_type": task_type,
            "split_method": split_method,
            "similarity_threshold": similarity_threshold
            if split_method == "similarity"
            else None,
            "test_size": float(test_size),
            "validation_size": float(val_size),
            "feature_type": FEATURE_TYPE,
            "input_type": INPUT_TYPE,
            "model": model_name,
            "parameters": dict(parameters),
            "fixed_parameters": fixed_parameters,
            "training_parameters": dict(training_parameters),
            "max_len": int(max_len),
            "expected_seeds": [int(seed) for seed in seeds],
            "successful_seed_count": int(len(successful_metrics)),
            "search_record_ids": seed_record_ids,
            "validation_summary": validation_summary,
            "selection_metric": "f1"
            if task_type == "classification"
            else "rmse",
            "selection_direction": "maximize"
            if task_type == "classification"
            else "minimize",
            "updated_at": utc_now(),
        }
        results["parameter_summaries"][parameter_summary_id] = parameter_summary
        parameter_summaries.append(parameter_summary)
        persist_results(
            results,
            output_json,
            search_csv,
            parameter_summary_csv,
            best_summary_csv,
        )

    eligible = [
        summary
        for summary in parameter_summaries
        if summary.get("status") == "complete"
        and summary.get("selection_metric")
        in summary.get("validation_summary", {})
    ]
    if not eligible:
        raise RuntimeError(
            f"No complete parameter candidate is available for "
            f"{dataset_name}/{model_name}."
        )

    if task_type == "classification":
        best_summary = max(
            eligible,
            key=lambda item: (
                item["validation_summary"]["f1"]["mean"],
                -item["validation_summary"]["f1"]["std"],
            ),
        )
    else:
        best_summary = min(
            eligible,
            key=lambda item: (
                item["validation_summary"]["rmse"]["mean"],
                item["validation_summary"]["rmse"]["std"],
            ),
        )

    print("\nBest validation parameter summary:")
    print(json.dumps(to_serializable(best_summary), indent=2, ensure_ascii=False))
    return best_summary, parameter_summaries


# =============================================================================
# Final refit and saving
# =============================================================================

def find_search_record_for_best_seed(
    results: Mapping[str, Any],
    best_parameter_summary: Mapping[str, Any],
    seed: int,
) -> Mapping[str, Any]:
    for record_id in best_parameter_summary.get("search_record_ids", []):
        record = results.get("search_records", {}).get(record_id)
        if (
            record
            and int(record.get("random_seed")) == int(seed)
            and record.get("status") == "success"
        ):
            return record
    raise RuntimeError(
        f"Cannot find a successful search record for best parameters and seed={seed}."
    )


def save_final_model_checkpoint(
    output_path: Path,
    model: nn.Module,
    metadata: Mapping[str, Any],
) -> None:
    payload = dict(metadata)
    payload["state_dict"] = cpu_state_dict(model)
    atomic_torch_save(payload, output_path)


def load_saved_dl_model(
    checkpoint_path: str | Path,
    device_text: str = "cuda:0",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Rebuild a best LSTM/Transformer model from a saved self-describing checkpoint."""
    device = torch.device(device_text)
    checkpoint = torch_load_compatible(Path(checkpoint_path), map_location=device)
    model, _ = build_dl_model(
        model_name=checkpoint["model_name"],
        task_type=checkpoint["task_type"],
        parameters=checkpoint["model_parameters"],
        max_len=int(checkpoint["max_len"]),
        device=device,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def finalize_best_model(
    results: Dict[str, Any],
    output_json: Path,
    search_csv: Path,
    parameter_summary_csv: Path,
    best_summary_csv: Path,
    best_model_dir: Path,
    best_metrics_dir: Path,
    resume_dir: Path,
    contexts: Mapping[int, Mapping[str, Any]],
    best_parameter_summary: Mapping[str, Any],
    split_method: str,
    similarity_threshold: float,
    dataset_name: str,
    task_type: str,
    model_name: str,
    seeds: Sequence[int],
    max_len: int,
    device: torch.device,
    training_parameters: Mapping[str, Any],
) -> Dict[str, Any]:
    best_result_id = make_best_result_id(
        split_method=split_method,
        similarity_threshold=similarity_threshold,
        dataset_name=dataset_name,
        task_type=task_type,
        model_name=model_name,
        training_parameters=training_parameters,
        max_len=max_len,
    )

    parameters = dict(best_parameter_summary["parameters"])
    fixed_parameters = dict(best_parameter_summary["fixed_parameters"])
    safe_split = sanitize_filename(split_method)
    safe_dataset = sanitize_filename(dataset_name)
    safe_model = sanitize_filename(model_name)
    combination_model_dir = best_model_dir / safe_split / safe_dataset
    combination_metrics_dir = best_metrics_dir / safe_split / safe_dataset
    combination_metrics_path = combination_metrics_dir / f"{safe_model}.json"

    existing_best = results["best_results"].get(best_result_id, {})
    same_selection = (
        existing_best.get("best_parameters") == parameters
        and existing_best.get("parameter_summary_id")
        == best_parameter_summary.get("parameter_summary_id")
    )
    existing_seed_results = {
        int(item["random_seed"]): item
        for item in existing_best.get("test_runs", [])
        if same_selection and item.get("status") == "success"
    }

    seed_test_results: List[Dict[str, Any]] = []
    for seed in seeds:
        seed = int(seed)
        model_path = combination_model_dir / f"{safe_model}__seed{seed}.pt"
        existing_seed = existing_seed_results.get(seed)
        if existing_seed and model_path.exists():
            print(
                f"Final model already exists for {dataset_name}/{model_name}/"
                f"seed={seed}; skipping refit."
            )
            seed_test_results.append(existing_seed)
            continue

        best_seed_search_record = find_search_record_for_best_seed(
            results=results,
            best_parameter_summary=best_parameter_summary,
            seed=seed,
        )
        final_epochs = int(best_seed_search_record["best_epoch"])
        final_resume_path = (
            resume_dir
            / safe_split
            / safe_dataset
            / safe_model
            / "final_refit"
            / f"{best_parameter_summary['parameter_summary_id']}__seed{seed}.pt"
        )
        context = contexts[seed]
        started_at = utc_now()

        try:
            model, test_metrics, effective_fixed_parameters, encoder_metadata = (
                train_fixed_epochs_and_test(
                    model_name=model_name,
                    parameters=parameters,
                    task_type=task_type,
                    context=context,
                    random_seed=seed,
                    max_len=max_len,
                    device=device,
                    training_parameters=training_parameters,
                    final_epochs=final_epochs,
                    resume_checkpoint_path=final_resume_path,
                )
            )

            model_metadata = {
                "checkpoint_schema_version": 1,
                "model_name": model_name,
                "task_type": task_type,
                "dataset": dataset_name,
                "split_method": split_method,
                "similarity_threshold": similarity_threshold
                if split_method == "similarity"
                else None,
                "feature_type": FEATURE_TYPE,
                "input_type": INPUT_TYPE,
                "model_parameters": parameters,
                "fixed_parameters": effective_fixed_parameters,
                "training_parameters": dict(training_parameters),
                "max_len": int(max_len),
                "random_seed": seed,
                "selection_best_epoch": final_epochs,
                "final_training_epochs": final_epochs,
                "final_training_data": "train+validation",
                "split_sizes": context["split_sizes"],
                "refit_train_size": int(
                    context["split_sizes"]["train"]
                    + context["split_sizes"]["validation"]
                ),
                "test_metrics": test_metrics,
                "integer_encoder": encoder_metadata,
                "created_at": utc_now(),
            }
            save_final_model_checkpoint(
                output_path=model_path,
                model=model,
                metadata=model_metadata,
            )
            release_cuda_memory(model)

            seed_result = {
                "status": "success",
                "random_seed": seed,
                "started_at": started_at,
                "finished_at": utc_now(),
                "model_path": str(model_path),
                "final_training_epochs": final_epochs,
                "fixed_parameters": effective_fixed_parameters,
                "split_sizes": context["split_sizes"],
                "refit_train_size": int(
                    context["split_sizes"]["train"]
                    + context["split_sizes"]["validation"]
                ),
                "test_metrics": test_metrics,
                "error": None,
            }
            seed_test_results.append(seed_result)
            print(f"Final seed {seed} test metrics: {test_metrics}")
        except Exception as exc:
            seed_test_results.append(
                {
                    "status": "failed",
                    "random_seed": seed,
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "model_path": str(model_path),
                    "final_training_epochs": final_epochs,
                    "fixed_parameters": fixed_parameters,
                    "split_sizes": context["split_sizes"],
                    "test_metrics": None,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                }
            )
            print(f"Final refit FAILED for seed {seed}: {exc}")

        successful_test_metrics = [
            item["test_metrics"]
            for item in seed_test_results
            if item.get("status") == "success" and item.get("test_metrics")
        ]
        partial_best_result = {
            "best_result_id": best_result_id,
            "status": "complete"
            if len(successful_test_metrics) == len(seeds)
            else "incomplete",
            "dataset": dataset_name,
            "task_type": task_type,
            "split_method": split_method,
            "similarity_threshold": similarity_threshold
            if split_method == "similarity"
            else None,
            "feature_type": FEATURE_TYPE,
            "input_type": INPUT_TYPE,
            "model": model_name,
            "selection_metric": "f1"
            if task_type == "classification"
            else "rmse",
            "selection_direction": "maximize"
            if task_type == "classification"
            else "minimize",
            "best_parameters": parameters,
            "fixed_parameters": fixed_parameters,
            "training_parameters": dict(training_parameters),
            "parameter_summary_id": best_parameter_summary["parameter_summary_id"],
            "validation_summary": best_parameter_summary["validation_summary"],
            "refit_strategy": (
                "merge_train_and_validation_then_train_for_seed_specific_best_epoch"
            ),
            "test_runs": seed_test_results,
            "test_summary": summarize_metrics(successful_test_metrics),
            "best_metrics_path": str(combination_metrics_path),
            "updated_at": utc_now(),
        }
        results["best_results"][best_result_id] = partial_best_result
        persist_results(
            results,
            output_json,
            search_csv,
            parameter_summary_csv,
            best_summary_csv,
        )
        write_metrics_report(combination_metrics_path, partial_best_result)

    successful_test_metrics = [
        item["test_metrics"]
        for item in seed_test_results
        if item.get("status") == "success" and item.get("test_metrics")
    ]
    best_result = {
        "best_result_id": best_result_id,
        "status": "complete"
        if len(successful_test_metrics) == len(seeds)
        else "incomplete",
        "dataset": dataset_name,
        "task_type": task_type,
        "split_method": split_method,
        "similarity_threshold": similarity_threshold
        if split_method == "similarity"
        else None,
        "feature_type": FEATURE_TYPE,
        "input_type": INPUT_TYPE,
        "model": model_name,
        "selection_metric": "f1" if task_type == "classification" else "rmse",
        "selection_direction": "maximize"
        if task_type == "classification"
        else "minimize",
        "best_parameters": parameters,
        "fixed_parameters": fixed_parameters,
        "training_parameters": dict(training_parameters),
        "parameter_summary_id": best_parameter_summary["parameter_summary_id"],
        "validation_summary": best_parameter_summary["validation_summary"],
        "refit_strategy": (
            "merge_train_and_validation_then_train_for_seed_specific_best_epoch"
        ),
        "test_runs": seed_test_results,
        "test_summary": summarize_metrics(successful_test_metrics),
        "best_metrics_path": str(combination_metrics_path),
        "updated_at": utc_now(),
    }
    results["best_results"][best_result_id] = best_result
    persist_results(
        results,
        output_json,
        search_csv,
        parameter_summary_csv,
        best_summary_csv,
    )
    write_metrics_report(combination_metrics_path, best_result)
    return best_result


# =============================================================================
# Main experiment loop
# =============================================================================

def collect_csv_files(data_dir: Path, datasets: Optional[Sequence[str]]) -> List[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    if datasets:
        files = [data_dir / f"{dataset}.csv" for dataset in datasets]
        missing = [str(path) for path in files if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Requested dataset files not found: {missing}")
        return files

    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV datasets found in {data_dir}")
    return files


def run_experiments(args: argparse.Namespace) -> None:
    device = resolve_cuda_device(args.device)
    output_json = Path(args.output_json).expanduser().resolve()
    search_csv = Path(args.search_csv).expanduser().resolve()
    parameter_summary_csv = Path(args.parameter_summary_csv).expanduser().resolve()
    best_summary_csv = Path(args.best_summary_csv).expanduser().resolve()
    best_model_dir = Path(args.best_model_dir).expanduser().resolve()
    best_metrics_dir = Path(args.best_metrics_dir).expanduser().resolve()
    resume_dir = Path(args.resume_dir).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()

    training_parameters: Dict[str, Any] = {
        "optimizer": "Adam",
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": int(args.scheduler_t_max),
        "num_workers": int(args.num_workers),
        "log_interval": int(args.log_interval),
        "classification_loss": "BCELoss",
        "regression_loss": "MSELoss",
    }

    results = load_or_initialize_results(output_json)
    results.setdefault("environment", {})
    results["environment"].update(
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu_name": gpu_name(device),
        }
    )

    run_config_key = stable_id(
        "run",
        {
            "split_method": args.split_method,
            "similarity_threshold": args.similarity_threshold
            if args.split_method == "similarity"
            else None,
            "test_size": args.test_size,
            "val_size": args.val_size,
            "seeds": args.seeds,
            "models": args.models,
            "datasets": args.datasets,
            "max_len": args.max_len,
            "training_parameters": training_parameters,
        },
    )
    results["run_configs"][run_config_key] = {
        "run_config_id": run_config_key,
        "started_at": utc_now(),
        "split_method": args.split_method,
        "similarity_threshold": args.similarity_threshold
        if args.split_method == "similarity"
        else None,
        "test_size": float(args.test_size),
        "validation_size": float(args.val_size),
        "seeds": [int(seed) for seed in args.seeds],
        "models": list(args.models),
        "feature_type": FEATURE_TYPE,
        "input_type": INPUT_TYPE,
        "requested_datasets": list(args.datasets) if args.datasets else None,
        "data_dir": str(data_dir),
        "max_len": int(args.max_len),
        "training_parameters": training_parameters,
        "output_json": str(output_json),
        "search_csv": str(search_csv),
        "parameter_summary_csv": str(parameter_summary_csv),
        "best_summary_csv": str(best_summary_csv),
        "best_model_dir": str(best_model_dir),
        "best_metrics_dir": str(best_metrics_dir),
        "resume_dir": str(resume_dir),
        "device": str(device),
        "gpu_name": gpu_name(device),
        "status": "running",
    }
    persist_results(
        results,
        output_json,
        search_csv,
        parameter_summary_csv,
        best_summary_csv,
    )

    print("=" * 80)
    print("Deep Learning Fixed-Parameter Training")
    print("=" * 80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"Device: {device} ({gpu_name(device)})")
    print(f"Split method: {args.split_method}")
    print(f"Models: {args.models}")
    print(f"Seeds: {args.seeds}")
    print(f"LSTM candidates: {len(build_lstm_candidates())}")
    print(f"Transformer candidates: {len(build_transformer_candidates())}")
    print(f"Training parameters: {training_parameters}")

    csv_files = collect_csv_files(data_dir, args.datasets)
    print(f"Found {len(csv_files)} datasets: {[path.stem for path in csv_files]}")

    dataset_failures: List[Dict[str, Any]] = []
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
            data = pd.read_csv(csv_file)
            validate_dataframe(data, dataset_name, task_type)
            print(f"Loaded {dataset_name}: {len(data)} samples")

            contexts = prepare_split_contexts(
                data=data,
                dataset_name=dataset_name,
                task_type=task_type,
                seeds=args.seeds,
                split_method=args.split_method,
                test_size=args.test_size,
                val_size=args.val_size,
                similarity_threshold=args.similarity_threshold,
            )

            for model_name in args.models:
                try:
                    best_parameter_summary, _ = train_model_configuration(
                        results=results,
                        output_json=output_json,
                        search_csv=search_csv,
                        parameter_summary_csv=parameter_summary_csv,
                        best_summary_csv=best_summary_csv,
                        resume_dir=resume_dir,
                        contexts=contexts,
                        split_method=args.split_method,
                        similarity_threshold=args.similarity_threshold,
                        dataset_name=dataset_name,
                        task_type=task_type,
                        model_name=model_name,
                        seeds=args.seeds,
                        test_size=args.test_size,
                        val_size=args.val_size,
                        max_len=args.max_len,
                        device=device,
                        training_parameters=training_parameters,
                    )
                    finalize_best_model(
                        results=results,
                        output_json=output_json,
                        search_csv=search_csv,
                        parameter_summary_csv=parameter_summary_csv,
                        best_summary_csv=best_summary_csv,
                        best_model_dir=best_model_dir,
                        best_metrics_dir=best_metrics_dir,
                        resume_dir=resume_dir,
                        contexts=contexts,
                        best_parameter_summary=best_parameter_summary,
                        split_method=args.split_method,
                        similarity_threshold=args.similarity_threshold,
                        dataset_name=dataset_name,
                        task_type=task_type,
                        model_name=model_name,
                        seeds=args.seeds,
                        max_len=args.max_len,
                        device=device,
                        training_parameters=training_parameters,
                    )
                except Exception as exc:
                    failure = {
                        "dataset": dataset_name,
                        "task_type": task_type,
                        "model": model_name,
                        "split_method": args.split_method,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                        "time": utc_now(),
                    }
                    dataset_failures.append(failure)
                    results.setdefault("combination_failures", []).append(failure)
                    print(f"FAILED model combination {dataset_name}/{model_name}: {exc}")
                    persist_results(
                        results,
                        output_json,
                        search_csv,
                        parameter_summary_csv,
                        best_summary_csv,
                    )
                    release_cuda_memory()

            del contexts
            gc.collect()
        except Exception as exc:
            failure = {
                "dataset": dataset_name,
                "task_type": task_type,
                "model": None,
                "split_method": args.split_method,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "time": utc_now(),
            }
            dataset_failures.append(failure)
            results.setdefault("dataset_failures", []).append(failure)
            print(f"FAILED dataset {dataset_name}: {exc}")
            persist_results(
                results,
                output_json,
                search_csv,
                parameter_summary_csv,
                best_summary_csv,
            )

    results["run_configs"][run_config_key]["finished_at"] = utc_now()
    results["run_configs"][run_config_key]["status"] = (
        "completed_with_failures" if dataset_failures else "completed"
    )
    results["run_configs"][run_config_key]["failure_count"] = len(dataset_failures)
    persist_results(
        results,
        output_json,
        search_csv,
        parameter_summary_csv,
        best_summary_csv,
    )

    print("\n" + "=" * 80)
    print("Run finished")
    print("=" * 80)
    print(f"Global JSON: {output_json}")
    print(f"Search table: {search_csv}")
    print(f"Parameter summary table: {parameter_summary_csv}")
    print(f"Best summary table: {best_summary_csv}")
    print(f"Best models: {best_model_dir}")
    print(f"Best metrics: {best_metrics_dir}")
    print(f"Failures in this run: {len(dataset_failures)}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train fixed LSTM and Transformer configurations on all "
            "peptide classification and regression datasets."
        )
    )
    parser.add_argument(
        "--split_method",
        required=True,
        choices=["random", "similarity"],
        help="Run the script separately for random and similarity splitting.",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.8,
        help="Similarity threshold used only for similarity splitting.",
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="Random seeds for splitting and training.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(SUPPORTED_MODELS),
        default=list(SUPPORTED_MODELS),
        help="Models to train; default: lstm transformer.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset stems. Omit to process every CSV in data_dir.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="pephub/raw_data",
        help="Directory containing the original CSV datasets.",
    )
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--scheduler_t_max", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Required CUDA device, for example cuda:0.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="outputs/dl/run_state.json",
        help="Random and similarity runs may append to the same global JSON.",
    )
    parser.add_argument(
        "--search_csv",
        type=str,
        default="outputs/dl/training_runs.csv",
        help="One-row-per-parameter/seed search table.",
    )
    parser.add_argument(
        "--parameter_summary_csv",
        type=str,
        default="outputs/dl/parameter_summary.csv",
    )
    parser.add_argument(
        "--best_summary_csv",
        type=str,
        default="outputs/dl/summary.csv",
    )
    parser.add_argument(
        "--best_model_dir",
        type=str,
        default="outputs/models/dl",
    )
    parser.add_argument(
        "--best_metrics_dir",
        type=str,
        default="outputs/metrics/dl",
    )
    parser.add_argument(
        "--resume_dir",
        type=str,
        default="outputs/checkpoints/dl",
    )

    args = parser.parse_args()
    if not 0.0 < args.test_size < 1.0:
        parser.error("--test_size must be between 0 and 1.")
    if not 0.0 < args.val_size < 1.0:
        parser.error("--val_size must be between 0 and 1.")
    if args.test_size + args.val_size >= 1.0:
        parser.error("--test_size + --val_size must be less than 1.")
    if not args.seeds:
        parser.error("At least one seed is required.")
    if args.max_len <= 0:
        parser.error("--max_len must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive.")
    if args.max_epochs <= 0:
        parser.error("--max_epochs must be positive.")
    if args.patience <= 0:
        parser.error("--patience must be positive.")
    if args.scheduler_t_max <= 0:
        parser.error("--scheduler_t_max must be positive.")
    if args.num_workers < 0:
        parser.error("--num_workers cannot be negative.")
    if args.log_interval <= 0:
        parser.error("--log_interval must be positive.")

    args.seeds = list(dict.fromkeys(args.seeds))
    args.models = list(dict.fromkeys(args.models))
    if args.datasets:
        args.datasets = list(dict.fromkeys(args.datasets))
    return args


def main() -> None:
    args = parse_args()
    run_experiments(args)


if __name__ == "__main__":
    main()
