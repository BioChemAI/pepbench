#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train and evaluate traditional peptide-property machine-learning models.

The script uses fixed model configurations and preserves the same data
preparation, validation, refit, and test procedure across model families.

Models
------
- Random Forest: classification and regression
- SVM/SVR: classification and regression
- XGBoost: classification and regression

Features
--------
Each feature is evaluated independently; features are NOT concatenated:
- descriptor
- onehot
- frequency

Evaluation protocol
-------------------
Each fixed model configuration is trained with every requested random seed.
The test set is untouched until the model is refitted on the combined training
and validation data.

Outputs
-------
1. One global JSON file containing:
   - every parameter/seed validation result;
   - parameter summaries across seeds;
   - the selected best parameters and final test results.
2. Fitted models under ``outputs/models/ml``.
3. Versioned metrics JSON reports under ``outputs/metrics/ml``.

Successful seed records in the output JSON are reused after interruption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from pephub.results import write_metrics_report

from pephub.dataset import PepDataset
from pephub.featurizer import PeptideFeaturizer
from pephub.splitter import split_dataset, split_dataset_by_similarity

# -----------------------------------------------------------------------------
# ML imports.
# -----------------------------------------------------------------------------
try:
    import sklearn
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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
    from sklearn.preprocessing import StandardScaler, label_binarize
    from sklearn.svm import SVC, SVR
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required. Install the project environment before running."
    ) from exc

try:
    import xgboost as xgb
except ImportError as exc:
    raise ImportError(
        "xgboost is required. Install the project environment before running."
    ) from exc


# =============================================================================
# Search spaces
# =============================================================================

RF_CLASSIFICATION_GRID: Dict[str, Sequence[Any]] = {
    "n_estimators": [300],
    "max_depth": [None],
    "min_samples_leaf": [1],
    "max_features": ["sqrt"],
}

RF_REGRESSION_GRID: Dict[str, Sequence[Any]] = {
    "n_estimators": [300],
    "max_depth": [None],
    "min_samples_leaf": [1],
    "max_features": [1.0],
}

SVC_GRID: List[Dict[str, Sequence[Any]]] = [
    {"kernel": ["linear"], "C": [1.0]},
]

SVR_GRID: List[Dict[str, Sequence[Any]]] = [
    {"kernel": ["rbf"], "C": [10.0], "gamma": ["scale"], "epsilon": [0.1]},
]

XGB_GRID: Dict[str, Sequence[Any]] = {
    "n_estimators": [300],
    "max_depth": [6],
    "learning_rate": [0.1],
    "min_child_weight": [1],
    "subsample": [0.8],
}

SUPPORTED_FEATURES = ("descriptor", "onehot", "frequency")
SUPPORTED_MODELS = ("random_forest", "svm", "xgboost")


# =============================================================================
# General helpers
# =============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_filename(value: str) -> str:
    """Convert arbitrary names to safe file/directory names."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return cleaned or "unnamed"


def to_serializable(obj: Any) -> Any:
    """Recursively convert NumPy/Path values to JSON-serializable objects."""
    if isinstance(obj, Path):
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


def stable_id(prefix: str, payload: Mapping[str, Any], length: int = 24) -> str:
    normalized = json.dumps(
        to_serializable(dict(payload)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def atomic_json_dump(data: Mapping[str, Any], output_path: Path) -> None:
    """Atomically replace a JSON file so interruptions do not corrupt it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = to_serializable(data)

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=str(output_path.parent),
        text=True,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def load_or_initialize_results(output_path: Path) -> Dict[str, Any]:
    if output_path.exists():
        try:
            with output_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Existing result file is not valid JSON: {output_path}"
            ) from exc

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
            "xgboost": xgb.__version__,
        },
        "search_spaces": build_search_space_metadata(),
        "run_configs": {},
        "search_records": {},
        "parameter_summaries": {},
        "best_results": {},
    }


def persist_results(results: Dict[str, Any], output_path: Path) -> None:
    results["updated_at"] = utc_now()
    atomic_json_dump(results, output_path)


def build_search_space_metadata() -> Dict[str, Any]:
    return {
        "random_forest": {
            "classification": RF_CLASSIFICATION_GRID,
            "regression": RF_REGRESSION_GRID,
        },
        "svm": {
            "classification": SVC_GRID,
            "regression": SVR_GRID,
        },
        "xgboost": {
            "classification": XGB_GRID,
            "regression": XGB_GRID,
        },
        "selection": {
            "classification": {"metric": "f1", "direction": "maximize"},
            "regression": {"metric": "rmse", "direction": "minimize"},
        },
    }


def summarize_metrics(metric_dicts: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Calculate mean/std for every numeric metric available in all runs."""
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


def determine_task_type(dataset_name: str) -> str:
    """Preserve the original filename-based task detection convention."""
    return "regression" if "reg" in dataset_name.lower() else "classification"


def get_parameter_grid(model_name: str, task_type: str) -> List[Dict[str, Any]]:
    if model_name == "random_forest":
        grid = RF_CLASSIFICATION_GRID if task_type == "classification" else RF_REGRESSION_GRID
    elif model_name == "svm":
        grid = SVC_GRID if task_type == "classification" else SVR_GRID
    elif model_name == "xgboost":
        grid = XGB_GRID
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    fixed_parameters = grid[0] if isinstance(grid, list) else grid
    return [{name: values[0] for name, values in fixed_parameters.items()}]


# =============================================================================
# Feature extraction: preserve the original logic.
# =============================================================================

def extract_features(
    sequences: List[str],
    feature_type: str = "descriptor",
    padding_len: Optional[int] = None,
    descriptor_list: Optional[List[str]] = None,
    esm_model_name: Optional[str] = None,
    esm_pooling: str = "mean",
    device: Optional[str] = None,
) -> np.ndarray:
    """Extract one independent feature representation for peptide sequences."""
    if feature_type in ["onehot", "integer", "blosum62"] and padding_len is None:
        padding_len = max(len(seq) for seq in sequences)
        print(f"Auto-detected max sequence length: {padding_len}")

    featurizer_kwargs: Dict[str, Any] = {
        "feature_type": feature_type,
        "padding_len": padding_len
        if feature_type in ["onehot", "integer", "blosum62"]
        else None,
    }

    if feature_type == "descriptor" and descriptor_list is not None:
        featurizer_kwargs["descriptor_list"] = descriptor_list

    if feature_type == "esm":
        if esm_model_name is not None:
            featurizer_kwargs["esm_model_name"] = esm_model_name
        featurizer_kwargs["esm_pooling"] = esm_pooling
        if device is not None:
            featurizer_kwargs["device"] = device

    featurizer = PeptideFeaturizer(**featurizer_kwargs)

    features: List[np.ndarray] = []
    print(f"Extracting {feature_type.upper()} features for {len(sequences)} sequences...")
    for index, sequence in enumerate(sequences):
        if (index + 1) % 100 == 0:
            print(f"  Processed {index + 1}/{len(sequences)} sequences")

        feature = featurizer.transform(sequence)
        if not isinstance(feature, np.ndarray):
            raise ValueError(f"Unexpected feature type: {type(feature)}")
        if feature.ndim > 1:
            feature = feature.flatten()
        features.append(feature)

    feature_matrix = np.asarray(features)

    if feature_type in ["onehot", "integer", "blosum62"] and features:
        expected_length = len(features[0])
        for index, feature in enumerate(features):
            if len(feature) != expected_length:
                raise ValueError(
                    f"Feature length mismatch: sequence {index} has length {len(feature)}, "
                    f"expected {expected_length}."
                )

    if not np.issubdtype(feature_matrix.dtype, np.number):
        feature_matrix = feature_matrix.astype(np.float32)
    else:
        feature_matrix = feature_matrix.astype(np.float32, copy=False)

    if not np.all(np.isfinite(feature_matrix)):
        raise ValueError(
            f"{feature_type} features contain NaN or infinite values. "
            "Please inspect the featurizer output."
        )

    print(f"Feature extraction complete. Feature shape: {feature_matrix.shape}")
    return feature_matrix


# =============================================================================
# Data splitting and per-seed feature preparation
# =============================================================================

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

    if val_data is None or len(val_data) == 0:
        raise ValueError(
            "Validation data is empty. Hyperparameter selection requires a validation set."
        )

    return train_data, test_data, val_data


def prepare_feature_contexts(
    data: pd.DataFrame,
    dataset_name: str,
    task_type: str,
    feature_type: str,
    seeds: Sequence[int],
    split_method: str,
    test_size: float,
    val_size: float,
    similarity_threshold: float,
    descriptor_list: Optional[List[str]],
) -> Dict[int, Dict[str, Any]]:
    """Split and extract a single feature representation once per seed."""
    if feature_type == "onehot":
        # Use one dataset-level padding length so all seeds have the same dimension.
        padding_len = max(len(sequence) for sequence in data["peps"].tolist())
    else:
        padding_len = None

    contexts: Dict[int, Dict[str, Any]] = {}

    for seed in seeds:
        print("\n" + "-" * 80)
        print(
            f"Preparing data: dataset={dataset_name}, feature={feature_type}, "
            f"split={split_method}, seed={seed}"
        )
        print("-" * 80)

        train_data, test_data, val_data = split_dataset_for_seed(
            data=data,
            task_type=task_type,
            split_method=split_method,
            test_size=test_size,
            val_size=val_size,
            random_seed=seed,
            similarity_threshold=similarity_threshold,
        )

        X_train = extract_features(
            train_data["peps"].tolist(),
            feature_type=feature_type,
            padding_len=padding_len,
            descriptor_list=descriptor_list,
        )
        X_val = extract_features(
            val_data["peps"].tolist(),
            feature_type=feature_type,
            padding_len=padding_len,
            descriptor_list=descriptor_list,
        )
        X_test = extract_features(
            test_data["peps"].tolist(),
            feature_type=feature_type,
            padding_len=padding_len,
            descriptor_list=descriptor_list,
        )

        y_train = train_data["label"].to_numpy()
        y_val = val_data["label"].to_numpy()
        y_test = test_data["label"].to_numpy()

        if task_type == "regression":
            y_train = y_train.astype(np.float64)
            y_val = y_val.astype(np.float64)
            y_test = y_test.astype(np.float64)

        contexts[int(seed)] = {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "feature_dimension": int(X_train.shape[1]),
            "padding_len": padding_len,
            "split_sizes": {
                "train": int(len(train_data)),
                "validation": int(len(val_data)),
                "test": int(len(test_data)),
            },
        }

        print(
            f"Split sizes: train={len(train_data)}, validation={len(val_data)}, "
            f"test={len(test_data)}; feature_dim={X_train.shape[1]}"
        )

    return contexts


# =============================================================================
# Metric calculation
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
    model: Optional[Any] = None,
    X_eval: Optional[np.ndarray] = None,
) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {}

    if task_type == "classification":
        unique_labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
        binary = len(unique_labels) == 2
        average = "binary" if binary else "macro"
        positive_label = unique_labels[-1] if binary else None

        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision"] = float(
            precision_score(
                y_true,
                y_pred,
                average=average,
                pos_label=positive_label if binary else 1,
                zero_division=0,
            )
        )
        metrics["recall"] = float(
            recall_score(
                y_true,
                y_pred,
                average=average,
                pos_label=positive_label if binary else 1,
                zero_division=0,
            )
        )
        metrics["f1"] = float(
            f1_score(
                y_true,
                y_pred,
                average=average,
                pos_label=positive_label if binary else 1,
                zero_division=0,
            )
        )

        metrics["roc_auc"] = None
        metrics["auprc"] = None

        if model is not None and X_eval is not None and hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(X_eval)
                classes = np.asarray(model.classes_)

                if len(classes) == 2:
                    positive_class = classes[-1]
                    positive_index = int(np.where(classes == positive_class)[0][0])
                    y_binary = (np.asarray(y_true) == positive_class).astype(int)
                    positive_scores = probabilities[:, positive_index]
                    metrics["roc_auc"] = float(roc_auc_score(y_binary, positive_scores))
                    metrics["auprc"] = float(
                        average_precision_score(y_binary, positive_scores)
                    )
                else:
                    y_binary = label_binarize(y_true, classes=classes)
                    metrics["roc_auc"] = float(
                        roc_auc_score(
                            y_binary,
                            probabilities,
                            average="macro",
                            multi_class="ovr",
                        )
                    )
                    metrics["auprc"] = float(
                        average_precision_score(
                            y_binary,
                            probabilities,
                            average="macro",
                        )
                    )
            except Exception as exc:
                warnings.warn(f"Could not compute ROC-AUC/AUPRC: {exc}")

    else:
        mse = float(mean_squared_error(y_true, y_pred))
        metrics["mse"] = mse
        metrics["rmse"] = float(np.sqrt(mse))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))

    return metrics


# =============================================================================
# XGBoost device detection
# =============================================================================

def xgboost_build_has_cuda() -> bool:
    try:
        build_info = xgb.build_info()
        value = build_info.get("USE_CUDA", False)
        if isinstance(value, str):
            return value.lower() in {"true", "1", "yes", "on"}
        return bool(value)
    except Exception:
        return False


def probe_xgboost_cuda() -> bool:
    """Run a tiny fit to verify that XGBoost can actually use CUDA."""
    if not xgboost_build_has_cuda():
        return False

    X_probe = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32
    )
    y_probe = np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float32)

    try:
        probe = xgb.XGBRegressor(
            n_estimators=1,
            max_depth=1,
            learning_rate=0.1,
            objective="reg:squarederror",
            tree_method="hist",
            device="cuda",
            n_jobs=1,
            verbosity=0,
        )
        probe.fit(X_probe, y_probe, verbose=False)
        try:
            booster_config = json.loads(probe.get_booster().save_config())
            actual_device = (
                booster_config.get("learner", {})
                .get("generic_param", {})
                .get("device", "cpu")
            )
            return str(actual_device).lower().startswith("cuda")
        except Exception:
            # If the device cannot be verified, do not claim GPU use.
            return False
    except Exception as exc:
        warnings.warn(f"XGBoost CUDA probe failed; CPU will be used. Reason: {exc}")
        return False


def resolve_xgb_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"

    cuda_available = probe_xgboost_cuda()
    if requested == "cuda" and not cuda_available:
        warnings.warn(
            "--xgb_device cuda was requested, but the CUDA probe failed. "
            "Falling back to CPU."
        )
    return "cuda" if cuda_available else "cpu"


# =============================================================================
# Model training and prediction
# =============================================================================

def build_model(
    model_name: str,
    task_type: str,
    parameters: Mapping[str, Any],
    random_seed: int,
    xgb_device: str,
    svm_cache_size: float,
    xgb_n_jobs: int,
) -> Tuple[Any, Optional[StandardScaler], Dict[str, Any]]:
    params = dict(parameters)
    fixed_parameters: Dict[str, Any] = {}

    if model_name == "random_forest":
        fixed_parameters = {
            "random_state": int(random_seed),
            "n_jobs": -1,
            "min_samples_split": 2,
            "bootstrap": True,
        }
        model_class = (
            RandomForestClassifier
            if task_type == "classification"
            else RandomForestRegressor
        )
        model = model_class(**params, **fixed_parameters)
        return model, None, fixed_parameters

    if model_name == "svm":
        if task_type == "classification":
            fixed_parameters = {
                "probability": True,
                "random_state": int(random_seed),
                "cache_size": float(svm_cache_size),
            }
            model = SVC(**params, **fixed_parameters)
        else:
            fixed_parameters = {"cache_size": float(svm_cache_size)}
            model = SVR(**params, **fixed_parameters)
        return model, StandardScaler(), fixed_parameters

    if model_name == "xgboost":
        common_fixed = {
            "random_state": int(random_seed),
            "n_jobs": int(xgb_n_jobs),
            "tree_method": "hist",
            "device": xgb_device,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "verbosity": 0,
        }

        if task_type == "classification":
            fixed_parameters = {
                **common_fixed,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
            }
            model = xgb.XGBClassifier(**params, **fixed_parameters)
        else:
            fixed_parameters = {
                **common_fixed,
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
            }
            model = xgb.XGBRegressor(**params, **fixed_parameters)
        return model, None, fixed_parameters

    raise ValueError(f"Unsupported model: {model_name}")


def fit_model(
    model_name: str,
    model: Any,
    scaler: Optional[StandardScaler],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> Tuple[Any, Optional[StandardScaler]]:
    if scaler is not None:
        X_train_fit = scaler.fit_transform(X_train)
        X_val_fit = scaler.transform(X_val) if X_val is not None else None
    else:
        X_train_fit = X_train
        X_val_fit = X_val

    if model_name == "xgboost" and X_val_fit is not None and y_val is not None:
        model.fit(
            X_train_fit,
            y_train,
            eval_set=[(X_val_fit, y_val)],
            verbose=False,
        )
    else:
        model.fit(X_train_fit, y_train)

    return model, scaler


def predict_with_bundle(
    model: Any,
    scaler: Optional[StandardScaler],
    X_eval: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    X_model = scaler.transform(X_eval) if scaler is not None else X_eval
    predictions = model.predict(X_model)
    return predictions, X_model


def train_and_evaluate_validation(
    model_name: str,
    task_type: str,
    parameters: Mapping[str, Any],
    context: Mapping[str, Any],
    random_seed: int,
    xgb_device: str,
    svm_cache_size: float,
    xgb_n_jobs: int,
) -> Tuple[Dict[str, Optional[float]], Dict[str, Any]]:
    model, scaler, fixed_parameters = build_model(
        model_name=model_name,
        task_type=task_type,
        parameters=parameters,
        random_seed=random_seed,
        xgb_device=xgb_device,
        svm_cache_size=svm_cache_size,
        xgb_n_jobs=xgb_n_jobs,
    )

    if model_name == "xgboost" and task_type == "classification":
        n_classes = int(len(np.unique(context["y_train"])))
        if n_classes > 2:
            model.set_params(
                objective="multi:softprob",
                eval_metric="mlogloss",
                num_class=n_classes,
            )
            fixed_parameters.update(
                {
                    "objective": "multi:softprob",
                    "eval_metric": "mlogloss",
                    "num_class": n_classes,
                }
            )

    model, scaler = fit_model(
        model_name=model_name,
        model=model,
        scaler=scaler,
        X_train=context["X_train"],
        y_train=context["y_train"],
        X_val=context["X_val"],
        y_val=context["y_val"],
    )

    y_pred, X_val_model = predict_with_bundle(model, scaler, context["X_val"])
    metrics = compute_metrics(
        y_true=context["y_val"],
        y_pred=y_pred,
        task_type=task_type,
        model=model,
        X_eval=X_val_model,
    )
    return metrics, fixed_parameters


def refit_best_and_test(
    model_name: str,
    task_type: str,
    parameters: Mapping[str, Any],
    context: Mapping[str, Any],
    random_seed: int,
    xgb_device: str,
    svm_cache_size: float,
    xgb_n_jobs: int,
) -> Tuple[Any, Optional[StandardScaler], Dict[str, Optional[float]], Dict[str, Any]]:
    X_train_val = np.concatenate([context["X_train"], context["X_val"]], axis=0)
    y_train_val = np.concatenate([context["y_train"], context["y_val"]], axis=0)

    model, scaler, fixed_parameters = build_model(
        model_name=model_name,
        task_type=task_type,
        parameters=parameters,
        random_seed=random_seed,
        xgb_device=xgb_device,
        svm_cache_size=svm_cache_size,
        xgb_n_jobs=xgb_n_jobs,
    )

    if model_name == "xgboost" and task_type == "classification":
        n_classes = int(len(np.unique(context["y_train"])))
        if n_classes > 2:
            model.set_params(
                objective="multi:softprob",
                eval_metric="mlogloss",
                num_class=n_classes,
            )
            fixed_parameters.update(
                {
                    "objective": "multi:softprob",
                    "eval_metric": "mlogloss",
                    "num_class": n_classes,
                }
            )

    model, scaler = fit_model(
        model_name=model_name,
        model=model,
        scaler=scaler,
        X_train=X_train_val,
        y_train=y_train_val,
        X_val=None,
        y_val=None,
    )

    y_pred, X_test_model = predict_with_bundle(model, scaler, context["X_test"])
    metrics = compute_metrics(
        y_true=context["y_test"],
        y_pred=y_pred,
        task_type=task_type,
        model=model,
        X_eval=X_test_model,
    )

    return model, scaler, metrics, fixed_parameters


# =============================================================================
# Search/result IDs
# =============================================================================

def make_search_record_id(
    split_method: str,
    dataset_name: str,
    task_type: str,
    feature_type: str,
    model_name: str,
    parameters: Mapping[str, Any],
    random_seed: int,
    similarity_threshold: float,
) -> str:
    payload = {
        "split_method": split_method,
        "similarity_threshold": similarity_threshold
        if split_method == "similarity"
        else None,
        "dataset": dataset_name,
        "task_type": task_type,
        "feature_type": feature_type,
        "model": model_name,
        "parameters": dict(parameters),
        "random_seed": int(random_seed),
    }
    return stable_id("search", payload)


def make_parameter_summary_id(
    split_method: str,
    dataset_name: str,
    task_type: str,
    feature_type: str,
    model_name: str,
    parameters: Mapping[str, Any],
    similarity_threshold: float,
) -> str:
    payload = {
        "split_method": split_method,
        "similarity_threshold": similarity_threshold
        if split_method == "similarity"
        else None,
        "dataset": dataset_name,
        "task_type": task_type,
        "feature_type": feature_type,
        "model": model_name,
        "parameters": dict(parameters),
    }
    return stable_id("param", payload)


def make_best_result_id(
    split_method: str,
    dataset_name: str,
    task_type: str,
    feature_type: str,
    model_name: str,
    similarity_threshold: float,
) -> str:
    payload = {
        "split_method": split_method,
        "similarity_threshold": similarity_threshold
        if split_method == "similarity"
        else None,
        "dataset": dataset_name,
        "task_type": task_type,
        "feature_type": feature_type,
        "model": model_name,
    }
    return stable_id("best", payload)


# =============================================================================
# Hyperparameter search for one dataset/feature/model
# =============================================================================

def train_model_configuration(
    results: Dict[str, Any],
    output_json: Path,
    contexts: Mapping[int, Mapping[str, Any]],
    split_method: str,
    similarity_threshold: float,
    dataset_name: str,
    task_type: str,
    feature_type: str,
    model_name: str,
    seeds: Sequence[int],
    test_size: float,
    val_size: float,
    xgb_device: str,
    svm_cache_size: float,
    xgb_n_jobs: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    parameter_candidates = get_parameter_grid(model_name, task_type)
    parameter_summaries: List[Dict[str, Any]] = []

    print("\n" + "=" * 80)
    print(
        f"Searching: dataset={dataset_name}, task={task_type}, feature={feature_type}, "
        f"model={model_name}, candidates={len(parameter_candidates)}"
    )
    print("=" * 80)

    for candidate_index, parameters in enumerate(parameter_candidates, start=1):
        print(
            f"\n[{candidate_index}/{len(parameter_candidates)}] "
            f"{model_name} parameters: {parameters}"
        )

        seed_record_ids: List[str] = []
        successful_metrics: List[Dict[str, Any]] = []
        fixed_parameters_seen: Optional[Dict[str, Any]] = None

        for seed in seeds:
            record_id = make_search_record_id(
                split_method=split_method,
                dataset_name=dataset_name,
                task_type=task_type,
                feature_type=feature_type,
                model_name=model_name,
                parameters=parameters,
                random_seed=seed,
                similarity_threshold=similarity_threshold,
            )
            seed_record_ids.append(record_id)

            existing = results["search_records"].get(record_id)
            if existing and existing.get("status") == "success":
                print(f"  Seed {seed}: existing successful record found; skipping.")
                successful_metrics.append(existing["validation_metrics"])
                fixed_parameters_seen = existing.get("fixed_parameters")
                continue

            context = contexts[int(seed)]
            started_at = utc_now()
            try:
                validation_metrics, fixed_parameters = train_and_evaluate_validation(
                    model_name=model_name,
                    task_type=task_type,
                    parameters=parameters,
                    context=context,
                    random_seed=int(seed),
                    xgb_device=xgb_device,
                    svm_cache_size=svm_cache_size,
                    xgb_n_jobs=xgb_n_jobs,
                )
                fixed_parameters_seen = fixed_parameters

                record = {
                    "record_id": record_id,
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
                    "random_seed": int(seed),
                    "feature_type": feature_type,
                    "feature_dimension": int(context["feature_dimension"]),
                    "padding_len": context["padding_len"],
                    "model": model_name,
                    "parameters": dict(parameters),
                    "fixed_parameters": fixed_parameters,
                    "split_sizes": context["split_sizes"],
                    "validation_metrics": validation_metrics,
                    "selection_metric": "f1"
                    if task_type == "classification"
                    else "rmse",
                    "error": None,
                }
                results["search_records"][record_id] = record
                successful_metrics.append(validation_metrics)
                print(f"  Seed {seed}: validation metrics = {validation_metrics}")

            except Exception as exc:
                record = {
                    "record_id": record_id,
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
                    "random_seed": int(seed),
                    "feature_type": feature_type,
                    "feature_dimension": int(context["feature_dimension"]),
                    "padding_len": context["padding_len"],
                    "model": model_name,
                    "parameters": dict(parameters),
                    "fixed_parameters": fixed_parameters_seen,
                    "split_sizes": context["split_sizes"],
                    "validation_metrics": None,
                    "selection_metric": "f1"
                    if task_type == "classification"
                    else "rmse",
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                }
                results["search_records"][record_id] = record
                print(f"  Seed {seed}: FAILED - {exc}")

            # Save after every specific parameter/seed combination.
            persist_results(results, output_json)

        parameter_summary_id = make_parameter_summary_id(
            split_method=split_method,
            dataset_name=dataset_name,
            task_type=task_type,
            feature_type=feature_type,
            model_name=model_name,
            parameters=parameters,
            similarity_threshold=similarity_threshold,
        )

        complete = len(successful_metrics) == len(seeds)
        validation_summary = summarize_metrics(successful_metrics)
        parameter_summary = {
            "parameter_summary_id": parameter_summary_id,
            "status": "complete" if complete else "incomplete",
            "dataset": dataset_name,
            "task_type": task_type,
            "split_method": split_method,
            "similarity_threshold": similarity_threshold
            if split_method == "similarity"
            else None,
            "test_size": float(test_size),
            "validation_size": float(val_size),
            "feature_type": feature_type,
            "model": model_name,
            "parameters": dict(parameters),
            "fixed_parameters": fixed_parameters_seen,
            "expected_seeds": [int(seed) for seed in seeds],
            "successful_seed_count": int(len(successful_metrics)),
            "search_record_ids": seed_record_ids,
            "validation_summary": validation_summary,
            "selection_metric": "f1" if task_type == "classification" else "rmse",
            "selection_direction": "maximize"
            if task_type == "classification"
            else "minimize",
            "updated_at": utc_now(),
        }
        results["parameter_summaries"][parameter_summary_id] = parameter_summary
        parameter_summaries.append(parameter_summary)
        persist_results(results, output_json)

    eligible = [
        item
        for item in parameter_summaries
        if item["status"] == "complete"
        and item["selection_metric"] in item.get("validation_summary", {})
    ]
    if not eligible:
        raise RuntimeError(
            f"No complete parameter candidate is available for {dataset_name}, "
            f"{feature_type}, {model_name}."
        )

    if task_type == "classification":
        # Max F1; if tied, prefer lower F1 std.
        best_summary = max(
            eligible,
            key=lambda item: (
                item["validation_summary"]["f1"]["mean"],
                -item["validation_summary"]["f1"]["std"],
            ),
        )
    else:
        # Min RMSE; if tied, prefer lower RMSE std.
        best_summary = min(
            eligible,
            key=lambda item: (
                item["validation_summary"]["rmse"]["mean"],
                item["validation_summary"]["rmse"]["std"],
            ),
        )

    print("\nBest validation parameters:")
    print(json.dumps(to_serializable(best_summary), indent=2, ensure_ascii=False))
    return best_summary, parameter_summaries


# =============================================================================
# Final refit, model saving, and best-metrics saving
# =============================================================================

def save_model_bundle(
    model_path: Path,
    model: Any,
    scaler: Optional[StandardScaler],
    metadata: Mapping[str, Any],
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "scaler": scaler,
        "metadata": to_serializable(dict(metadata)),
    }
    joblib.dump(bundle, model_path, compress=3)


def finalize_best_model(
    results: Dict[str, Any],
    output_json: Path,
    best_model_dir: Path,
    best_metrics_dir: Path,
    contexts: Mapping[int, Mapping[str, Any]],
    best_parameter_summary: Mapping[str, Any],
    split_method: str,
    similarity_threshold: float,
    dataset_name: str,
    task_type: str,
    feature_type: str,
    model_name: str,
    seeds: Sequence[int],
    xgb_device: str,
    svm_cache_size: float,
    xgb_n_jobs: int,
) -> Dict[str, Any]:
    best_result_id = make_best_result_id(
        split_method=split_method,
        dataset_name=dataset_name,
        task_type=task_type,
        feature_type=feature_type,
        model_name=model_name,
        similarity_threshold=similarity_threshold,
    )

    parameters = dict(best_parameter_summary["parameters"])
    seed_test_results: List[Dict[str, Any]] = []

    safe_dataset = sanitize_filename(dataset_name)
    safe_feature = sanitize_filename(feature_type)
    safe_model = sanitize_filename(model_name)
    safe_split = sanitize_filename(split_method)

    combination_model_dir = best_model_dir / safe_split / safe_dataset
    combination_metrics_dir = best_metrics_dir / safe_split / safe_dataset
    combination_metrics_path = (
        combination_metrics_dir / f"{safe_model}__{safe_feature}.json"
    )

    existing_best = results["best_results"].get(best_result_id, {})
    same_best_selection = (
        existing_best.get("best_parameters") == parameters
        and existing_best.get("parameter_summary_id")
        == best_parameter_summary.get("parameter_summary_id")
    )
    existing_seed_results = {
        int(item["random_seed"]): item
        for item in existing_best.get("test_runs", [])
        if same_best_selection and item.get("status") == "success"
    }

    for seed in seeds:
        seed = int(seed)
        model_path = combination_model_dir / (
            f"{safe_model}__{safe_feature}__seed{seed}.joblib"
        )

        existing_seed = existing_seed_results.get(seed)
        if existing_seed and model_path.exists():
            print(
                f"Final model already exists for {dataset_name}/{model_name}/"
                f"{feature_type}/seed={seed}; skipping refit."
            )
            seed_test_results.append(existing_seed)
            continue

        context = contexts[seed]
        started_at = utc_now()
        try:
            model, scaler, test_metrics, fixed_parameters = refit_best_and_test(
                model_name=model_name,
                task_type=task_type,
                parameters=parameters,
                context=context,
                random_seed=seed,
                xgb_device=xgb_device,
                svm_cache_size=svm_cache_size,
                xgb_n_jobs=xgb_n_jobs,
            )

            model_metadata = {
                "dataset": dataset_name,
                "task_type": task_type,
                "split_method": split_method,
                "similarity_threshold": similarity_threshold
                if split_method == "similarity"
                else None,
                "feature_type": feature_type,
                "feature_dimension": context["feature_dimension"],
                "padding_len": context["padding_len"],
                "model": model_name,
                "best_parameters": parameters,
                "fixed_parameters": fixed_parameters,
                "random_seed": seed,
                "training_data": "train+validation",
                "test_metrics": test_metrics,
                "created_at": utc_now(),
            }
            save_model_bundle(
                model_path=model_path,
                model=model,
                scaler=scaler,
                metadata=model_metadata,
            )

            seed_result = {
                "status": "success",
                "random_seed": seed,
                "started_at": started_at,
                "finished_at": utc_now(),
                "model_path": str(model_path),
                "fixed_parameters": fixed_parameters,
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
                    "fixed_parameters": None,
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

        # Persist partial final results after each seed.
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
            "feature_type": feature_type,
            "model": model_name,
            "selection_metric": "f1" if task_type == "classification" else "rmse",
            "selection_direction": "maximize"
            if task_type == "classification"
            else "minimize",
            "best_parameters": parameters,
            "parameter_summary_id": best_parameter_summary["parameter_summary_id"],
            "validation_summary": best_parameter_summary["validation_summary"],
            "refit_strategy": "merge_train_and_validation_then_fit",
            "test_runs": seed_test_results,
            "test_summary": summarize_metrics(successful_test_metrics),
            "best_metrics_path": str(combination_metrics_path),
            "updated_at": utc_now(),
        }
        results["best_results"][best_result_id] = partial_best_result
        persist_results(results, output_json)
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
        "feature_type": feature_type,
        "model": model_name,
        "selection_metric": "f1" if task_type == "classification" else "rmse",
        "selection_direction": "maximize"
        if task_type == "classification"
        else "minimize",
        "best_parameters": parameters,
        "parameter_summary_id": best_parameter_summary["parameter_summary_id"],
        "validation_summary": best_parameter_summary["validation_summary"],
        "refit_strategy": "merge_train_and_validation_then_fit",
        "test_runs": seed_test_results,
        "test_summary": summarize_metrics(successful_test_metrics),
        "best_metrics_path": str(combination_metrics_path),
        "updated_at": utc_now(),
    }

    results["best_results"][best_result_id] = best_result
    persist_results(results, output_json)
    write_metrics_report(combination_metrics_path, best_result)
    return best_result


# =============================================================================
# Main experiment loop
# =============================================================================

def run_experiments(args: argparse.Namespace) -> None:
    output_json = Path(args.output_json).expanduser().resolve()
    best_model_dir = Path(args.best_model_dir).expanduser().resolve()
    best_metrics_dir = Path(args.best_metrics_dir).expanduser().resolve()

    results = load_or_initialize_results(output_json)
    xgb_device = resolve_xgb_device(args.xgb_device)

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
            "features": args.features,
            "datasets": args.datasets,
        },
    )
    results["run_configs"][run_config_key] = {
        "run_config_id": run_config_key,
        "started_at": utc_now(),
        "split_method": args.split_method,
        "similarity_threshold": args.similarity_threshold
        if args.split_method == "similarity"
        else None,
        "test_size": args.test_size,
        "validation_size": args.val_size,
        "seeds": [int(seed) for seed in args.seeds],
        "features": list(args.features),
        "models": list(SUPPORTED_MODELS),
        "requested_datasets": list(args.datasets) if args.datasets else None,
        "output_json": str(output_json),
        "best_model_dir": str(best_model_dir),
        "best_metrics_dir": str(best_metrics_dir),
        "requested_xgb_device": args.xgb_device,
        "effective_xgb_device": xgb_device,
        "svm_cache_size_mb": args.svm_cache_size,
        "xgb_n_jobs": args.xgb_n_jobs,
        "status": "running",
    }
    persist_results(results, output_json)

    print("=" * 80)
    print("Traditional ML Fixed-Parameter Training")
    print("=" * 80)
    print(f"scikit-learn version: {sklearn.__version__}")
    print(f"XGBoost version: {xgb.__version__}")
    print(f"Split method: {args.split_method}")
    print(f"Features: {args.features}")
    print(f"Seeds: {args.seeds}")
    print(f"XGBoost device: {xgb_device}")
    print(f"Global JSON: {output_json}")
    print(f"Best models: {best_model_dir}")
    print(f"Best metrics: {best_metrics_dir}")

    dataset_loader = PepDataset(data_dir=args.data_dir)
    raw_data_path = Path(dataset_loader.data_dir)
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {raw_data_path}")

    csv_files = sorted(raw_data_path.glob("*.csv"))
    if args.datasets:
        requested = {name.lower() for name in args.datasets}
        csv_files = [path for path in csv_files if path.stem.lower() in requested]

    if not csv_files:
        raise FileNotFoundError(
            f"No matching CSV datasets were found in {raw_data_path}."
        )

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
            data = dataset_loader.load_dataset(dataset_name)
            if "peps" not in data.columns or "label" not in data.columns:
                raise KeyError(
                    f"Dataset {dataset_name} must contain 'peps' and 'label' columns."
                )

            print(f"Loaded {dataset_name}: {len(data)} samples")

            for feature_type in args.features:
                print("\n" + "*" * 80)
                print(f"Feature: {feature_type}")
                print("*" * 80)

                contexts = prepare_feature_contexts(
                    data=data,
                    dataset_name=dataset_name,
                    task_type=task_type,
                    feature_type=feature_type,
                    seeds=args.seeds,
                    split_method=args.split_method,
                    test_size=args.test_size,
                    val_size=args.val_size,
                    similarity_threshold=args.similarity_threshold,
                    descriptor_list=None,
                )

                for model_name in SUPPORTED_MODELS:
                    try:
                        best_parameter_summary, _ = train_model_configuration(
                            results=results,
                            output_json=output_json,
                            contexts=contexts,
                            split_method=args.split_method,
                            similarity_threshold=args.similarity_threshold,
                            dataset_name=dataset_name,
                            task_type=task_type,
                            feature_type=feature_type,
                            model_name=model_name,
                            seeds=args.seeds,
                            test_size=args.test_size,
                            val_size=args.val_size,
                            xgb_device=xgb_device,
                            svm_cache_size=args.svm_cache_size,
                            xgb_n_jobs=args.xgb_n_jobs,
                        )

                        finalize_best_model(
                            results=results,
                            output_json=output_json,
                            best_model_dir=best_model_dir,
                            best_metrics_dir=best_metrics_dir,
                            contexts=contexts,
                            best_parameter_summary=best_parameter_summary,
                            split_method=args.split_method,
                            similarity_threshold=args.similarity_threshold,
                            dataset_name=dataset_name,
                            task_type=task_type,
                            feature_type=feature_type,
                            model_name=model_name,
                            seeds=args.seeds,
                            xgb_device=xgb_device,
                            svm_cache_size=args.svm_cache_size,
                            xgb_n_jobs=args.xgb_n_jobs,
                        )
                    except Exception as exc:
                        failure = {
                            "dataset": dataset_name,
                            "task_type": task_type,
                            "feature_type": feature_type,
                            "model": model_name,
                            "split_method": args.split_method,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "traceback": traceback.format_exc(),
                            "time": utc_now(),
                        }
                        dataset_failures.append(failure)
                        print(
                            f"FAILED model combination: {dataset_name}/{feature_type}/"
                            f"{model_name}: {exc}"
                        )
                        results.setdefault("combination_failures", []).append(failure)
                        persist_results(results, output_json)

                # Release feature matrices before moving to the next feature.
                del contexts

        except Exception as exc:
            failure = {
                "dataset": dataset_name,
                "task_type": task_type,
                "feature_type": None,
                "model": None,
                "split_method": args.split_method,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "time": utc_now(),
            }
            dataset_failures.append(failure)
            print(f"FAILED dataset {dataset_name}: {exc}")
            results.setdefault("dataset_failures", []).append(failure)
            persist_results(results, output_json)

    results["run_configs"][run_config_key]["finished_at"] = utc_now()
    results["run_configs"][run_config_key]["status"] = (
        "completed_with_failures" if dataset_failures else "completed"
    )
    results["run_configs"][run_config_key]["failure_count"] = len(dataset_failures)
    persist_results(results, output_json)

    print("\n" + "=" * 80)
    print("Run finished")
    print("=" * 80)
    print(f"Global JSON: {output_json}")
    print(f"Best models: {best_model_dir}")
    print(f"Best metrics: {best_metrics_dir}")
    print(f"Failures in this run: {len(dataset_failures)}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Random Forest, SVM/SVR, and XGBoost with fixed parameters "
            "on peptide classification and regression datasets."
        )
    )
    parser.add_argument(
        "--split_method",
        required=True,
        choices=["random", "similarity"],
        help="Dataset split method. Run the script separately for each method.",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.8,
        help="Similarity threshold used only when --split_method similarity.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test-set proportion (default: 0.2).",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="Validation-set proportion (default: 0.1).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="Random seeds used for splitting and training (default: 42 43 44).",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        choices=list(SUPPORTED_FEATURES),
        default=list(SUPPORTED_FEATURES),
        help="Independent feature types to evaluate.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help=(
            "Optional dataset names without .csv. Omit to process every CSV in "
            "the raw-data directory."
        ),
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Optional data directory passed to PepDataset. Omit to use its default.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="outputs/ml/run_state.json",
        help=(
            "Global JSON file. The random and similarity runs may use the same "
            "path; records are distinguished by split_method."
        ),
    )
    parser.add_argument(
        "--best_model_dir",
        type=str,
        default="outputs/models/ml",
        help="Directory used to save refitted best models.",
    )
    parser.add_argument(
        "--best_metrics_dir",
        type=str,
        default="outputs/metrics/ml",
        help="Directory used to save best validation/test metric JSON files.",
    )
    parser.add_argument(
        "--xgb_device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help=(
            "XGBoost device. auto probes CUDA and otherwise falls back to CPU. "
            "Random Forest and SVM remain CPU estimators in scikit-learn 1.2.2."
        ),
    )
    parser.add_argument(
        "--xgb_n_jobs",
        type=int,
        default=-1,
        help="CPU threads available to XGBoost (default: -1).",
    )
    parser.add_argument(
        "--svm_cache_size",
        type=float,
        default=4096.0,
        help="SVC/SVR kernel cache size in MB (default: 4096).",
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

    # Remove duplicates while preserving user order.
    args.seeds = list(dict.fromkeys(args.seeds))
    args.features = list(dict.fromkeys(args.features))
    if args.datasets:
        args.datasets = list(dict.fromkeys(args.datasets))

    return args


def main() -> None:
    args = parse_args()
    run_experiments(args)


if __name__ == "__main__":
    main()
