"""Utilities for writing consistent experiment metric reports.

Training scripts may keep model-specific checkpoint formats, but every metrics
JSON written for the paper follows the schema produced by
``write_metrics_report``. This keeps downstream analysis independent of the
model implementation.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


METRICS_SCHEMA_VERSION = "1.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    """Serialize common scientific Python scalar and path objects."""
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _normalise_run(run: Mapping[str, Any]) -> dict[str, Any]:
    """Return the common per-seed representation used by all trainers."""
    return {
        "status": run.get("status", "success"),
        "seed": run.get("random_seed", run.get("seed")),
        "split_sizes": run.get("split_sizes"),
        "epochs": run.get(
            "final_training_epochs",
            run.get("final_epochs", run.get("best_epoch")),
        ),
        "metrics": run.get("test_metrics"),
        "model_path": run.get("model_path"),
        "error": run.get("error"),
    }


def _summarise_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize numeric test metrics when a trainer has not done so."""
    values_by_metric: dict[str, list[float]] = {}
    for run in runs:
        for name, value in (run.get("test_metrics") or {}).items():
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                values_by_metric.setdefault(name, []).append(numeric)

    return {
        name: {
            "mean": sum(values) / len(values),
            "std": (
                sum((value - sum(values) / len(values)) ** 2 for value in values)
                / len(values)
            )
            ** 0.5,
            "n": len(values),
        }
        for name, values in sorted(values_by_metric.items())
    }


def build_metrics_report(result: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a model-specific result dictionary to the shared JSON schema."""
    parameters = result.get("best_parameters")
    if parameters is None:
        parameters = {
            "head": result.get("head_parameters"),
            "training": result.get("training_parameters"),
            "encoder": result.get("esm_parameters"),
        }
        parameters = {key: value for key, value in parameters.items() if value is not None}

    raw_runs: Sequence[Mapping[str, Any]] = result.get("test_runs") or [result]
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": result.get("status", "complete"),
        "dataset": result.get("dataset"),
        "task_type": result.get("task_type", result.get("task")),
        "split": {
            "method": result.get("split_method"),
            "similarity_threshold": result.get("similarity_threshold"),
        },
        "model": {
            "name": result.get("model"),
            "feature_type": result.get("feature_type"),
        },
        "parameters": parameters,
        "selection": {
            "metric": result.get("selection_metric"),
            "direction": result.get("selection_direction"),
            "validation_summary": result.get("validation_summary"),
        },
        "test": {
            "runs": [_normalise_run(run) for run in raw_runs],
            "summary": result.get("test_summary") or _summarise_runs(raw_runs),
        },
    }


def write_metrics_report(path: Path | str, result: Mapping[str, Any]) -> None:
    """Atomically write a UTF-8 metrics report using the shared schema."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=str(output_path.parent),
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                build_metrics_report(result),
                handle,
                indent=2,
                ensure_ascii=False,
                default=_json_default,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

