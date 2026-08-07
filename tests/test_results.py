"""Tests for the shared experiment metrics schema."""

import json

from pephub.results import METRICS_SCHEMA_VERSION, write_metrics_report


def test_write_metrics_report_uses_shared_schema(tmp_path):
    output_path = tmp_path / "metrics.json"
    result = {
        "status": "complete",
        "dataset": "AMP",
        "task_type": "classification",
        "split_method": "random",
        "model": "example",
        "feature_type": "descriptor",
        "best_parameters": {"alpha": 1.0},
        "test_runs": [
            {
                "status": "success",
                "random_seed": 42,
                "test_metrics": {"accuracy": 0.8, "f1": 0.75},
            }
        ],
    }

    write_metrics_report(output_path, result)
    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["schema_version"] == METRICS_SCHEMA_VERSION
    assert report["model"] == {"name": "example", "feature_type": "descriptor"}
    assert report["test"]["runs"][0]["seed"] == 42
    assert report["test"]["summary"]["f1"]["mean"] == 0.75
