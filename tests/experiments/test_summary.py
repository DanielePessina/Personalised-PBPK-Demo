"""Regression tests for dataset summary stability."""

from __future__ import annotations

from experiments.folds import create_fold_artifact
from experiments.summary import compute_dataset_summary


def test_dataset_summary_contains_required_fields(experiment_dataset):
    """The summary payload should expose the fields downstream code expects."""
    folds = create_fold_artifact(
        experiment_name="exp",
        dataset=experiment_dataset,
        outer_fold_count=5,
        outer_seed=123,
    )
    summary = compute_dataset_summary(experiment_dataset, folds)
    payload = summary.to_dict()
    required_keys = {
        "dataset_path",
        "dataset_fingerprint",
        "patient_count",
        "average_measurement_points",
        "minimum_measurement_points",
        "maximum_measurement_points",
        "average_age",
        "average_weight",
        "average_height",
        "average_bsa",
        "average_dose_rate",
        "average_dose_duration",
        "fold_sizes",
    }
    assert required_keys <= payload.keys()
    assert payload["patient_count"] > 0
    assert payload["average_measurement_points"] > 0.0
    assert len(payload["fold_sizes"]) == 5
    assert {"fold_index", "train", "test"} <= payload["fold_sizes"][0].keys()
    assert "validation" not in payload["fold_sizes"][0]
