"""Regression tests for result bundle persistence and validation."""

from __future__ import annotations

from datetime import datetime, timezone

from experiments.folds import create_fold_artifact
from experiments.results import BundleWriter
from experiments.schemas import ManifestStatus, RunManifest
from experiments.summary import compute_dataset_summary
from experiments.validate import validate_bundle, validate_prediction_frame


def test_prediction_frame_validation_flags_missing_columns(prediction_frame_template):
    """Missing required prediction columns should be reported explicitly."""
    broken = prediction_frame_template.drop(columns=["predicted"])
    errors = validate_prediction_frame(broken)
    assert errors
    assert "predicted" in errors[0]


def test_bundle_validation_accepts_complete_bundle(tmp_path, experiment_dataset, prediction_frame_template):
    """A complete bundle with valid predictions should pass validation."""
    folds = create_fold_artifact(
        experiment_name="exp",
        dataset=experiment_dataset,
        outer_fold_count=5,
        outer_seed=123,
    )
    summary = compute_dataset_summary(experiment_dataset, folds)
    writer = BundleWriter(output_root=tmp_path, experiment_name="exp", analysis="stub", run_id="run-1")
    manifest = RunManifest(
        experiment_name="exp",
        analysis="stub",
        run_id="run-1",
        config_version=1,
        dataset_path=str(experiment_dataset.dataset_path),
        dataset_fingerprint=experiment_dataset.dataset_fingerprint,
        fold_file=str(tmp_path / "folds.yaml"),
        code_version="test",
        status=ManifestStatus(
            status="completed",
            started_at=datetime.now(timezone.utc).isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),
            wall_clock_seconds=0.1,
        ),
    )
    writer.write_config({"version": 1})
    writer.write_dataset_summary(summary)
    writer.write_folds(folds)
    writer.write_manifest(manifest)
    writer.write_predictions(0, "train", prediction_frame_template)
    writer.write_predictions(0, "test", prediction_frame_template.assign(split="test", inner_split="test"))
    writer.write_metrics(0, "train", prediction_frame_template[["split"]].assign(mse=0.0, mae=0.0, rmse=0.0, r2=1.0, n_points=1))
    writer.write_metrics(
        0,
        "test",
        prediction_frame_template[["split"]].assign(split="test", mse=0.0, mae=0.0, rmse=0.0, r2=1.0, n_points=1),
    )
    writer.write_history(0, [{"epoch": 0, "loss": 0.0}])
    writer.save_fold_metadata(0, {"analysis": "stub"})
    writer.save_fold_status(0, {"status": "completed", "failure_reason": None})
    writer.save_fold_metrics(
        0,
        prediction_frame_template[["split"]].assign(mse=0.0, mae=0.0, rmse=0.0, r2=1.0, n_points=1),
    )
    assert validate_bundle(writer.bundle_dir) == []
    fold_dir = writer.bundle_dir / "fold_0"
    assert not (fold_dir / "validation_predictions.parquet").exists()
    assert not (fold_dir / "validation_metrics.parquet").exists()
