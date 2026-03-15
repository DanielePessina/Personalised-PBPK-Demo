"""Validation helpers for result bundles and prediction schema stability."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .results import REQUIRED_PREDICTION_COLUMNS, ROOT_BUNDLE_FILES

REQUIRED_FOLD_FILES = [
    "metadata.yaml",
    "status.yaml",
    "training_history.parquet",
    "metrics.parquet",
    "train_metrics.parquet",
    "test_metrics.parquet",
    "train_predictions.parquet",
    "test_predictions.parquet",
]


def validate_prediction_frame(frame: pd.DataFrame) -> list[str]:
    """Validate the required prediction schema."""
    errors: list[str] = []
    missing = [column for column in REQUIRED_PREDICTION_COLUMNS if column not in frame.columns]
    if missing:
        errors.append(f"Prediction frame missing required columns: {missing}")
    if frame.empty:
        errors.append("Prediction frame is empty.")
    return errors


def validate_bundle(bundle_dir: str | Path) -> list[str]:
    """Validate the required files and prediction payloads in a bundle directory."""
    path = Path(bundle_dir)
    errors: list[str] = []
    for filename in ROOT_BUNDLE_FILES:
        if not (path / filename).exists():
            errors.append(f"Missing required bundle file: {filename}")

    fold_dirs = sorted(candidate for candidate in path.glob("fold_*") if candidate.is_dir())
    if not fold_dirs:
        errors.append("Bundle does not contain any fold directories.")
    for fold_dir in fold_dirs:
        for filename in REQUIRED_FOLD_FILES:
            if not (fold_dir / filename).exists():
                errors.append(f"{fold_dir.name}: missing required fold file {filename}")

    for prediction_path in sorted(path.glob("fold_*/*_predictions.parquet")):
        frame = pd.read_parquet(prediction_path)
        errors.extend(f"{prediction_path.name}: {error}" for error in validate_prediction_frame(frame))

    for metrics_path in sorted(path.glob("fold_*/metrics.parquet")):
        try:
            pd.read_parquet(metrics_path)
        except Exception as exc:
            errors.append(f"{metrics_path.name}: unable to read metrics parquet ({exc})")
    return errors


def find_analysis_bundles(analysis_root: str | Path) -> list[Path]:
    """Return all run bundles under an analysis directory."""
    root = Path(analysis_root)
    if not root.exists():
        return []
    return [path for path in sorted(root.iterdir()) if path.is_dir()]
