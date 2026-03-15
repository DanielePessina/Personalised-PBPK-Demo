"""Persistence helpers for experiment result bundles."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
import pickle

import equinox as eqx
import pandas as pd
import yaml

from .schemas import DatasetSummary, FoldArtifact, RunManifest

REQUIRED_PREDICTION_COLUMNS = [
    "experiment_name",
    "analysis",
    "run_id",
    "outer_fold",
    "inner_split",
    "split",
    "patient_id",
    "measurement_index",
    "time",
    "observed",
    "predicted",
    "dose_rate",
    "dose_duration",
]

ROOT_BUNDLE_FILES = [
    "config.yaml",
    "dataset_summary.yaml",
    "folds.yaml",
    "run_manifest.yaml",
]


def write_yaml(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write a mapping to a block-style YAML file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=False, default_flow_style=False)
    return output_path


class BundleWriter:
    """Persist result bundles under a stable per-analysis directory."""

    def __init__(self, *, output_root: str | Path, experiment_name: str, analysis: str, run_id: str):
        self.output_root = Path(output_root)
        self.bundle_dir = self.output_root / experiment_name / analysis / run_id
        self.bundle_dir.mkdir(parents=True, exist_ok=True)

    def write_config(self, payload: Mapping[str, Any]) -> Path:
        """Write resolved configuration metadata."""
        return write_yaml(self.bundle_dir / "config.yaml", payload)

    def write_dataset_summary(self, summary: DatasetSummary) -> Path:
        """Write persisted dataset summary metadata."""
        return write_yaml(self.bundle_dir / "dataset_summary.yaml", summary.to_dict())

    def write_folds(self, fold_artifact: FoldArtifact) -> Path:
        """Write the fold artifact copied into the analysis bundle."""
        return write_yaml(self.bundle_dir / "folds.yaml", fold_artifact.to_dict())

    def write_manifest(self, manifest: RunManifest) -> Path:
        """Write the run manifest."""
        return write_yaml(self.bundle_dir / "run_manifest.yaml", manifest.to_dict())

    def write_predictions(self, fold_index: int, split_name: str, frame: pd.DataFrame) -> Path:
        """Write long-form predictions for a split."""
        missing = [column for column in REQUIRED_PREDICTION_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"Prediction frame missing required columns: {missing}")
        fold_dir = self._fold_dir(fold_index)
        path = fold_dir / f"{split_name}_predictions.parquet"
        frame.to_parquet(path, index=False)
        return path

    def write_metrics(self, fold_index: int, split_name: str, frame: pd.DataFrame) -> Path:
        """Write metrics for a split."""
        fold_dir = self._fold_dir(fold_index)
        path = fold_dir / f"{split_name}_metrics.parquet"
        frame.to_parquet(path, index=False)
        return path

    def write_history(self, fold_index: int, history: Iterable[Mapping[str, Any]]) -> Path:
        """Write training history for a fold."""
        fold_dir = self._fold_dir(fold_index)
        path = fold_dir / "training_history.parquet"
        frame = pd.DataFrame(list(history))
        frame.to_parquet(path, index=False)
        return path

    def write_metadata(self, fold_index: int, name: str, payload: Mapping[str, Any]) -> Path:
        """Write analysis-specific YAML metadata for a fold."""
        fold_dir = self._fold_dir(fold_index)
        return write_yaml(fold_dir / f"{name}.yaml", payload)

    def write_checkpoint(
        self,
        fold_index: int,
        name: str,
        state: Any,
        *,
        serializer: Callable[[Any, Path], None] | None = None,
    ) -> Path:
        """Write a fold checkpoint using a supplied serializer or pickle fallback."""
        fold_dir = self._fold_dir(fold_index)
        path = fold_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if serializer is not None:
            serializer(state, path)
            return path
        with path.open("wb") as handle:
            pickle.dump(state, handle)
        return path

    def write_table(self, fold_index: int, name: str, frame: pd.DataFrame) -> Path:
        """Write an analysis-specific parquet table for a fold."""
        fold_dir = self._fold_dir(fold_index)
        path = fold_dir / name
        frame.to_parquet(path, index=False)
        return path

    def _fold_dir(self, fold_index: int) -> Path:
        fold_dir = self.bundle_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        return fold_dir

    def save_fold_metadata(self, fold_index: int, payload: Mapping[str, Any]) -> Path:
        """Compatibility wrapper for fold metadata."""
        return self.write_metadata(fold_index, "metadata", payload)

    def save_fold_status(self, fold_index: int, payload: Mapping[str, Any]) -> Path:
        """Persist fold status metadata."""
        return self.write_metadata(fold_index, "status", payload)

    def save_fold_history(self, fold_index: int, history: Iterable[Mapping[str, Any]]) -> Path:
        """Persist fold training history."""
        return self.write_history(fold_index, history)

    def save_fold_predictions(self, fold_index: int, split_name: str, frame: pd.DataFrame) -> Path:
        """Compatibility wrapper for fold predictions."""
        return self.write_predictions(fold_index, split_name, frame)

    def save_fold_metrics(self, fold_index: int, frame: pd.DataFrame) -> Path:
        """Persist aggregated fold metrics."""
        path = self._fold_dir(fold_index) / "metrics.parquet"
        frame.to_parquet(path, index=False)
        return path

    def save_checkpoint_bytes(self, fold_index: int, name: str, payload: bytes) -> Path:
        """Persist raw checkpoint bytes."""
        path = self._fold_dir(fold_index) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def save_eqx_checkpoint(self, fold_index: int, name: str, model: Any) -> Path:
        """Persist an Equinox checkpoint."""
        buffer = BytesIO()
        eqx.tree_serialise_leaves(buffer, model)
        return self.save_checkpoint_bytes(fold_index, name, buffer.getvalue())
