"""Typed schemas for experiment orchestration and persisted artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class DatasetSummary:
    """Persisted dataset summary for the remifentanil cohort."""

    dataset_path: str
    dataset_fingerprint: str
    patient_count: int
    average_measurement_points: float
    minimum_measurement_points: int
    maximum_measurement_points: int
    average_age: float
    average_weight: float
    average_height: float
    average_bsa: float
    average_dose_rate: float
    average_dose_duration: float
    fold_sizes: list[dict[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-serializable representation."""
        return asdict(self)


@dataclass(slots=True)
class FoldCounts:
    """Patient counts for a single outer fold."""

    train: int
    test: int

    def to_dict(self) -> dict[str, int]:
        """Return a YAML-serializable representation."""
        return asdict(self)


@dataclass(slots=True)
class FoldSpec:
    """Patient-level split specification for a single outer fold."""

    fold_index: int
    train_patient_ids: list[int]
    test_patient_ids: list[int]
    counts: FoldCounts

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-serializable representation."""
        payload = asdict(self)
        payload["counts"] = self.counts.to_dict()
        return payload


@dataclass(slots=True)
class FoldArtifact:
    """Serializable fold artifact shared across analyses."""

    experiment_name: str
    dataset_path: str
    dataset_fingerprint: str
    outer_fold_count: int
    outer_seed: int
    generated_at: str
    folds: list[FoldSpec]

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-serializable representation."""
        return {
            "experiment_name": self.experiment_name,
            "dataset_path": self.dataset_path,
            "dataset_fingerprint": self.dataset_fingerprint,
            "outer_fold_count": self.outer_fold_count,
            "outer_seed": self.outer_seed,
            "generated_at": self.generated_at,
            "folds": [fold.to_dict() for fold in self.folds],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FoldArtifact":
        """Build a fold artifact from a deserialized mapping."""
        return cls(
            experiment_name=payload["experiment_name"],
            dataset_path=payload["dataset_path"],
            dataset_fingerprint=payload["dataset_fingerprint"],
            outer_fold_count=int(payload["outer_fold_count"]),
            outer_seed=int(payload["outer_seed"]),
            generated_at=payload["generated_at"],
            folds=[
                FoldSpec(
                    fold_index=int(fold["fold_index"]),
                    train_patient_ids=[int(value) for value in fold["train_patient_ids"]],
                    test_patient_ids=[int(value) for value in fold["test_patient_ids"]],
                    counts=FoldCounts(
                        train=int(fold["counts"]["train"]),
                        test=int(fold["counts"]["test"]),
                    ),
                )
                for fold in payload["folds"]
            ],
        )


@dataclass(slots=True)
class ManifestStatus:
    """Run manifest status fields persisted for each analysis bundle."""

    status: str
    started_at: str
    completed_at: str | None
    wall_clock_seconds: float | None
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-serializable representation."""
        return asdict(self)


@dataclass(slots=True)
class RunManifest:
    """Top-level analysis run manifest."""

    experiment_name: str
    analysis: str
    run_id: str
    config_version: int
    dataset_path: str
    dataset_fingerprint: str
    fold_file: str
    code_version: str
    status: ManifestStatus

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-serializable representation."""
        payload = asdict(self)
        payload["status"] = self.status.to_dict()
        return payload


@dataclass(slots=True)
class SplitConfig:
    """Split policy loaded from YAML."""

    outer_folds: int
    outer_seed: int


@dataclass(slots=True)
class ExperimentConfig:
    """Experiment-wide configuration loaded from YAML."""

    version: int
    experiment_name: str
    dataset_path: str
    output_root: str
    analyses: list[str]
    split: SplitConfig
    model_configs: dict[str, dict[str, Any]]
