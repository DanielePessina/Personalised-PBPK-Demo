"""Deterministic patient-level fold generation for experiment orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from .schemas import FoldArtifact, FoldCounts, FoldSpec
from .summary import RemifentanilDataset


def create_fold_artifact(
    *,
    experiment_name: str,
    dataset: RemifentanilDataset,
    outer_fold_count: int,
    outer_seed: int,
) -> FoldArtifact:
    """Create deterministic outer train/test folds by patient ID."""
    if outer_fold_count < 2:
        raise ValueError("At least two outer folds are required.")

    patient_ids = np.array(sorted(dataset.raw_by_id), dtype=np.int64)
    outer_rng = np.random.default_rng(outer_seed)
    shuffled = patient_ids.copy()
    outer_rng.shuffle(shuffled)
    outer_chunks = np.array_split(shuffled, outer_fold_count)

    fold_specs: list[FoldSpec] = []
    for fold_index, test_chunk in enumerate(outer_chunks):
        train_pool = np.concatenate([chunk for idx, chunk in enumerate(outer_chunks) if idx != fold_index])
        train_ids = np.sort(train_pool).astype(np.int64)
        test_ids = np.sort(test_chunk).astype(np.int64)

        fold_specs.append(
            FoldSpec(
                fold_index=fold_index,
                train_patient_ids=[int(value) for value in train_ids.tolist()],
                test_patient_ids=[int(value) for value in test_ids.tolist()],
                counts=FoldCounts(
                    train=len(train_ids),
                    test=len(test_ids),
                ),
            )
        )

    return FoldArtifact(
        experiment_name=experiment_name,
        dataset_path=str(dataset.dataset_path),
        dataset_fingerprint=dataset.dataset_fingerprint,
        outer_fold_count=outer_fold_count,
        outer_seed=outer_seed,
        generated_at=datetime.now(timezone.utc).isoformat(),
        folds=fold_specs,
    )


def save_fold_artifact(fold_artifact: FoldArtifact, output_path: str | Path) -> Path:
    """Persist a fold artifact to YAML."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(fold_artifact.to_dict(), handle, sort_keys=False, default_flow_style=False)
    return path


def load_fold_artifact(path: str | Path) -> FoldArtifact:
    """Load a YAML fold artifact from disk."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in fold artifact {path}.")
    return FoldArtifact.from_dict(payload)


def validate_fold_artifact(fold_artifact: FoldArtifact, dataset: RemifentanilDataset) -> list[str]:
    """Validate fold disjointness and dataset fingerprint alignment."""
    errors: list[str] = []
    known_ids = set(dataset.raw_by_id)
    if fold_artifact.dataset_fingerprint != dataset.dataset_fingerprint:
        errors.append("Dataset fingerprint does not match the loaded dataset.")

    test_id_counts: dict[int, int] = {}
    for fold in fold_artifact.folds:
        train_ids = set(fold.train_patient_ids)
        test_ids = set(fold.test_patient_ids)
        if train_ids & test_ids:
            errors.append(f"Fold {fold.fold_index} has train/test overlap.")
        if not (train_ids | test_ids) <= known_ids:
            errors.append(f"Fold {fold.fold_index} contains unknown patient IDs.")
        for patient_id in test_ids:
            test_id_counts[patient_id] = test_id_counts.get(patient_id, 0) + 1

    if set(test_id_counts) != known_ids:
        errors.append("Outer test folds do not cover the complete patient set exactly once.")
    duplicate_test_ids = sorted(patient_id for patient_id, count in test_id_counts.items() if count != 1)
    if duplicate_test_ids:
        errors.append(f"Outer test folds assign some patients to test more than once: {duplicate_test_ids}")
    return errors
