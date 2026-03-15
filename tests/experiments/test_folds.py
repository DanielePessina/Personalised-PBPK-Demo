"""Regression tests for deterministic fold generation."""

from __future__ import annotations

from collections import Counter

from experiments.folds import create_fold_artifact, validate_fold_artifact


def test_fold_artifact_is_deterministic(experiment_dataset):
    """The same seeds should produce identical fold assignments."""
    first = create_fold_artifact(
        experiment_name="exp",
        dataset=experiment_dataset,
        outer_fold_count=5,
        outer_seed=123,
    )
    second = create_fold_artifact(
        experiment_name="exp",
        dataset=experiment_dataset,
        outer_fold_count=5,
        outer_seed=123,
    )
    assert first.to_dict()["folds"] == second.to_dict()["folds"]


def test_fold_artifact_has_patient_disjoint_partitions(experiment_dataset):
    """Train and test patient sets must be disjoint and exhaustive across folds."""
    artifact = create_fold_artifact(
        experiment_name="exp",
        dataset=experiment_dataset,
        outer_fold_count=5,
        outer_seed=123,
    )
    assert validate_fold_artifact(artifact, experiment_dataset) == []
    test_id_counts: Counter[int] = Counter()
    for fold in artifact.folds:
        train_ids = set(fold.train_patient_ids)
        test_ids = set(fold.test_patient_ids)
        assert not (train_ids & test_ids)
        test_id_counts.update(test_ids)
        assert fold.counts.train == len(train_ids)
        assert fold.counts.test == len(test_ids)
    assert set(test_id_counts) == set(experiment_dataset.raw_by_id)
    assert set(test_id_counts.values()) == {1}
