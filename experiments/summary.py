"""Dataset loading and summary utilities for experiment orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import numpy as np
from rich.table import Table

from pharmacokinetics import remifentanil

from .schemas import DatasetSummary, FoldArtifact


@dataclass(slots=True)
class RemifentanilDataset:
    """In-memory representation of the canonical remifentanil cohort."""

    dataset_path: Path
    dataset_fingerprint: str
    raw_patients: list[remifentanil.RawPatient]
    physio_patients: list[remifentanil.PhysiologicalParameters]
    raw_by_id: dict[int, remifentanil.RawPatient]
    physio_by_id: dict[int, remifentanil.PhysiologicalParameters]


def fingerprint_file(path: Path) -> str:
    """Return a SHA256 fingerprint for a dataset file."""
    hasher = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_remifentanil_dataset(dataset_path: str | Path) -> RemifentanilDataset:
    """Load raw and physiological patient data for orchestration."""
    path = Path(dataset_path).resolve()
    raw_patients = remifentanil.import_patients(str(path))
    physio_patients = [remifentanil.create_physiological_parameters(patient) for patient in raw_patients]
    raw_by_id = {int(patient.id): patient for patient in raw_patients}
    physio_by_id = {int(patient.id): patient for patient in physio_patients}
    return RemifentanilDataset(
        dataset_path=path,
        dataset_fingerprint=fingerprint_file(path),
        raw_patients=raw_patients,
        physio_patients=physio_patients,
        raw_by_id=raw_by_id,
        physio_by_id=physio_by_id,
    )


def subset_patients_by_id[T](patient_mapping: dict[int, T], patient_ids: Iterable[int]) -> list[T]:
    """Return patients in the same order as the supplied IDs."""
    return [patient_mapping[int(patient_id)] for patient_id in patient_ids]


def compute_dataset_summary(
    dataset: RemifentanilDataset,
    fold_artifact: FoldArtifact | None = None,
) -> DatasetSummary:
    """Compute persisted summary statistics for the remifentanil cohort."""
    raw_patients = dataset.raw_patients
    measurement_counts = np.array([int(np.asarray(patient.mask).sum()) for patient in raw_patients], dtype=np.int64)
    ages = np.array([float(patient.age) for patient in raw_patients], dtype=np.float64)
    weights = np.array([float(patient.weight) for patient in raw_patients], dtype=np.float64)
    heights = np.array([float(patient.height) for patient in raw_patients], dtype=np.float64)
    bsas = np.array([float(patient.bsa) for patient in raw_patients], dtype=np.float64)
    dose_rates = np.array([float(patient.dose_rate) for patient in raw_patients], dtype=np.float64)
    dose_durations = np.array([float(patient.dose_duration) for patient in raw_patients], dtype=np.float64)

    fold_sizes: list[dict[str, int]] = []
    if fold_artifact is not None:
        fold_sizes = [
            {
                "fold_index": fold.fold_index,
                "train": fold.counts.train,
                "test": fold.counts.test,
            }
            for fold in fold_artifact.folds
        ]

    return DatasetSummary(
        dataset_path=str(dataset.dataset_path),
        dataset_fingerprint=dataset.dataset_fingerprint,
        patient_count=len(raw_patients),
        average_measurement_points=float(measurement_counts.mean()),
        minimum_measurement_points=int(measurement_counts.min()),
        maximum_measurement_points=int(measurement_counts.max()),
        average_age=float(ages.mean()),
        average_weight=float(weights.mean()),
        average_height=float(heights.mean()),
        average_bsa=float(bsas.mean()),
        average_dose_rate=float(dose_rates.mean()),
        average_dose_duration=float(dose_durations.mean()),
        fold_sizes=fold_sizes,
    )


def dataset_summary_table(summary: DatasetSummary) -> Table:
    """Render a stable dataset summary table for the CLI."""
    table = Table(title="Dataset Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Patients", str(summary.patient_count))
    table.add_row("Avg measurement points", f"{summary.average_measurement_points:.2f}")
    table.add_row("Measurement points range", f"{summary.minimum_measurement_points} - {summary.maximum_measurement_points}")
    table.add_row("Avg age", f"{summary.average_age:.2f}")
    table.add_row("Avg weight", f"{summary.average_weight:.2f}")
    table.add_row("Avg height", f"{summary.average_height:.2f}")
    table.add_row("Avg BSA", f"{summary.average_bsa:.3f}")
    table.add_row("Avg dose rate", f"{summary.average_dose_rate:.3f}")
    table.add_row("Avg dose duration", f"{summary.average_dose_duration:.3f}")
    if summary.fold_sizes:
        for fold in summary.fold_sizes:
            table.add_row(
                f"Fold {fold['fold_index']} sizes",
                f"Train={fold['train']} Test={fold['test']}",
            )
    return table
