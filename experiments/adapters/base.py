"""Shared adapter contracts and helper utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from pharmacokinetics import remifentanil

from ..schemas import FoldSpec
from ..summary import RemifentanilDataset, subset_patients_by_id


@dataclass(slots=True)
class PatientSplit:
    """Raw and physiological patient views for a single split."""

    patient_ids: list[int]
    raw_patients: list[remifentanil.RawPatient]
    physio_patients: list[remifentanil.PhysiologicalParameters]


@dataclass(slots=True)
class PreparedFoldData:
    """Prepared split inputs shared by multiple adapters."""

    fold_index: int
    train: PatientSplit
    test: PatientSplit


def build_fold_data(dataset: RemifentanilDataset, fold_spec: FoldSpec) -> PreparedFoldData:
    """Create split-specific patient views for one outer fold."""
    return PreparedFoldData(
        fold_index=fold_spec.fold_index,
        train=PatientSplit(
            patient_ids=fold_spec.train_patient_ids,
            raw_patients=subset_patients_by_id(dataset.raw_by_id, fold_spec.train_patient_ids),
            physio_patients=subset_patients_by_id(dataset.physio_by_id, fold_spec.train_patient_ids),
        ),
        test=PatientSplit(
            patient_ids=fold_spec.test_patient_ids,
            raw_patients=subset_patients_by_id(dataset.raw_by_id, fold_spec.test_patient_ids),
            physio_patients=subset_patients_by_id(dataset.physio_by_id, fold_spec.test_patient_ids),
        ),
    )


def build_prediction_frame(
    *,
    experiment_name: str,
    analysis: str,
    run_id: str,
    fold_index: int,
    split_name: str,
    patients: Iterable[remifentanil.RawPatient],
    predictions_by_patient: Mapping[int, np.ndarray],
    covariate_columns: Iterable[str],
) -> pd.DataFrame:
    """Create the standard long-form prediction table."""
    rows: list[dict[str, Any]] = []
    covariate_columns = list(covariate_columns)
    for patient in patients:
        valid_mask = np.asarray(patient.mask, dtype=bool)
        times = np.asarray(patient.t_meas)[valid_mask]
        observed = np.asarray(patient.c_meas)[valid_mask]
        predicted = np.asarray(predictions_by_patient[int(patient.id)])
        for measurement_index, (time_value, observed_value, predicted_value) in enumerate(
            zip(times, observed, predicted, strict=True)
        ):
            row = {
                "experiment_name": experiment_name,
                "analysis": analysis,
                "run_id": run_id,
                "outer_fold": fold_index,
                "inner_split": split_name,
                "split": split_name,
                "patient_id": int(patient.id),
                "measurement_index": measurement_index,
                "time": float(time_value),
                "observed": float(observed_value),
                "predicted": float(predicted_value),
                "dose_rate": float(patient.dose_rate),
                "dose_duration": float(patient.dose_duration),
            }
            for column in covariate_columns:
                row[column] = float(getattr(patient, column))
            rows.append(row)
    return pd.DataFrame(rows)


def metrics_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute stable regression metrics for one prediction frame."""
    residuals = predictions["observed"] - predictions["predicted"]
    mse = float(np.mean(np.square(residuals)))
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(mse))
    observed = predictions["observed"].to_numpy(dtype=float)
    predicted = predictions["predicted"].to_numpy(dtype=float)
    centered = observed - observed.mean()
    denom = float(np.sum(centered**2))
    r2 = float(1.0 - np.sum((observed - predicted) ** 2) / denom) if denom > 0.0 else float("nan")
    return pd.DataFrame(
        [
            {
                "split": str(predictions["split"].iloc[0]),
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "n_points": int(len(predictions)),
            }
        ]
    )


class AnalysisAdapter(ABC):
    """Shared contract implemented by each analysis adapter."""

    analysis_name: str

    @abstractmethod
    def prepare_inputs(self, dataset: RemifentanilDataset, fold_spec: FoldSpec, config: dict[str, Any]) -> Any:
        """Prepare model inputs for a single fold."""

    @abstractmethod
    def fit(self, model_inputs: Any, config: dict[str, Any], reporter: Callable[[str], None]) -> Any:
        """Fit the model for a fold and return a trained-state object."""

    @abstractmethod
    def predict(self, trained_state: Any, split_name: str) -> pd.DataFrame:
        """Return long-form predictions for one split."""

    @abstractmethod
    def evaluate(self, long_predictions: pd.DataFrame) -> pd.DataFrame:
        """Return metrics for one prediction frame."""

    @abstractmethod
    def save_artifacts(self, trained_state: Any, bundle_writer: Any) -> None:
        """Persist model-specific artifacts for one trained fold."""
