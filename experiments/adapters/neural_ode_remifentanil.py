"""Remifentanil Neural ODE adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import equinox as eqx
import numpy as np
import pandas as pd

from pharmacokinetics import remifentanil, remifentanil_node

from .base import AnalysisAdapter, PreparedFoldData, build_fold_data, build_prediction_frame, metrics_frame

NODE_COVARIATES = ("age", "weight", "height", "sex", "dose_rate", "dose_duration")


@dataclass(slots=True)
class RemiNODEPreparedInputs:
    """Prepared fold inputs for the remifentanil NODE adapter."""

    fold_data: PreparedFoldData


@dataclass(slots=True)
class RemiNODETrainedState:
    """Trained remifentanil NODE fold state."""

    fold_index: int
    model: remifentanil_node.RemiNODE
    scalers: dict[str, Any]
    predictions: dict[str, pd.DataFrame]
    history: list[dict[str, float]]
    metadata: dict[str, Any]


def _predict_split(
    *,
    model: remifentanil_node.RemiNODE,
    scalers: dict[str, Any],
    split_name: str,
    fold_index: int,
    patients: list[remifentanil.RawPatient],
    experiment_name: str,
    run_id: str,
) -> pd.DataFrame:
    processed, patient_ids, _ = remifentanil_node.prepare_dataset(patients, scalers=scalers)
    predictions_by_patient: dict[int, np.ndarray] = {}
    for patient_id, node_data in zip(patient_ids.tolist(), processed, strict=True):
        valid_mask = np.asarray(node_data.measurement_mask, dtype=bool)
        scaled_times = node_data.t_meas_scaled[valid_mask]
        predicted_scaled = model(scaled_times, node_data.y0, node_data.static_augmentations)
        predicted = remifentanil_node.unscale_predictions(predicted_scaled, scalers)
        predictions_by_patient[int(patient_id)] = np.asarray(predicted)
    return build_prediction_frame(
        experiment_name=experiment_name,
        analysis="neural_ode_remifentanil",
        run_id=run_id,
        fold_index=fold_index,
        split_name=split_name,
        patients=patients,
        predictions_by_patient=predictions_by_patient,
        covariate_columns=NODE_COVARIATES,
    )


def _serialize_model(model: remifentanil_node.RemiNODE, path: Path) -> None:
    with path.open("wb") as handle:
        eqx.tree_serialise_leaves(handle, model)


class RemifentanilNODEAdapter(AnalysisAdapter):
    """Adapter wrapping the maintained remifentanil NODE module."""

    analysis_name = "neural_ode_remifentanil"

    def prepare_inputs(self, dataset: Any, fold_spec: Any, config: dict[str, Any]) -> RemiNODEPreparedInputs:
        return RemiNODEPreparedInputs(fold_data=build_fold_data(dataset, fold_spec))

    def fit(
        self,
        model_inputs: RemiNODEPreparedInputs,
        config: dict[str, Any],
        reporter: Callable[[str], None],
    ) -> RemiNODETrainedState:
        reporter(f"Training NODE fold {model_inputs.fold_data.fold_index}")
        hyperparams = remifentanil_node.resolve_hyperparams(
            {
                "model": config["model"],
                "training": config["training"],
            }
        )
        model, scalers = remifentanil_node.train_remifentanil_node(
            model_inputs.fold_data.train.raw_patients,
            hyperparams=hyperparams,
            seed=int(config.get("seed", 42)) + model_inputs.fold_data.fold_index,
        )
        fold_index = model_inputs.fold_data.fold_index
        run_id = config["run_id"]
        predictions = {
            "train": _predict_split(
                model=model,
                scalers=scalers,
                split_name="train",
                fold_index=fold_index,
                patients=model_inputs.fold_data.train.raw_patients,
                experiment_name=config["experiment_name"],
                run_id=run_id,
            ),
            "test": _predict_split(
                model=model,
                scalers=scalers,
                split_name="test",
                fold_index=fold_index,
                patients=model_inputs.fold_data.test.raw_patients,
                experiment_name=config["experiment_name"],
                run_id=run_id,
            ),
        }
        return RemiNODETrainedState(
            fold_index=fold_index,
            model=model,
            scalers=scalers,
            predictions=predictions,
            history=[],
            metadata={
                "hyperparameter_source": config["source_config_path"],
                "resolved_hyperparameters": hyperparams,
            },
        )

    def predict(self, trained_state: RemiNODETrainedState, split_name: str) -> pd.DataFrame:
        return trained_state.predictions[split_name].copy()

    def evaluate(self, long_predictions: pd.DataFrame) -> pd.DataFrame:
        return metrics_frame(long_predictions)

    def save_artifacts(self, trained_state: RemiNODETrainedState, bundle_writer: Any) -> None:
        bundle_writer.write_history(trained_state.fold_index, trained_state.history)
        bundle_writer.write_metadata(trained_state.fold_index, "node_metadata", trained_state.metadata)
        bundle_writer.write_checkpoint(
            trained_state.fold_index,
            "remifentanil_node.eqx",
            trained_state.model,
            serializer=_serialize_model,
        )
        bundle_writer.write_checkpoint(trained_state.fold_index, "node_scalers.pkl", trained_state.scalers)
