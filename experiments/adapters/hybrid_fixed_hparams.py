"""Fixed-hyperparameter hybrid model adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from pharmacokinetics import remifentanil

from .base import AnalysisAdapter, PreparedFoldData, build_fold_data, build_prediction_frame, metrics_frame

HYBRID_COVARIATES = ("age", "weight", "height", "bsa", "dose_rate", "dose_duration")
PARAMETER_IS_FRACTION = jnp.array((False, False, False, False, False, True, True, False))


class HybridMLP(eqx.Module):
    """MLP that predicts kinetic parameters from patient covariates."""

    network: eqx.nn.MLP

    def __init__(self, *, width_size: int, depth: int, key: jax.Array):
        self.network = eqx.nn.MLP(
            in_size=len(HYBRID_COVARIATES),
            out_size=8,
            width_size=width_size,
            depth=depth,
            activation=jax.nn.swish,
            key=key,
        )

    def __call__(self, covariates: jnp.ndarray) -> jnp.ndarray:
        preimage = self.network(covariates)
        positive = jax.nn.softplus(preimage)
        fractions = jax.nn.sigmoid(preimage)
        return jnp.where(PARAMETER_IS_FRACTION, fractions, positive)


@dataclass(slots=True)
class HybridPreparedInputs:
    """Prepared fold inputs for the hybrid adapter."""

    fold_data: PreparedFoldData
    train_covariates: jnp.ndarray
    test_covariates: jnp.ndarray
    scaler: MinMaxScaler


@dataclass(slots=True)
class HybridTrainedState:
    """Trained hybrid fold state."""

    fold_index: int
    model: HybridMLP
    scaler: MinMaxScaler
    predictions: dict[str, pd.DataFrame]
    history: list[dict[str, float]]
    metadata: dict[str, Any]


def _covariate_matrix(physio_patients: list[remifentanil.PhysiologicalParameters]) -> np.ndarray:
    return np.asarray(
        [
            [
                patient.age,
                patient.weight,
                patient.height,
                patient.bsa,
                patient.dose_rate,
                patient.dose_duration,
            ]
            for patient in physio_patients
        ],
        dtype=np.float64,
    )


def _simulate_with_individual_kinetics(
    physio_patients: list[remifentanil.PhysiologicalParameters],
    kinetic_matrix: jnp.ndarray,
) -> jnp.ndarray:
    stacked = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *physio_patients)

    def simulate_single(physio_params: remifentanil.PhysiologicalParameters, kinetic_vec: jnp.ndarray) -> jnp.ndarray:
        _, predicted = remifentanil.simulate_patient_separated(physio_params, kinetic_vec)
        return predicted

    return jax.vmap(simulate_single, in_axes=(0, 0))(stacked, kinetic_matrix)


@eqx.filter_value_and_grad(has_aux=True)
def _hybrid_loss(
    model: HybridMLP,
    covariates: jnp.ndarray,
    physio_patients: list[remifentanil.PhysiologicalParameters],
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    kinetic_matrix = jax.vmap(model)(covariates)
    predictions = _simulate_with_individual_kinetics(physio_patients, kinetic_matrix)
    observed = jnp.stack([patient.c_meas for patient in physio_patients])
    mask = jnp.stack([patient.mask for patient in physio_patients])
    residuals = observed - predictions
    mse = jnp.sum(jnp.where(mask, residuals**2, 0.0)) / jnp.sum(mask)
    return mse, {"rmse": jnp.sqrt(mse)}


def _prediction_frame(
    *,
    model: HybridMLP,
    covariates: jnp.ndarray,
    split_name: str,
    fold_index: int,
    patients: list[remifentanil.RawPatient],
    physio_patients: list[remifentanil.PhysiologicalParameters],
    experiment_name: str,
    run_id: str,
) -> pd.DataFrame:
    kinetic_matrix = jax.vmap(model)(covariates)
    predictions = np.asarray(_simulate_with_individual_kinetics(physio_patients, kinetic_matrix))
    predictions_by_patient: dict[int, np.ndarray] = {}
    for patient, patient_predictions in zip(patients, predictions, strict=True):
        valid_count = int(np.asarray(patient.mask).sum())
        predictions_by_patient[int(patient.id)] = np.asarray(patient_predictions)[:valid_count]
    return build_prediction_frame(
        experiment_name=experiment_name,
        analysis="hybrid_fixed_hparams",
        run_id=run_id,
        fold_index=fold_index,
        split_name=split_name,
        patients=patients,
        predictions_by_patient=predictions_by_patient,
        covariate_columns=HYBRID_COVARIATES,
    )


def _serialize_model(model: HybridMLP, path: Path) -> None:
    with path.open("wb") as handle:
        eqx.tree_serialise_leaves(handle, model)


class HybridFixedHyperparametersAdapter(AnalysisAdapter):
    """Hybrid model adapter with YAML-fixed hyperparameters."""

    analysis_name = "hybrid_fixed_hparams"

    def prepare_inputs(self, dataset: Any, fold_spec: Any, config: dict[str, Any]) -> HybridPreparedInputs:
        fold_data = build_fold_data(dataset, fold_spec)
        scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
        train_covariates = _covariate_matrix(fold_data.train.physio_patients)
        test_covariates = _covariate_matrix(fold_data.test.physio_patients)
        return HybridPreparedInputs(
            fold_data=fold_data,
            train_covariates=jnp.asarray(scaler.fit_transform(train_covariates), dtype=jnp.float64),
            test_covariates=jnp.asarray(scaler.transform(test_covariates), dtype=jnp.float64),
            scaler=scaler,
        )

    def fit(
        self,
        model_inputs: HybridPreparedInputs,
        config: dict[str, Any],
        reporter: Callable[[str], None],
    ) -> HybridTrainedState:
        key = jax.random.PRNGKey(int(config.get("seed", 42)) + model_inputs.fold_data.fold_index)
        model = HybridMLP(
            width_size=int(config["model"]["width_size"]),
            depth=int(config["model"]["depth"]),
            key=key,
        )
        optimizer = optax.adam(float(config["optimizer"]["learning_rate"]))
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        epochs = int(config["optimizer"]["epochs"])
        report_every = max(1, int(config["optimizer"].get("report_every", 25)))
        history: list[dict[str, float]] = []

        for epoch in range(epochs):
            (loss, aux), grads = _hybrid_loss(model, model_inputs.train_covariates, model_inputs.fold_data.train.physio_patients)
            updates, opt_state = optimizer.update(grads, opt_state, params=model)
            model = eqx.apply_updates(model, updates)
            record = {"epoch": float(epoch), "loss": float(loss), "rmse": float(aux["rmse"])}
            history.append(record)
            if epoch % report_every == 0 or epoch == epochs - 1:
                reporter(f"Hybrid fold {model_inputs.fold_data.fold_index}: epoch={epoch} rmse={record['rmse']:.4f}")

        run_id = config["run_id"]
        fold_index = model_inputs.fold_data.fold_index
        predictions = {
            "train": _prediction_frame(
                model=model,
                covariates=model_inputs.train_covariates,
                split_name="train",
                fold_index=fold_index,
                patients=model_inputs.fold_data.train.raw_patients,
                physio_patients=model_inputs.fold_data.train.physio_patients,
                experiment_name=config["experiment_name"],
                run_id=run_id,
            ),
            "test": _prediction_frame(
                model=model,
                covariates=model_inputs.test_covariates,
                split_name="test",
                fold_index=fold_index,
                patients=model_inputs.fold_data.test.raw_patients,
                physio_patients=model_inputs.fold_data.test.physio_patients,
                experiment_name=config["experiment_name"],
                run_id=run_id,
            ),
        }
        metadata = {
            "hyperparameter_source": config["source_config_path"],
            "resolved_hyperparameters": {
                "model": config["model"],
                "optimizer": config["optimizer"],
            },
            "covariate_columns": list(HYBRID_COVARIATES),
        }
        return HybridTrainedState(
            fold_index=fold_index,
            model=model,
            scaler=model_inputs.scaler,
            predictions=predictions,
            history=history,
            metadata=metadata,
        )

    def predict(self, trained_state: HybridTrainedState, split_name: str) -> pd.DataFrame:
        return trained_state.predictions[split_name].copy()

    def evaluate(self, long_predictions: pd.DataFrame) -> pd.DataFrame:
        return metrics_frame(long_predictions)

    def save_artifacts(self, trained_state: HybridTrainedState, bundle_writer: Any) -> None:
        bundle_writer.write_history(trained_state.fold_index, trained_state.history)
        bundle_writer.write_metadata(trained_state.fold_index, "hybrid_metadata", trained_state.metadata)
        bundle_writer.write_checkpoint(
            trained_state.fold_index,
            "hybrid_model.eqx",
            trained_state.model,
            serializer=_serialize_model,
        )
