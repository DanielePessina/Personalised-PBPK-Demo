"""Optax-backed NLME adapter for patient-level cross-validation."""

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

from pharmacokinetics import nlme, remifentanil

from .base import AnalysisAdapter, PreparedFoldData, build_fold_data, build_prediction_frame, metrics_frame

EPS = 1e-8
NLME_COVARIATES = ("age", "weight", "height", "bsa")


@dataclass(slots=True)
class NLMEPreparedInputs:
    """Prepared fold inputs for the NLME adapter."""

    fold_data: PreparedFoldData
    train_covariates: jnp.ndarray
    test_covariates: jnp.ndarray
    scaler: MinMaxScaler


@dataclass(slots=True)
class NLMETrainedState:
    """Trained NLME fold state and persisted payloads."""

    fold_index: int
    model: nlme.NLMEModel
    scaler: MinMaxScaler
    predictions: dict[str, pd.DataFrame]
    history: list[dict[str, float]]
    metadata: dict[str, Any]


def _covariate_matrix(physio_patients: list[remifentanil.PhysiologicalParameters]) -> jnp.ndarray:
    return jnp.asarray(
        [[patient.age, patient.weight, patient.height, patient.bsa] for patient in physio_patients],
        dtype=jnp.float64,
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
def _nlme_loss(
    model: nlme.NLMEModel,
    train_covariates: jnp.ndarray,
    train_physio_patients: list[remifentanil.PhysiologicalParameters],
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    kinetic_matrix = model.individual_parameters(train_covariates, use_eta=True)
    predictions = _simulate_with_individual_kinetics(train_physio_patients, kinetic_matrix)
    observed = jnp.stack([patient.c_meas for patient in train_physio_patients])
    mask = jnp.stack([patient.mask for patient in train_physio_patients])

    sigma_add = model.sigma_add()
    sigma_prop = model.sigma_prop()
    variance = jnp.clip(sigma_add**2 + (sigma_prop * predictions) ** 2, 1e-12, None)
    residuals = observed - predictions

    data_nll = 0.5 * jnp.sum(jnp.where(mask, (residuals**2) / variance + jnp.log(2.0 * jnp.pi * variance), 0.0))
    chol_diag = model.L_diag()
    re_nll = 0.5 * (
        model.eta.shape[0] * 2.0 * jnp.sum(jnp.log(chol_diag))
        + jnp.sum((model.eta / chol_diag) ** 2)
    )
    loss = data_nll + re_nll
    rmse = jnp.sqrt(jnp.sum(jnp.where(mask, residuals**2, 0.0)) / (jnp.sum(mask) + EPS))
    return loss, {"rmse": rmse}


def _predict_split(
    *,
    model: nlme.NLMEModel,
    covariates: jnp.ndarray,
    split_name: str,
    patients: list[remifentanil.RawPatient],
    physio_patients: list[remifentanil.PhysiologicalParameters],
    experiment_name: str,
    run_id: str,
    fold_index: int,
) -> pd.DataFrame:
    use_eta = split_name == "train"
    kinetic_matrix = model.individual_parameters(covariates, use_eta=use_eta)
    predictions = np.asarray(_simulate_with_individual_kinetics(physio_patients, kinetic_matrix))
    predictions_by_patient: dict[int, np.ndarray] = {}
    for patient, patient_predictions in zip(patients, predictions, strict=True):
        valid_count = int(np.asarray(patient.mask).sum())
        predictions_by_patient[int(patient.id)] = np.asarray(patient_predictions)[:valid_count]
    return build_prediction_frame(
        experiment_name=experiment_name,
        analysis="nlme_optax",
        run_id=run_id,
        fold_index=fold_index,
        split_name=split_name,
        patients=patients,
        predictions_by_patient=predictions_by_patient,
        covariate_columns=NLME_COVARIATES,
    )


def _serialize_eqx_model(model: nlme.NLMEModel, path: Path) -> None:
    with path.open("wb") as handle:
        eqx.tree_serialise_leaves(handle, model)


class NLMEOptaxAdapter(AnalysisAdapter):
    """Clean NLME adapter used by the experiment runner."""

    analysis_name = "nlme_optax"

    def prepare_inputs(
        self,
        dataset: Any,
        fold_spec: Any,
        config: dict[str, Any],
    ) -> NLMEPreparedInputs:
        fold_data = build_fold_data(dataset, fold_spec)
        scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
        train_covariates = np.asarray(_covariate_matrix(fold_data.train.physio_patients))
        test_covariates = np.asarray(_covariate_matrix(fold_data.test.physio_patients))
        return NLMEPreparedInputs(
            fold_data=fold_data,
            train_covariates=jnp.asarray(scaler.fit_transform(train_covariates), dtype=jnp.float64),
            test_covariates=jnp.asarray(scaler.transform(test_covariates), dtype=jnp.float64),
            scaler=scaler,
        )

    def fit(
        self,
        model_inputs: NLMEPreparedInputs,
        config: dict[str, Any],
        reporter: Callable[[str], None],
    ) -> NLMETrainedState:
        model = nlme.NLMEModel(n_patients=len(model_inputs.fold_data.train.physio_patients))
        optimizer = optax.adam(float(config["optimizer"]["learning_rate"]))
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        history: list[dict[str, float]] = []
        epochs = int(config["optimizer"]["epochs"])
        report_every = max(1, int(config["optimizer"].get("report_every", 10)))

        for epoch in range(epochs):
            (loss, aux), grads = _nlme_loss(model, model_inputs.train_covariates, model_inputs.fold_data.train.physio_patients)
            updates, opt_state = optimizer.update(grads, opt_state, params=model)
            model = eqx.apply_updates(model, updates)
            record = {
                "epoch": float(epoch),
                "loss": float(loss),
                "rmse": float(aux["rmse"]),
            }
            history.append(record)
            if epoch % report_every == 0 or epoch == epochs - 1:
                reporter(f"NLME fold {model_inputs.fold_data.fold_index}: epoch={epoch} loss={record['loss']:.4f}")

        run_id = config["run_id"]
        fold_index = model_inputs.fold_data.fold_index
        predictions = {
            "train": _predict_split(
                model=model,
                covariates=model_inputs.train_covariates,
                split_name="train",
                patients=model_inputs.fold_data.train.raw_patients,
                physio_patients=model_inputs.fold_data.train.physio_patients,
                experiment_name=config["experiment_name"],
                run_id=run_id,
                fold_index=fold_index,
            ),
            "test": _predict_split(
                model=model,
                covariates=model_inputs.test_covariates,
                split_name="test",
                patients=model_inputs.fold_data.test.raw_patients,
                physio_patients=model_inputs.fold_data.test.physio_patients,
                experiment_name=config["experiment_name"],
                run_id=run_id,
                fold_index=fold_index,
            ),
        }
        metadata = {
            "population_parameters": {
                name: float(value)
                for name, value in zip(
                    nlme.param_names,
                    np.asarray(model.natural_from_pre(model.pop_pre)),
                    strict=True,
                )
            },
            "residual_error": {
                "sigma_add": float(model.sigma_add()),
                "sigma_prop": float(model.sigma_prop()),
            },
            "covariate_names": list(NLME_COVARIATES),
            "random_effects": np.asarray(model.eta).tolist(),
            "covariance_cholesky_diagonal": np.asarray(model.L_diag()).tolist(),
        }
        return NLMETrainedState(
            fold_index=fold_index,
            model=model,
            scaler=model_inputs.scaler,
            predictions=predictions,
            history=history,
            metadata=metadata,
        )

    def predict(self, trained_state: NLMETrainedState, split_name: str) -> pd.DataFrame:
        return trained_state.predictions[split_name].copy()

    def evaluate(self, long_predictions: pd.DataFrame) -> pd.DataFrame:
        return metrics_frame(long_predictions)

    def save_artifacts(self, trained_state: NLMETrainedState, bundle_writer: Any) -> None:
        bundle_writer.write_history(trained_state.fold_index, trained_state.history)
        bundle_writer.write_metadata(trained_state.fold_index, "nlme_metadata", trained_state.metadata)
        bundle_writer.write_checkpoint(
            trained_state.fold_index,
            "nlme_model.eqx",
            trained_state.model,
            serializer=_serialize_eqx_model,
        )
