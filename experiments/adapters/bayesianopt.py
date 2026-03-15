"""Bayesian optimization adapter for remifentanil kinetic parameter fitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
from skopt import dump as skopt_dump
from skopt import gp_minimize
from skopt.callbacks import DeadlineStopper
from skopt.space import Real

from pharmacokinetics import remifentanil

from .base import AnalysisAdapter, PreparedFoldData, build_fold_data, build_prediction_frame, metrics_frame

EPS = 1e-12
BO_COVARIATES = ("age", "weight", "height", "bsa")
PARAMETER_NAMES = remifentanil.KINETIC_PARAMETER_NAMES


@dataclass(slots=True)
class BayesianOptPreparedInputs:
    """Prepared fold inputs for the Bayesian optimization adapter."""

    fold_data: PreparedFoldData


@dataclass(slots=True)
class BayesianOptTrainedState:
    """Fold-local outputs for one completed Bayesian optimization run."""

    fold_index: int
    best_parameters: np.ndarray
    predictions: dict[str, pd.DataFrame]
    history: list[dict[str, float]]
    metadata: dict[str, Any]
    optimize_result: Any
    best_params_table: pd.DataFrame


def _weighted_negative_log_likelihood(
    physio_patients: list[remifentanil.PhysiologicalParameters],
    kinetic_parameters: np.ndarray,
    scale: float,
) -> float:
    """Return the pooled weighted negative log-likelihood for one fold."""
    predicted = np.asarray(
        remifentanil.vectorized_simulate_separated(
            physio_patients,
            np.asarray(kinetic_parameters, dtype=np.float64),
        )
    )
    total = 0.0
    for patient, patient_predicted in zip(physio_patients, predicted, strict=True):
        mask = np.asarray(patient.mask, dtype=bool)
        observed = np.asarray(patient.c_meas, dtype=np.float64)[mask]
        prediction = np.asarray(patient_predicted, dtype=np.float64)[mask]
        variance = np.clip((scale * observed) ** 2, a_min=EPS, a_max=None)
        residual = prediction - observed
        total += float(np.sum(0.5 * (np.log(2.0 * np.pi * variance) + (residual**2) / variance)))
    return total


def _prediction_map(
    patients: list[remifentanil.RawPatient],
    physio_patients: list[remifentanil.PhysiologicalParameters],
    kinetic_parameters: np.ndarray,
) -> dict[int, np.ndarray]:
    """Return valid-length predictions keyed by patient id for one split."""
    predicted = np.asarray(
        remifentanil.vectorized_simulate_separated(
            physio_patients,
            np.asarray(kinetic_parameters, dtype=np.float64),
        )
    )
    return {
        int(patient.id): np.asarray(patient_predictions, dtype=np.float64)[: int(np.asarray(patient.mask).sum())]
        for patient, patient_predictions in zip(patients, predicted, strict=True)
    }


def _space_from_config(config: dict[str, Any]) -> list[Real]:
    """Build the scikit-optimize parameter space from YAML config."""
    configured_space = config["space"]
    return [
        Real(
            float(configured_space[name]["low"]),
            float(configured_space[name]["high"]),
            prior=str(configured_space[name]["prior"]),
            name=name,
        )
        for name in PARAMETER_NAMES
    ]


def _best_params_table(config: dict[str, Any], best_parameters: np.ndarray) -> pd.DataFrame:
    """Build a stable best-parameter artifact table."""
    rows = []
    configured_space = config["space"]
    for name, value in zip(PARAMETER_NAMES, best_parameters, strict=True):
        rows.append(
            {
                "parameter": name,
                "value": float(value),
                "low": float(configured_space[name]["low"]),
                "high": float(configured_space[name]["high"]),
                "prior": str(configured_space[name]["prior"]),
            }
        )
    return pd.DataFrame(rows)


def _skopt_result_serializer(payload: Any, path: Path) -> None:
    """Persist an OptimizeResult without the non-serializable objective closure."""
    skopt_dump(payload, path, store_objective=False)


class BayesianOptimizationAdapter(AnalysisAdapter):
    """Population-only Bayesian optimization adapter for remifentanil."""

    analysis_name = "bayesianopt"

    def prepare_inputs(
        self,
        dataset: Any,
        fold_spec: Any,
        config: dict[str, Any],
    ) -> BayesianOptPreparedInputs:
        return BayesianOptPreparedInputs(fold_data=build_fold_data(dataset, fold_spec))

    def fit(
        self,
        model_inputs: BayesianOptPreparedInputs,
        config: dict[str, Any],
        reporter: Callable[[str], None],
    ) -> BayesianOptTrainedState:
        objective_scale = float(config["objective"]["scale"])
        if str(config["objective"]["name"]) != "weighted_nll":
            raise ValueError("Bayesian optimization currently supports only objective.name='weighted_nll'.")

        reporter(
            f"BayesianOpt fold {model_inputs.fold_data.fold_index}: "
            f"starting gp_minimize with {config['optimizer']['n_calls']} calls"
        )

        search_space = _space_from_config(config)
        history: list[dict[str, float]] = []
        started = time.perf_counter()

        def objective(candidate: list[float]) -> float:
            objective_value = _weighted_negative_log_likelihood(
                model_inputs.fold_data.train.physio_patients,
                np.asarray(candidate, dtype=np.float64),
                objective_scale,
            )
            elapsed_seconds = time.perf_counter() - started
            best_objective_so_far = min(
                objective_value,
                history[-1]["best_objective_so_far"] if history else objective_value,
            )
            record: dict[str, float] = {
                "iteration": float(len(history)),
                "objective": float(objective_value),
                "best_objective_so_far": float(best_objective_so_far),
                "elapsed_seconds": float(elapsed_seconds),
            }
            for name, value in zip(PARAMETER_NAMES, candidate, strict=True):
                record[name] = float(value)
            history.append(record)
            return float(objective_value)

        callbacks = []
        deadline_minutes = config["optimizer"].get("deadline_minutes")
        if deadline_minutes is not None:
            callbacks.append(DeadlineStopper(60.0 * float(deadline_minutes)))

        result = gp_minimize(
            objective,
            search_space,
            n_calls=int(config["optimizer"]["n_calls"]),
            random_state=int(config["seed"]),
            n_initial_points=int(config["optimizer"]["n_initial_points"]),
            initial_point_generator=str(config["optimizer"]["initial_point_generator"]),
            acq_func=str(config["optimizer"]["acq_func"]),
            acq_optimizer=str(config["optimizer"]["acq_optimizer"]),
            noise=config["optimizer"]["noise"],
            xi=float(config["optimizer"]["xi"]),
            kappa=float(config["optimizer"]["kappa"]),
            n_restarts_optimizer=int(config["optimizer"]["n_restarts_optimizer"]),
            callback=callbacks,
            verbose=bool(config["optimizer"]["verbose"]),
        )

        best_parameters = np.asarray(result.x, dtype=np.float64)
        fold_index = model_inputs.fold_data.fold_index
        predictions = {
            "train": build_prediction_frame(
                experiment_name=config["experiment_name"],
                analysis="bayesianopt",
                run_id=config["run_id"],
                fold_index=fold_index,
                split_name="train",
                patients=model_inputs.fold_data.train.raw_patients,
                predictions_by_patient=_prediction_map(
                    model_inputs.fold_data.train.raw_patients,
                    model_inputs.fold_data.train.physio_patients,
                    best_parameters,
                ),
                covariate_columns=BO_COVARIATES,
            ),
            "test": build_prediction_frame(
                experiment_name=config["experiment_name"],
                analysis="bayesianopt",
                run_id=config["run_id"],
                fold_index=fold_index,
                split_name="test",
                patients=model_inputs.fold_data.test.raw_patients,
                predictions_by_patient=_prediction_map(
                    model_inputs.fold_data.test.raw_patients,
                    model_inputs.fold_data.test.physio_patients,
                    best_parameters,
                ),
                covariate_columns=BO_COVARIATES,
            ),
        }

        best_params_table = _best_params_table(config, best_parameters)
        metadata = {
            "parameter_names": list(PARAMETER_NAMES),
            "best_objective": float(result.fun),
            "best_parameters": {
                name: float(value) for name, value in zip(PARAMETER_NAMES, best_parameters, strict=True)
            },
            "objective": {
                "name": str(config["objective"]["name"]),
                "scale": objective_scale,
            },
            "optimizer": {
                "n_calls": int(config["optimizer"]["n_calls"]),
                "n_initial_points": int(config["optimizer"]["n_initial_points"]),
                "initial_point_generator": str(config["optimizer"]["initial_point_generator"]),
                "acq_func": str(config["optimizer"]["acq_func"]),
                "acq_optimizer": str(config["optimizer"]["acq_optimizer"]),
                "noise": config["optimizer"]["noise"],
                "xi": float(config["optimizer"]["xi"]),
                "kappa": float(config["optimizer"]["kappa"]),
                "n_restarts_optimizer": int(config["optimizer"]["n_restarts_optimizer"]),
                "deadline_minutes": None
                if deadline_minutes is None
                else float(deadline_minutes),
                "verbose": bool(config["optimizer"]["verbose"]),
            },
            "space": [
                {
                    "name": name,
                    "low": float(config["space"][name]["low"]),
                    "high": float(config["space"][name]["high"]),
                    "prior": str(config["space"][name]["prior"]),
                }
                for name in PARAMETER_NAMES
            ],
            "n_evaluations": len(history),
            "saved_result_note": (
                "bayesianopt_result.pkl is persisted with skopt.dump(store_objective=False); "
                "the fold-local objective closure is intentionally omitted."
            ),
        }

        reporter(
            f"BayesianOpt fold {fold_index}: best objective={float(result.fun):.4f} "
            f"after {len(history)} evaluations"
        )

        return BayesianOptTrainedState(
            fold_index=fold_index,
            best_parameters=best_parameters,
            predictions=predictions,
            history=history,
            metadata=metadata,
            optimize_result=result,
            best_params_table=best_params_table,
        )

    def predict(self, trained_state: BayesianOptTrainedState, split_name: str) -> pd.DataFrame:
        return trained_state.predictions[split_name].copy()

    def evaluate(self, long_predictions: pd.DataFrame) -> pd.DataFrame:
        return metrics_frame(long_predictions)

    def save_artifacts(self, trained_state: BayesianOptTrainedState, bundle_writer: Any) -> None:
        bundle_writer.write_metadata(trained_state.fold_index, "bayesianopt_metadata", trained_state.metadata)
        bundle_writer.write_checkpoint(
            trained_state.fold_index,
            "bayesianopt_result.pkl",
            trained_state.optimize_result,
            serializer=_skopt_result_serializer,
        )
        bundle_writer.write_table(
            trained_state.fold_index,
            "bayesianopt_best_params.parquet",
            trained_state.best_params_table,
        )
