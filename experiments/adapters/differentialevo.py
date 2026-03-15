"""Evosax Differential Evolution adapter for remifentanil kinetic parameter fitting."""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Any, Callable

from evosax.algorithms import DifferentialEvolution
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from pharmacokinetics import remifentanil

from .base import AnalysisAdapter, PreparedFoldData, build_fold_data, build_prediction_frame, metrics_frame

EPS = 1e-12
DE_COVARIATES = ("age", "weight", "height", "bsa")
PARAMETER_NAMES = remifentanil.KINETIC_PARAMETER_NAMES


@dataclass(slots=True)
class DifferentialEvoParameterSpec:
    """Decoded parameter bounds and scaling rules for one DE run."""

    names: tuple[str, ...]
    lows: jnp.ndarray
    highs: jnp.ndarray
    log_scale_mask: jnp.ndarray
    scales: tuple[str, ...]


@dataclass(slots=True)
class DifferentialEvoPreparedInputs:
    """Prepared fold inputs for the Differential Evolution adapter."""

    fold_data: PreparedFoldData
    train_stacked_physio: Any


@dataclass(slots=True)
class DifferentialEvoTrainedState:
    """Fold-local outputs for one completed Differential Evolution run."""

    fold_index: int
    best_parameters: np.ndarray
    predictions: dict[str, pd.DataFrame]
    history: list[dict[str, float]]
    metadata: dict[str, Any]
    best_params_table: pd.DataFrame
    final_population_table: pd.DataFrame


def _parameter_spec_from_config(config: dict[str, Any]) -> DifferentialEvoParameterSpec:
    """Parse and validate optimizer parameter definitions from YAML config."""
    configured_parameters = config["parameters"]
    if len(configured_parameters) != len(PARAMETER_NAMES):
        raise ValueError(
            f"Differential Evolution expects {len(PARAMETER_NAMES)} parameters, "
            f"received {len(configured_parameters)}."
        )

    names = tuple(str(item["name"]) for item in configured_parameters)
    if names != tuple(PARAMETER_NAMES):
        raise ValueError(
            "Differential Evolution parameters must match remifentanil.KINETIC_PARAMETER_NAMES order. "
            f"Expected {PARAMETER_NAMES}, received {names}."
        )

    scales = tuple(str(item["scale"]) for item in configured_parameters)
    invalid_scales = sorted({scale for scale in scales if scale not in {"linear", "log"}})
    if invalid_scales:
        raise ValueError(f"Unsupported parameter scales: {invalid_scales}")

    return DifferentialEvoParameterSpec(
        names=names,
        lows=jnp.asarray([float(item["low"]) for item in configured_parameters], dtype=jnp.float64),
        highs=jnp.asarray([float(item["high"]) for item in configured_parameters], dtype=jnp.float64),
        log_scale_mask=jnp.asarray([scale == "log" for scale in scales], dtype=bool),
        scales=scales,
    )


def _stack_physio_patients(physio_patients: list[remifentanil.PhysiologicalParameters]) -> Any:
    """Stack patient pytrees once so objective evaluation can vmap over candidates."""
    return jax.tree_util.tree_map(lambda *values: jnp.stack(values), *physio_patients)


def _decode_population(
    normalized_population: jnp.ndarray,
    parameter_spec: DifferentialEvoParameterSpec,
) -> jnp.ndarray:
    """Decode unit-cube candidates into physical PBPK parameter values."""
    return _decode_population_arrays(
        normalized_population,
        parameter_spec.lows,
        parameter_spec.highs,
        parameter_spec.log_scale_mask,
    )


def _decode_population_arrays(
    normalized_population: jnp.ndarray,
    lows: jnp.ndarray,
    highs: jnp.ndarray,
    log_scale_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Decode unit-cube candidates into physical PBPK parameter values using array-only inputs."""
    clipped = jnp.clip(normalized_population, 0.0, 1.0)
    linear_values = lows + clipped * (highs - lows)
    log_values = jnp.exp(
        jnp.log(lows) + clipped * (jnp.log(highs) - jnp.log(lows))
    )
    return jnp.where(log_scale_mask[None, :], log_values, linear_values)


def _population_objective_chunk(
    normalized_population: jnp.ndarray,
    train_stacked_physio: Any,
    lows: jnp.ndarray,
    highs: jnp.ndarray,
    log_scale_mask: jnp.ndarray,
    objective_scale: float,
) -> jnp.ndarray:
    """Evaluate weighted NLL for a chunk of normalized candidates."""
    decoded_population = _decode_population_arrays(normalized_population, lows, highs, log_scale_mask)

    def simulate_candidate(kinetic_parameters: jnp.ndarray) -> jnp.ndarray:
        return remifentanil._vectorized_simulate_impl_separated(train_stacked_physio, kinetic_parameters)

    predicted = jax.vmap(simulate_candidate)(decoded_population)
    observed = train_stacked_physio.c_meas
    mask = train_stacked_physio.mask
    variance = jnp.clip((objective_scale * observed) ** 2, min=EPS)
    residual = predicted - observed[None, :, :]
    terms = 0.5 * (jnp.log(2.0 * jnp.pi * variance)[None, :, :] + (residual**2) / variance[None, :, :])
    masked_terms = jnp.where(mask[None, :, :], terms, 0.0)
    return jnp.sum(masked_terms, axis=(1, 2))


_jitted_population_objective_chunk = jax.jit(_population_objective_chunk)


def _evaluate_candidate_population(
    normalized_population: jnp.ndarray,
    train_stacked_physio: Any,
    parameter_spec: DifferentialEvoParameterSpec,
    objective_scale: float,
    candidate_batch_size: int,
    *,
    jit_enabled: bool,
) -> jnp.ndarray:
    """Evaluate a full candidate population, chunking to control memory use."""
    evaluator = _jitted_population_objective_chunk if jit_enabled else _population_objective_chunk
    population_size = int(normalized_population.shape[0])
    if candidate_batch_size >= population_size:
        return evaluator(
            normalized_population,
            train_stacked_physio,
            parameter_spec.lows,
            parameter_spec.highs,
            parameter_spec.log_scale_mask,
            objective_scale,
        )
    batches: list[jnp.ndarray] = []
    for start in range(0, population_size, candidate_batch_size):
        stop = min(start + candidate_batch_size, population_size)
        batches.append(
            evaluator(
                normalized_population[start:stop],
                train_stacked_physio,
                parameter_spec.lows,
                parameter_spec.highs,
                parameter_spec.log_scale_mask,
                objective_scale,
            )
        )
    return jnp.concatenate(batches, axis=0)


def _weighted_negative_log_likelihood_scalar(
    kinetic_parameters: np.ndarray,
    train_physio_patients: list[remifentanil.PhysiologicalParameters],
    objective_scale: float,
) -> float:
    """Return the pooled weighted NLL for a single decoded parameter vector."""
    predicted = np.asarray(
        remifentanil.vectorized_simulate_separated(
            train_physio_patients,
            np.asarray(kinetic_parameters, dtype=np.float64),
        )
    )
    total = 0.0
    for patient, patient_predicted in zip(train_physio_patients, predicted, strict=True):
        mask = np.asarray(patient.mask, dtype=bool)
        observed = np.asarray(patient.c_meas, dtype=np.float64)[mask]
        prediction = np.asarray(patient_predicted, dtype=np.float64)[mask]
        variance = np.clip((objective_scale * observed) ** 2, a_min=EPS, a_max=None)
        residual = prediction - observed
        total += float(np.sum(0.5 * (np.log(2.0 * np.pi * variance) + (residual**2) / variance)))
    return total


def _best_params_table(
    parameter_spec: DifferentialEvoParameterSpec,
    best_parameters: np.ndarray,
) -> pd.DataFrame:
    """Build a stable best-parameter artifact table."""
    rows = []
    for name, value, low, high, scale in zip(
        parameter_spec.names,
        best_parameters,
        np.asarray(parameter_spec.lows),
        np.asarray(parameter_spec.highs),
        parameter_spec.scales,
        strict=True,
    ):
        rows.append(
            {
                "parameter": name,
                "value": float(value),
                "low": float(low),
                "high": float(high),
                "scale": scale,
            }
        )
    return pd.DataFrame(rows)


def _final_population_table(
    normalized_population: np.ndarray,
    fitness: np.ndarray,
    parameter_spec: DifferentialEvoParameterSpec,
) -> pd.DataFrame:
    """Build a compact final-population artifact with decoded physical parameters."""
    decoded_population = np.asarray(
        _decode_population(jnp.asarray(normalized_population, dtype=jnp.float64), parameter_spec),
        dtype=np.float64,
    )
    order = np.argsort(np.asarray(fitness, dtype=np.float64))
    rows = []
    for rank, member_index in enumerate(order):
        row = {
            "member_index": int(member_index),
            "rank": int(rank),
            "fitness": float(fitness[member_index]),
        }
        for name, value in zip(parameter_spec.names, decoded_population[member_index], strict=True):
            row[name] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


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


def _build_initial_population(
    config: dict[str, Any],
    key: jax.Array,
    population_size: int,
    num_dimensions: int,
) -> jnp.ndarray:
    """Construct the initial normalized candidate population."""
    init_config = config["initialization"]
    strategy = str(init_config["strategy"])
    center_values = init_config.get("center", [0.5] * num_dimensions)
    if len(center_values) != num_dimensions:
        raise ValueError(f"initialization.center must have length {num_dimensions}.")

    center = jnp.asarray([float(value) for value in center_values], dtype=jnp.float64)
    if strategy == "uniform":
        population = jax.random.uniform(key, (population_size, num_dimensions), dtype=jnp.float64)
    elif strategy in {"normal", "centered_normal"}:
        stddev = float(init_config["normal_stddev"])
        population = center[None, :] + stddev * jax.random.normal(
            key,
            (population_size, num_dimensions),
            dtype=jnp.float64,
        )
    else:
        raise ValueError(f"Unsupported Differential Evolution initialization.strategy='{strategy}'.")

    population = jnp.clip(population, 0.0, 1.0)
    if bool(init_config.get("include_center_candidate", False)):
        population = population.at[0].set(center)
    return population


class DifferentialEvolutionAdapter(AnalysisAdapter):
    """Population-only evosax Differential Evolution adapter for remifentanil."""

    analysis_name = "differentialevo"

    def prepare_inputs(
        self,
        dataset: Any,
        fold_spec: Any,
        config: dict[str, Any],
    ) -> DifferentialEvoPreparedInputs:
        """Prepare fold-local patient views and stacked training tensors."""
        del config
        fold_data = build_fold_data(dataset, fold_spec)
        return DifferentialEvoPreparedInputs(
            fold_data=fold_data,
            train_stacked_physio=_stack_physio_patients(fold_data.train.physio_patients),
        )

    def fit(
        self,
        model_inputs: DifferentialEvoPreparedInputs,
        config: dict[str, Any],
        reporter: Callable[[str], None],
    ) -> DifferentialEvoTrainedState:
        """Fit one Differential Evolution run for a single fold."""
        if str(config["objective"]["name"]) != "weighted_nll":
            raise ValueError("Differential Evolution currently supports only objective.name='weighted_nll'.")

        parameter_spec = _parameter_spec_from_config(config)
        objective_scale = float(config["objective"]["scale"])
        optimizer_config = config["optimizer"]
        evaluation_config = config["evaluation"]
        artifact_config = config["artifacts"]

        population_size = int(optimizer_config["population_size"])
        num_generations = int(optimizer_config["num_generations"])
        report_every = max(1, int(optimizer_config.get("report_every", 10)))
        history_stride = max(1, int(artifact_config.get("history_stride", report_every)))
        candidate_batch_size = min(population_size, max(1, int(evaluation_config["candidate_batch_size"])))
        jit_enabled = bool(evaluation_config["jit"])

        base_key = jax.random.fold_in(
            jax.random.PRNGKey(int(config["seed"])),
            int(model_inputs.fold_data.fold_index),
        )
        init_key, strategy_key = jax.random.split(base_key)
        initial_population = _build_initial_population(
            config,
            init_key,
            population_size,
            len(PARAMETER_NAMES),
        )
        initial_fitness = _evaluate_candidate_population(
            initial_population,
            model_inputs.train_stacked_physio,
            parameter_spec,
            objective_scale,
            candidate_batch_size,
            jit_enabled=jit_enabled,
        )

        strategy = DifferentialEvolution(
            population_size=population_size,
            solution=jnp.full((len(PARAMETER_NAMES),), 0.5, dtype=jnp.float64),
            num_diff=int(optimizer_config["num_diff"]),
        )
        params = replace(
            strategy.default_params,
            elitism=bool(optimizer_config["elitism"]),
            crossover_rate=float(optimizer_config["crossover_rate"]),
            differential_weight=float(optimizer_config["differential_weight"]),
        )
        state = strategy.init(strategy_key, initial_population, initial_fitness, params)
        initial_best_index = int(jnp.argmin(initial_fitness))
        state = replace(
            state,
            best_solution=initial_population[initial_best_index],
            best_fitness=float(initial_fitness[initial_best_index]),
        )

        history: list[dict[str, float]] = []
        started = time.perf_counter()
        reporter(
            f"DifferentialEvo fold {model_inputs.fold_data.fold_index}: "
            f"starting evosax.DifferentialEvolution with population={population_size}, "
            f"generations={num_generations}"
        )

        def generation_step(step_key: jax.Array, step_state: Any) -> tuple[Any, Any]:
            ask_key, tell_key = jax.random.split(step_key, 2)
            candidates, next_state = strategy.ask(ask_key, step_state, params)
            normalized_candidates = jnp.asarray(candidates, dtype=jnp.float64)
            fitness = _evaluate_candidate_population(
                normalized_candidates,
                model_inputs.train_stacked_physio,
                parameter_spec,
                objective_scale,
                candidate_batch_size,
                jit_enabled=jit_enabled,
            )
            next_state, metrics = strategy.tell(tell_key, normalized_candidates, fitness, next_state, params)
            return next_state, metrics

        compiled_generation_step = jax.jit(generation_step) if jit_enabled else generation_step

        for generation in range(num_generations):
            step_key = jax.random.fold_in(base_key, generation)
            state, metrics = compiled_generation_step(step_key, state)

            if generation % history_stride == 0 or generation == num_generations - 1:
                best_parameters_generation = np.asarray(
                    _decode_population(
                        jnp.asarray(metrics["best_solution_in_generation"], dtype=jnp.float64)[None, :],
                        parameter_spec,
                    )[0],
                    dtype=np.float64,
                )
                history_record: dict[str, float] = {
                    "generation": float(int(metrics["generation_counter"])),
                    "best_fitness": float(metrics["best_fitness"]),
                    "best_fitness_in_generation": float(metrics["best_fitness_in_generation"]),
                    "elapsed_seconds": float(time.perf_counter() - started),
                    "evaluation_count": float(population_size * (generation + 2)),
                }
                for name, value in zip(parameter_spec.names, best_parameters_generation, strict=True):
                    history_record[name] = float(value)
                history.append(history_record)

            if generation % report_every == 0 or generation == num_generations - 1:
                reporter(
                    f"DifferentialEvo fold {model_inputs.fold_data.fold_index}: "
                    f"generation={generation} best={float(metrics['best_fitness']):.4f}"
                )

        best_parameters = np.asarray(
            _decode_population(jnp.asarray(state.best_solution, dtype=jnp.float64)[None, :], parameter_spec)[0],
            dtype=np.float64,
        )
        final_population = np.asarray(state.population, dtype=np.float64)
        final_fitness = np.asarray(state.fitness, dtype=np.float64)
        fold_index = model_inputs.fold_data.fold_index

        predictions = {
            "train": build_prediction_frame(
                experiment_name=config["experiment_name"],
                analysis=self.analysis_name,
                run_id=config["run_id"],
                fold_index=fold_index,
                split_name="train",
                patients=model_inputs.fold_data.train.raw_patients,
                predictions_by_patient=_prediction_map(
                    model_inputs.fold_data.train.raw_patients,
                    model_inputs.fold_data.train.physio_patients,
                    best_parameters,
                ),
                covariate_columns=DE_COVARIATES,
            ),
            "test": build_prediction_frame(
                experiment_name=config["experiment_name"],
                analysis=self.analysis_name,
                run_id=config["run_id"],
                fold_index=fold_index,
                split_name="test",
                patients=model_inputs.fold_data.test.raw_patients,
                predictions_by_patient=_prediction_map(
                    model_inputs.fold_data.test.raw_patients,
                    model_inputs.fold_data.test.physio_patients,
                    best_parameters,
                ),
                covariate_columns=DE_COVARIATES,
            ),
        }

        best_params_table = _best_params_table(parameter_spec, best_parameters)
        final_population_table = _final_population_table(final_population, final_fitness, parameter_spec)
        metadata = {
            "parameter_names": list(parameter_spec.names),
            "best_objective": float(state.best_fitness),
            "best_parameters": {
                name: float(value) for name, value in zip(parameter_spec.names, best_parameters, strict=True)
            },
            "initial_best_objective": float(jnp.min(initial_fitness)),
            "objective": {
                "name": str(config["objective"]["name"]),
                "scale": objective_scale,
            },
            "optimizer": {
                "population_size": population_size,
                "num_generations": num_generations,
                "num_diff": int(optimizer_config["num_diff"]),
                "elitism": bool(optimizer_config["elitism"]),
                "crossover_rate": float(optimizer_config["crossover_rate"]),
                "differential_weight": float(optimizer_config["differential_weight"]),
                "report_every": report_every,
            },
            "initialization": {
                "strategy": str(config["initialization"]["strategy"]),
                "include_center_candidate": bool(config["initialization"].get("include_center_candidate", False)),
                "center": [float(value) for value in config["initialization"].get("center", [0.5] * len(PARAMETER_NAMES))],
                "normal_stddev": float(config["initialization"].get("normal_stddev", 0.15)),
            },
            "evaluation": {
                "candidate_batch_size": candidate_batch_size,
                "jit": jit_enabled,
            },
            "artifacts": {
                "save_final_population": bool(artifact_config["save_final_population"]),
                "history_stride": history_stride,
            },
            "parameters": [
                {
                    "name": name,
                    "low": float(low),
                    "high": float(high),
                    "scale": scale,
                }
                for name, low, high, scale in zip(
                    parameter_spec.names,
                    np.asarray(parameter_spec.lows),
                    np.asarray(parameter_spec.highs),
                    parameter_spec.scales,
                    strict=True,
                )
            ],
            "n_evaluations": int(population_size * (num_generations + 1)),
            "final_population_size": int(len(final_population_table)),
            "objective_consistency_check": float(
                _weighted_negative_log_likelihood_scalar(
                    best_parameters,
                    model_inputs.fold_data.train.physio_patients,
                    objective_scale,
                )
            ),
        }

        return DifferentialEvoTrainedState(
            fold_index=fold_index,
            best_parameters=best_parameters,
            predictions=predictions,
            history=history,
            metadata=metadata,
            best_params_table=best_params_table,
            final_population_table=final_population_table,
        )

    def predict(self, trained_state: DifferentialEvoTrainedState, split_name: str) -> pd.DataFrame:
        """Return long-form predictions for one split."""
        return trained_state.predictions[split_name].copy()

    def evaluate(self, long_predictions: pd.DataFrame) -> pd.DataFrame:
        """Return stable regression metrics for one prediction frame."""
        return metrics_frame(long_predictions)

    def save_artifacts(self, trained_state: DifferentialEvoTrainedState, bundle_writer: Any) -> None:
        """Persist DE-specific fold artifacts."""
        bundle_writer.write_metadata(trained_state.fold_index, "differentialevo_metadata", trained_state.metadata)
        bundle_writer.write_table(
            trained_state.fold_index,
            "differentialevo_best_params.parquet",
            trained_state.best_params_table,
        )
        if bool(trained_state.metadata["artifacts"]["save_final_population"]):
            bundle_writer.write_table(
                trained_state.fold_index,
                "differentialevo_final_population.parquet",
                trained_state.final_population_table,
            )
