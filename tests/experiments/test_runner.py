"""Regression tests for CLI behavior and end-to-end orchestration wiring."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from skopt import gp_minimize as skopt_gp_minimize
from skopt import load as skopt_load

import experiments.cli as cli_module
import experiments.runner as runner_module
from experiments.adapters.base import AnalysisAdapter, build_fold_data, build_prediction_frame, metrics_frame
from experiments.validate import validate_bundle


@dataclass(slots=True)
class StubState:
    """Minimal trained state used by the stub integration adapter."""

    fold_index: int
    predictions: dict[str, pd.DataFrame]


class StubAdapter(AnalysisAdapter):
    """Fast adapter used to test the runner without heavy training."""

    analysis_name = "stub"

    def prepare_inputs(self, dataset: Any, fold_spec: Any, config: dict[str, Any]) -> Any:
        return build_fold_data(dataset, fold_spec)

    def fit(self, model_inputs: Any, config: dict[str, Any], reporter):
        predictions = {}
        for split_name, split in (
            ("train", model_inputs.train),
            ("test", model_inputs.test),
        ):
            by_patient = {}
            for patient in split.raw_patients:
                valid_mask = patient.mask
                by_patient[int(patient.id)] = patient.c_meas[valid_mask]
            predictions[split_name] = build_prediction_frame(
                experiment_name=config["experiment_name"],
                analysis="stub",
                run_id=config["run_id"],
                fold_index=model_inputs.fold_index,
                split_name=split_name,
                patients=split.raw_patients,
                predictions_by_patient=by_patient,
                covariate_columns=("age", "weight"),
            )
        return StubState(fold_index=model_inputs.fold_index, predictions=predictions)

    def predict(self, trained_state: StubState, split_name: str) -> pd.DataFrame:
        return trained_state.predictions[split_name].copy()

    def evaluate(self, long_predictions: pd.DataFrame) -> pd.DataFrame:
        return metrics_frame(long_predictions)

    def save_artifacts(self, trained_state: StubState, bundle_writer: Any) -> None:
        bundle_writer.write_history(trained_state.fold_index, [{"epoch": 0, "loss": 0.0}])
        bundle_writer.write_metadata(trained_state.fold_index, "stub_metadata", {"ok": True})


def test_prepare_folds_only_cli(monkeypatch, stub_experiment_config, tmp_path):
    """The CLI should generate folds without running analyses."""
    config = stub_experiment_config
    config.output_root = str(tmp_path / "results")
    config.analyses = ["nlme"]
    config.model_configs = {"nlme": {"source_config_path": "inline"}}
    monkeypatch.setattr(runner_module, "load_experiment_config", lambda: config)
    exit_code = runner_module.run(["--prepare-folds-only"])
    assert exit_code == 0
    assert (Path(config.output_root) / config.experiment_name / "folds.yaml").exists()


def test_stubbed_runner_produces_valid_bundle(monkeypatch, stub_experiment_config, tmp_path):
    """A lightweight stub adapter should exercise the full bundle-writing path."""
    config = stub_experiment_config
    config.output_root = str(tmp_path / "results")
    config.analyses = ["nlme"]
    config.model_configs = {"nlme": {"source_config_path": "inline"}}
    monkeypatch.setattr(runner_module, "load_experiment_config", lambda: config)
    monkeypatch.setattr(runner_module, "ADAPTER_REGISTRY", {"nlme_optax": StubAdapter})

    exit_code = runner_module.run(["--analysis", "nlme"])
    assert exit_code == 0

    analysis_root = Path(config.output_root) / config.experiment_name / "nlme_optax"
    bundles = [path for path in analysis_root.iterdir() if path.is_dir()]
    assert len(bundles) == 1
    assert validate_bundle(bundles[0]) == []
    fold_dir = bundles[0] / "fold_0"
    assert not (fold_dir / "validation_predictions.parquet").exists()
    assert not (fold_dir / "validation_metrics.parquet").exists()


def test_validate_only_cli_reports_success(monkeypatch, stub_experiment_config, tmp_path):
    """Validate-only should resolve public names to the canonical bundle directories."""
    config = stub_experiment_config
    config.output_root = str(tmp_path / "results")
    config.analyses = ["hybrid", "node"]
    config.model_configs = {
        "hybrid": {"source_config_path": "inline"},
        "node": {"source_config_path": "inline"},
    }
    monkeypatch.setattr(runner_module, "load_experiment_config", lambda: config)
    monkeypatch.setattr(
        runner_module,
        "ADAPTER_REGISTRY",
        {
            "hybrid_fixed_hparams": StubAdapter,
            "neural_ode_remifentanil": StubAdapter,
        },
    )

    assert runner_module.run(["--analysis", "hybrid", "node"]) == 0
    assert (Path(config.output_root) / config.experiment_name / "hybrid_fixed_hparams").is_dir()
    assert (Path(config.output_root) / config.experiment_name / "neural_ode_remifentanil").is_dir()
    assert runner_module.run(["--analysis", "hybrid", "node", "--validate-only"]) == 0


def test_cli_public_analysis_choices_expose_short_names_only():
    """The CLI should advertise only the supported public names."""
    parser = cli_module.build_parser()
    analysis_action = next(action for action in parser._actions if action.dest == "analysis")

    assert tuple(analysis_action.choices) == ("bayesianopt", "differentialevo", "nlme", "hybrid", "node")


def test_cli_rejects_legacy_analysis_names():
    """Legacy canonical names should not be accepted through the public CLI."""
    parser = cli_module.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--analysis", "nlme_optax"])


def test_bayesianopt_runner_produces_valid_bundle_with_model_artifacts(
    monkeypatch,
    stub_experiment_config,
    tmp_path,
):
    """The Bayesian optimization adapter should write its fold-local artifacts."""
    from experiments.adapters import bayesianopt as bayesianopt_module

    config = stub_experiment_config
    config.output_root = str(tmp_path / "results")
    config.analyses = ["bayesianopt"]
    config.model_configs = {
        "bayesianopt": {
            "source_config_path": "inline",
            "seed": 333,
            "objective": {"name": "weighted_nll", "scale": 0.1},
            "space": {
                "k_TP": {"low": 1e-4, "high": 2.0, "prior": "log-uniform"},
                "k_PT": {"low": 1e-4, "high": 2.0, "prior": "log-uniform"},
                "k_PHP": {"low": 1e-4, "high": 2.0, "prior": "log-uniform"},
                "k_HPP": {"low": 1e-4, "high": 2.0, "prior": "log-uniform"},
                "k_EL_Pl": {"low": 1e-4, "high": 30.0, "prior": "log-uniform"},
                "Eff_kid": {"low": 1e-4, "high": 1.0, "prior": "log-uniform"},
                "Eff_hep": {"low": 1e-4, "high": 0.5, "prior": "log-uniform"},
                "k_EL_Tis": {"low": 1e-4, "high": 3.0, "prior": "log-uniform"},
            },
            "optimizer": {
                "n_calls": 1,
                "n_initial_points": 1,
                "initial_point_generator": "lhs",
                "acq_func": "gp_hedge",
                "acq_optimizer": "auto",
                "noise": "gaussian",
                "xi": 0.01,
                "kappa": 1.96,
                "n_restarts_optimizer": 1,
                "deadline_minutes": None,
                "verbose": False,
            },
        }
    }
    monkeypatch.setattr(runner_module, "load_experiment_config", lambda: config)

    def fake_gp_minimize(
        func,
        dimensions,
        n_calls,
        random_state,
        n_initial_points,
        initial_point_generator,
        acq_func,
        acq_optimizer,
        noise,
        xi,
        kappa,
        n_restarts_optimizer,
        callback,
        verbose,
    ):
        del n_calls, random_state, n_initial_points, initial_point_generator, acq_func
        del acq_optimizer, noise, xi, kappa, n_restarts_optimizer, callback, verbose
        return skopt_gp_minimize(func, dimensions, n_calls=10, random_state=0)

    monkeypatch.setattr(bayesianopt_module, "gp_minimize", fake_gp_minimize)

    exit_code = runner_module.run(["--analysis", "bayesianopt"])
    assert exit_code == 0

    analysis_root = Path(config.output_root) / config.experiment_name / "bayesianopt"
    bundles = [path for path in analysis_root.iterdir() if path.is_dir()]
    assert len(bundles) == 1
    assert validate_bundle(bundles[0]) == []

    fold_dir = bundles[0] / "fold_0"
    assert (fold_dir / "bayesianopt_metadata.yaml").exists()
    assert (fold_dir / "bayesianopt_result.pkl").exists()
    assert (fold_dir / "bayesianopt_best_params.parquet").exists()

    loaded_result = skopt_load(fold_dir / "bayesianopt_result.pkl")
    assert hasattr(loaded_result, "x")
    assert hasattr(loaded_result, "fun")
    assert hasattr(loaded_result, "x_iters")
    assert "func" not in loaded_result.specs["args"]

    history = pd.read_parquet(fold_dir / "training_history.parquet")
    assert set(["iteration", "objective", "best_objective_so_far", "elapsed_seconds"]).issubset(history.columns)


def test_differentialevo_runner_produces_valid_bundle_with_model_artifacts(
    monkeypatch,
    stub_experiment_config,
    tmp_path,
):
    """The Differential Evolution adapter should write its fold-local artifacts."""
    from experiments.adapters import differentialevo as differentialevo_module

    config = stub_experiment_config
    config.output_root = str(tmp_path / "results")
    config.analyses = ["differentialevo"]
    config.model_configs = {
        "differentialevo": {
            "source_config_path": "inline",
            "seed": 333,
            "objective": {"name": "weighted_nll", "scale": 0.1},
            "parameters": [
                {"name": "k_TP", "low": 1e-4, "high": 2.0, "scale": "linear"},
                {"name": "k_PT", "low": 1e-4, "high": 2.0, "scale": "linear"},
                {"name": "k_PHP", "low": 1e-4, "high": 2.0, "scale": "linear"},
                {"name": "k_HPP", "low": 1e-4, "high": 2.0, "scale": "linear"},
                {"name": "k_EL_Pl", "low": 1e-4, "high": 30.0, "scale": "linear"},
                {"name": "Eff_kid", "low": 1e-4, "high": 1.0, "scale": "linear"},
                {"name": "Eff_hep", "low": 1e-4, "high": 0.5, "scale": "linear"},
                {"name": "k_EL_Tis", "low": 1e-4, "high": 3.0, "scale": "linear"},
            ],
            "initialization": {
                "strategy": "uniform",
                "include_center_candidate": True,
                "center": [0.5] * 8,
                "normal_stddev": 0.15,
            },
            "optimizer": {
                "population_size": 4,
                "num_generations": 2,
                "num_diff": 1,
                "elitism": True,
                "crossover_rate": 0.9,
                "differential_weight": 0.8,
                "report_every": 1,
            },
            "evaluation": {
                "candidate_batch_size": 2,
                "jit": False,
            },
            "artifacts": {
                "save_final_population": True,
                "history_stride": 1,
            },
        }
    }
    monkeypatch.setattr(runner_module, "load_experiment_config", lambda: config)

    @dataclass(slots=True)
    class FakeParams:
        elitism: bool
        crossover_rate: float
        differential_weight: float

    @dataclass(slots=True)
    class FakeState:
        population: jnp.ndarray
        fitness: jnp.ndarray
        best_solution: jnp.ndarray
        best_fitness: float
        generation_counter: int

    class FakeDifferentialEvolution:
        def __init__(self, population_size, solution, num_diff):
            del solution, num_diff
            self.population_size = population_size
            self.default_params = FakeParams(elitism=True, crossover_rate=0.9, differential_weight=0.8)

        def init(self, key, population, fitness, params):
            del key, params
            best_index = int(jnp.argmin(fitness))
            return FakeState(
                population=population,
                fitness=fitness,
                best_solution=population[best_index],
                best_fitness=float(fitness[best_index]),
                generation_counter=0,
            )

        def ask(self, key, state, params):
            del key, params
            candidates = jnp.clip(state.population + 0.05, 0.0, 1.0)
            return candidates, state

        def tell(self, key, population, fitness, state, params):
            del key, params
            best_index = int(jnp.argmin(fitness))
            best_fitness = float(min(float(fitness[best_index]), float(state.best_fitness)))
            best_solution = population[best_index] if float(fitness[best_index]) <= float(state.best_fitness) else state.best_solution
            metrics = {
                "generation_counter": state.generation_counter,
                "best_fitness": best_fitness,
                "best_fitness_in_generation": float(fitness[best_index]),
                "best_solution_in_generation": population[best_index],
            }
            return dataclass_replace(
                state,
                population=population,
                fitness=fitness,
                best_solution=best_solution,
                best_fitness=best_fitness,
                generation_counter=state.generation_counter + 1,
            ), metrics

    def fake_population_eval(normalized_population, *args, **kwargs):
        del args, kwargs
        return jnp.sum(normalized_population, axis=1)

    def fake_prediction_map(patients, *args, **kwargs):
        del args, kwargs
        return {
            int(patient.id): np.asarray(patient.c_meas, dtype=np.float64)[np.asarray(patient.mask, dtype=bool)]
            for patient in patients
        }

    monkeypatch.setattr(differentialevo_module, "DifferentialEvolution", FakeDifferentialEvolution)
    monkeypatch.setattr(differentialevo_module, "_evaluate_candidate_population", fake_population_eval)
    monkeypatch.setattr(differentialevo_module, "_weighted_negative_log_likelihood_scalar", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(differentialevo_module, "_prediction_map", fake_prediction_map)

    exit_code = runner_module.run(["--analysis", "differentialevo"])
    assert exit_code == 0

    analysis_root = Path(config.output_root) / config.experiment_name / "differentialevo"
    bundles = [path for path in analysis_root.iterdir() if path.is_dir()]
    assert len(bundles) == 1
    assert validate_bundle(bundles[0]) == []

    fold_dir = bundles[0] / "fold_0"
    assert (fold_dir / "differentialevo_metadata.yaml").exists()
    assert (fold_dir / "differentialevo_best_params.parquet").exists()
    assert (fold_dir / "differentialevo_final_population.parquet").exists()

    history = pd.read_parquet(fold_dir / "training_history.parquet")
    assert set(
        ["generation", "best_fitness", "best_fitness_in_generation", "elapsed_seconds", "evaluation_count"]
    ).issubset(history.columns)
