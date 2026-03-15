from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from notebook_support.bayesianopt_results import (
    load_bayesianopt_results,
    plot_bayesianopt_double_cv_parity,
    resolve_latest_bayesianopt_run,
)


def test_resolve_latest_bayesianopt_run_picks_expected_timestamped_bundle(tmp_path) -> None:
    run_root = _create_bayesianopt_fixture(tmp_path)
    older = run_root.parent / "20260313T101647Z"
    older.mkdir(parents=True)

    resolved = resolve_latest_bayesianopt_run(results_root=tmp_path / "results")
    assert resolved == run_root
    assert resolved.parent.name == "bayesianopt"


def test_load_bayesianopt_results_aggregates_all_folds_and_tables(tmp_path) -> None:
    run_dir = _create_bayesianopt_fixture(tmp_path)
    payload = load_bayesianopt_results(run_dir)

    assert len(payload["fold_directories"]) == 5

    fold_overview = payload["fold_overview_table"]
    assert len(fold_overview) == 5
    assert fold_overview["n_evaluations"].min() > 0

    parameter_summary = payload["parameter_summary_table"]
    assert list(parameter_summary.columns) == ["parameter", "mean_value", "std_value", "low", "high", "prior"]
    assert len(parameter_summary) == 8

    metrics_summary = payload["metrics_summary_table"]
    assert set(metrics_summary["split"]) == {"train", "test"}
    assert payload["parity_data_available"] is True
    assert payload["parity_prediction_frames"]["train"].empty is False
    assert payload["parity_prediction_frames"]["test"].empty is False


def test_plot_bayesianopt_double_cv_parity_returns_two_axes_and_shared_figure(tmp_path) -> None:
    payload = load_bayesianopt_results(_create_bayesianopt_fixture(tmp_path))
    figure = plot_bayesianopt_double_cv_parity(payload)

    assert len(figure.axes) == 2
    assert figure.axes[0].get_title() == "Training CV"
    assert figure.axes[1].get_title() == "Held-out Test CV"


def _create_bayesianopt_fixture(tmp_path: Path) -> Path:
    run_dir = tmp_path / "results" / "remifentanil_benchmark" / "bayesianopt" / "20260314T120000Z"
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_yaml(
        run_dir / "run_manifest.yaml",
        {
            "experiment_name": "remifentanil_benchmark",
            "analysis": "bayesianopt",
            "run_id": "20260314T120000Z",
            "dataset_path": str(tmp_path / "nlme-remifentanil.xlsx"),
        },
    )
    _write_yaml(
        run_dir / "config.yaml",
        {
            "resolved_config": {
                "objective": {"name": "weighted_nll", "scale": 0.1},
                "optimizer": {"n_calls": 12, "n_initial_points": 4},
            }
        },
    )

    for fold_index in range(5):
        fold_dir = run_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        _write_yaml(
            fold_dir / "bayesianopt_metadata.yaml",
            {
                "best_objective": float(10.0 + fold_index),
                "n_evaluations": 12,
            },
        )

        best_params = pd.DataFrame(
            [
                {
                    "parameter": parameter,
                    "value": float(index + fold_index),
                    "low": 1e-4,
                    "high": 2.0,
                    "prior": "log-uniform",
                }
                for index, parameter in enumerate(
                    ("k_TP", "k_PT", "k_PHP", "k_HPP", "k_EL_Pl", "Eff_kid", "Eff_hep", "k_EL_Tis")
                )
            ]
        )
        best_params.to_parquet(fold_dir / "bayesianopt_best_params.parquet", index=False)

        metrics = pd.DataFrame(
            [
                {"split": "train", "rmse": 0.1 + fold_index, "mae": 0.05, "r2": 0.9, "n_points": 10},
                {"split": "test", "rmse": 0.2 + fold_index, "mae": 0.08, "r2": 0.8, "n_points": 5},
            ]
        )
        metrics.to_parquet(fold_dir / "metrics.parquet", index=False)

        for split_name in ("train", "test"):
            predictions = pd.DataFrame(
                [
                    {
                        "experiment_name": "remifentanil_benchmark",
                        "analysis": "bayesianopt",
                        "run_id": "20260314T120000Z",
                        "outer_fold": fold_index,
                        "inner_split": split_name,
                        "split": split_name,
                        "patient_id": fold_index + 1,
                        "measurement_index": measurement_index,
                        "time": float(measurement_index),
                        "observed": 1.0 + measurement_index,
                        "predicted": 1.1 + measurement_index,
                        "dose_rate": 2.0,
                        "dose_duration": 3.0,
                    }
                    for measurement_index in range(3)
                ]
            )
            predictions.to_parquet(fold_dir / f"{split_name}_predictions.parquet", index=False)

    return run_dir


def _write_yaml(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
