from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from notebook_support.hybrid_results import (
    load_hybrid_results,
    plot_hybrid_double_cv_parity,
    resolve_latest_hybrid_run,
)


def test_resolve_latest_hybrid_run_picks_expected_timestamped_bundle(tmp_path: Path) -> None:
    run_root = _create_hybrid_fixture(tmp_path)
    older = run_root.parent / "20260313T101647Z"
    older.mkdir(parents=True)

    resolved = resolve_latest_hybrid_run(results_root=tmp_path / "results")
    assert resolved == run_root
    assert resolved.parent.name == "hybrid_fixed_hparams"


def test_load_hybrid_results_aggregates_all_folds_and_tables(tmp_path: Path) -> None:
    run_dir = _create_hybrid_fixture(tmp_path)
    payload = load_hybrid_results(run_dir)

    assert len(payload["fold_directories"]) == 5

    fold_overview = payload["fold_overview_table"]
    assert len(fold_overview) == 5
    assert fold_overview["train_rmse"].min() > 0.0

    metrics_summary = payload["metrics_summary_table"]
    assert set(metrics_summary["split"]) == {"train", "test"}

    metrics_display = payload["metrics_display_table"]
    assert list(metrics_display.columns) == ["split", "mse", "rmse", "mae", "r2"]

    assert payload["covariate_columns"] == ["age", "weight", "height", "bsa", "dose_rate", "dose_duration"]
    assert payload["parity_data_available"] is True
    assert payload["parity_prediction_frames"]["train"].empty is False
    assert payload["parity_prediction_frames"]["test"].empty is False


def test_plot_hybrid_double_cv_parity_returns_two_axes_and_shared_figure(tmp_path: Path) -> None:
    payload = load_hybrid_results(_create_hybrid_fixture(tmp_path))
    figure = plot_hybrid_double_cv_parity(payload)

    assert len(figure.axes) == 2
    assert figure.axes[0].get_title() == "Training CV"
    assert figure.axes[1].get_title() == "Held-out Test CV"


def _create_hybrid_fixture(tmp_path: Path) -> Path:
    run_dir = tmp_path / "results" / "remifentanil_benchmark" / "hybrid_fixed_hparams" / "20260314T120000Z"
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_yaml(
        run_dir / "run_manifest.yaml",
        {
            "experiment_name": "remifentanil_benchmark",
            "analysis": "hybrid_fixed_hparams",
            "run_id": "20260314T120000Z",
            "dataset_path": str(tmp_path / "nlme-remifentanil.xlsx"),
        },
    )
    _write_yaml(
        run_dir / "config.yaml",
        {
            "resolved_config": {
                "model": {"width_size": 180, "depth": 1},
                "optimizer": {"learning_rate": 0.01, "epochs": 300, "report_every": 50},
            }
        },
    )

    for fold_index in range(5):
        fold_dir = run_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        _write_yaml(
            fold_dir / "hybrid_metadata.yaml",
            {
                "hyperparameter_source": str(tmp_path / "configs" / "experiments" / "hybrid_fixed_hparams.yaml"),
                "resolved_hyperparameters": {
                    "model": {"width_size": 180, "depth": 1},
                    "optimizer": {"learning_rate": 0.01, "epochs": 300, "report_every": 50},
                },
                "covariate_columns": ["age", "weight", "height", "bsa", "dose_rate", "dose_duration"],
            },
        )

        metrics = pd.DataFrame(
            [
                {
                    "split": "train",
                    "mse": 1.0 + fold_index,
                    "mae": 0.5 + fold_index,
                    "rmse": 1.1 + fold_index,
                    "r2": 0.9,
                    "n_points": 10,
                },
                {
                    "split": "test",
                    "mse": 2.0 + fold_index,
                    "mae": 0.8 + fold_index,
                    "rmse": 1.6 + fold_index,
                    "r2": 0.8,
                    "n_points": 5,
                },
            ]
        )
        metrics.to_parquet(fold_dir / "metrics.parquet", index=False)

        for split_name in ("train", "test"):
            predictions = pd.DataFrame(
                [
                    {
                        "experiment_name": "remifentanil_benchmark",
                        "analysis": "hybrid_fixed_hparams",
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
                        "age": 40.0,
                        "weight": 75.0,
                        "height": 178.0,
                        "bsa": 1.9,
                    }
                    for measurement_index in range(3)
                ]
            )
            predictions.to_parquet(fold_dir / f"{split_name}_predictions.parquet", index=False)

    return run_dir


def _write_yaml(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
