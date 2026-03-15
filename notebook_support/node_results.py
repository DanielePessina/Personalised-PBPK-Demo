"""Helpers to load and summarize persisted remifentanil NODE result bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from run_paths import resolve_latest_analysis_run


NODE_COVARIATE_COLUMNS = ("age", "weight", "height", "sex", "dose_rate", "dose_duration")


def resolve_latest_node_run(
    run_dir: str | Path | None = None,
    results_root: str | Path = "results",
) -> Path:
    """Resolve the remifentanil NODE run directory to analyze."""
    return resolve_latest_analysis_run(
        analysis_name="neural_ode_remifentanil",
        run_dir=run_dir,
        results_root=results_root,
        display_name="neural_ode_remifentanil",
    )


def load_node_results(run_dir: str | Path | None = None) -> dict[str, Any]:
    """Load and aggregate persisted remifentanil NODE bundle artifacts."""
    resolved_run_dir = resolve_latest_node_run(run_dir=run_dir)
    manifest = _read_yaml(resolved_run_dir / "run_manifest.yaml")
    config = _read_yaml(resolved_run_dir / "config.yaml")
    fold_dirs = sorted(
        path for path in resolved_run_dir.iterdir() if path.is_dir() and path.name.startswith("fold_")
    )
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories found under {resolved_run_dir}")

    metrics_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    parity_frames: dict[str, list[pd.DataFrame]] = {"train": [], "test": []}
    parity_missing: list[str] = []

    for fold_dir in fold_dirs:
        fold_index = int(fold_dir.name.split("_")[1])
        metrics = pd.read_parquet(fold_dir / "metrics.parquet").copy()
        metrics["fold"] = fold_index
        metrics_frames.append(metrics)

        train_metrics = metrics.loc[metrics["split"] == "train"].iloc[0]
        test_metrics = metrics.loc[metrics["split"] == "test"].iloc[0]
        fold_rows.append(
            {
                "fold": fold_index,
                "train_rmse": float(train_metrics["rmse"]),
                "test_rmse": float(test_metrics["rmse"]),
                "train_r2": float(train_metrics["r2"]),
                "test_r2": float(test_metrics["r2"]),
            }
        )

        for split_name, filename in (("train", "train_predictions.parquet"), ("test", "test_predictions.parquet")):
            prediction_path = fold_dir / filename
            if not prediction_path.is_file():
                parity_missing.append(str(prediction_path))
                continue

            predictions = pd.read_parquet(prediction_path).copy()
            required_columns = {"observed", "predicted", "outer_fold"}
            if not required_columns.issubset(predictions.columns):
                parity_missing.append(
                    f"{prediction_path}: missing columns {sorted(required_columns - set(predictions.columns))}"
                )
                continue

            if predictions.empty:
                parity_missing.append(f"{prediction_path}: empty")
                continue

            parity_frames[split_name].append(predictions)

    metrics_frame = pd.concat(metrics_frames, ignore_index=True)
    metrics_summary_table = (
        metrics_frame.groupby("split", as_index=False)
        .agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
        )
        .sort_values("split", ignore_index=True)
    )
    split_label_map = {"train": "Train", "test": "Held-out test"}
    metrics_display_table = pd.DataFrame(
        {
            "split": [split_label_map.get(value, str(value)) for value in metrics_summary_table["split"]],
            "mse": [
                f"{_format_sig(mean)} +/- {_format_sig(std)}"
                for mean, std in zip(metrics_summary_table["mse_mean"], metrics_summary_table["mse_std"], strict=True)
            ],
            "rmse": [
                f"{_format_sig(mean)} +/- {_format_sig(std)}"
                for mean, std in zip(metrics_summary_table["rmse_mean"], metrics_summary_table["rmse_std"], strict=True)
            ],
            "mae": [
                f"{_format_sig(mean)} +/- {_format_sig(std)}"
                for mean, std in zip(metrics_summary_table["mae_mean"], metrics_summary_table["mae_std"], strict=True)
            ],
            "r2": [
                f"{_format_sig(mean)} +/- {_format_sig(std)}"
                for mean, std in zip(metrics_summary_table["r2_mean"], metrics_summary_table["r2_std"], strict=True)
            ],
        }
    )

    resolved_config = config["resolved_config"]
    model_config = resolved_config["model"]
    training_config = resolved_config["training"]
    run_summary_table = pd.DataFrame(
        [
            {"field": "Experiment", "value": manifest["experiment_name"]},
            {"field": "Analysis", "value": manifest["analysis"]},
            {"field": "Run ID", "value": manifest["run_id"]},
            {"field": "Resolved run dir", "value": str(resolved_run_dir)},
            {"field": "Dataset", "value": Path(manifest["dataset_path"]).name},
            {"field": "Fold count", "value": len(fold_dirs)},
            {"field": "Covariates", "value": ", ".join(NODE_COVARIATE_COLUMNS)},
            {"field": "Width size", "value": model_config["width_size"]},
            {"field": "Depth", "value": model_config["depth"]},
            {"field": "Augment dim", "value": model_config["augment_dim"]},
            {"field": "Learning-rate strategy", "value": str(training_config["lr_strategy"])},
            {"field": "Steps strategy", "value": str(training_config["steps_strategy"])},
            {"field": "Length strategy", "value": str(training_config["length_strategy"])},
            {"field": "Batch size", "value": training_config["batch_size"]},
        ]
    )

    parity_data_available = all(parity_frames[split_name] for split_name in ("train", "test"))
    parity_data = {
        split_name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for split_name, frames in parity_frames.items()
    }
    parity_status_message = (
        "Train and held-out test predictions are available for all loaded folds."
        if parity_data_available and not parity_missing
        else "Parity plot unavailable because train/test prediction artifacts are missing or incomplete."
    )

    return {
        "resolved_run_dir": resolved_run_dir,
        "manifest": manifest,
        "config": config,
        "fold_directories": fold_dirs,
        "run_summary_table": run_summary_table,
        "fold_overview_table": pd.DataFrame(fold_rows).sort_values("fold", ignore_index=True),
        "metrics_summary_table": metrics_summary_table,
        "metrics_display_table": metrics_display_table,
        "parity_data_available": parity_data_available,
        "parity_status_message": parity_status_message,
        "parity_missing_artifacts": parity_missing,
        "parity_prediction_frames": parity_data,
    }


def plot_node_double_cv_parity(node_results: dict[str, Any]) -> plt.Figure:
    """Plot side-by-side training and held-out test parity plots by outer fold."""
    if not node_results.get("parity_data_available", False):
        raise ValueError(node_results.get("parity_status_message", "Parity data is unavailable."))

    train_frame = node_results["parity_prediction_frames"]["train"]
    test_frame = node_results["parity_prediction_frames"]["test"]
    plot_frame = pd.concat([train_frame, test_frame], ignore_index=True)
    min_val = float(min(plot_frame["observed"].min(), plot_frame["predicted"].min()))
    max_val = float(max(plot_frame["observed"].max(), plot_frame["predicted"].max()))
    padding = 0.03 * (max_val - min_val if max_val > min_val else 1.0)
    limits = (min_val - padding, max_val + padding)

    fold_ids = sorted(set(train_frame["outer_fold"].unique()).union(test_frame["outer_fold"].unique()))
    colors = plt.get_cmap("Dark2")(np.linspace(0.0, 1.0, max(len(fold_ids), 3)))[: len(fold_ids)]
    color_map = {int(fold_id): colors[idx] for idx, fold_id in enumerate(fold_ids)}

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=False)
    split_specs = [
        ("train", train_frame, axes[0], "Training CV"),
        ("test", test_frame, axes[1], "Held-out Test CV"),
    ]

    for _, split_frame, axis, title in split_specs:
        for fold_id in fold_ids:
            fold_frame = split_frame.loc[split_frame["outer_fold"] == fold_id]
            if fold_frame.empty:
                continue
            axis.scatter(
                fold_frame["observed"],
                fold_frame["predicted"],
                s=14,
                alpha=0.35,
                color=color_map[int(fold_id)],
                rasterized=True,
            )
        axis.plot(limits, limits, linestyle="--", linewidth=1.2, color="black", alpha=0.9)
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(title)
        axis.set_xlabel("Measured concentration")
        axis.set_ylabel("Predicted concentration")

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color_map[int(fold_id)], label=f"Fold {int(fold_id)}", markersize=6)
        for fold_id in fold_ids
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=max(1, len(legend_handles)),
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle("Cross-validation parity plot", y=0.98)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.95))
    return fig


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return payload


def _format_sig(value: float, digits: int = 3) -> str:
    return format(float(value), f".{digits}g")
