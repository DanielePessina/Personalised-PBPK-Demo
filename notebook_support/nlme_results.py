"""Helpers to load and summarize persisted NLME result bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from pharmacokinetics import nlme


def resolve_latest_nlme_run(
    run_dir: str | Path | None = None,
    results_root: str | Path = "results",
) -> Path:
    """Resolve the NLME run directory to analyze.

    Parameters
    ----------
    run_dir : str | Path | None, optional
        Explicit NLME run directory. When ``None``, the most recent visible
        directory under ``results/*/nlme_optax/`` is selected lexicographically.
    results_root : str | Path, optional
        Root directory that stores experiment bundles.

    Returns
    -------
    Path
        Resolved NLME run directory.
    """
    if run_dir is not None:
        resolved = Path(run_dir).resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"NLME run directory not found: {resolved}")
        return resolved

    root = Path(results_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Results directory not found: {root}")

    candidates: list[Path] = []
    for experiment_dir in sorted(root.iterdir()):
        if not experiment_dir.is_dir() or experiment_dir.name.startswith("."):
            continue
        analysis_dir = experiment_dir / "nlme_optax"
        if not analysis_dir.is_dir():
            continue
        for candidate in sorted(analysis_dir.iterdir()):
            if candidate.is_dir() and not candidate.name.startswith("."):
                candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(f"No visible nlme_optax runs found under {root}")
    return sorted(candidates, key=lambda path: path.name)[-1]


def load_nlme_results(run_dir: str | Path | None = None) -> dict[str, Any]:
    """Load and aggregate NLME bundle artifacts across folds.

    Parameters
    ----------
    run_dir : str | Path | None, optional
        Explicit NLME run directory. When ``None``, the latest visible run is
        discovered automatically.

    Returns
    -------
    dict[str, Any]
        Aggregated notebook payload containing manifest/config metadata and
        summary tables for population parameters, fixed effects, random effects,
        residual error, equations, and fit quality.
    """
    resolved_run_dir = resolve_latest_nlme_run(run_dir=run_dir)
    manifest = _read_yaml(resolved_run_dir / "run_manifest.yaml")
    config = _read_yaml(resolved_run_dir / "config.yaml")
    fold_dirs = sorted(
        path for path in resolved_run_dir.iterdir() if path.is_dir() and path.name.startswith("fold_")
    )
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories found under {resolved_run_dir}")

    population_pre = []
    population_natural = []
    betas = []
    l_diags = []
    sigma_adds = []
    sigma_props = []
    eta_stds = []
    metrics_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    parity_frames: dict[str, list[pd.DataFrame]] = {"train": [], "test": []}
    parity_missing: list[str] = []

    for fold_dir in fold_dirs:
        fold_index = int(fold_dir.name.split("_")[1])
        nlme_metadata = _read_yaml(fold_dir / "nlme_metadata.yaml")
        n_train_patients = len(nlme_metadata["random_effects"])
        model = nlme.load_model(str(fold_dir / "nlme_model.eqx"), nlme.NLMEModel(n_patients=n_train_patients))

        population_pre.append(np.asarray(model.pop_pre, dtype=float))
        population_natural.append(np.asarray(model.natural_from_pre(model.pop_pre), dtype=float))
        betas.append(np.asarray(model.beta, dtype=float))
        l_diags.append(np.asarray(model.L_diag(), dtype=float))
        sigma_adds.append(float(model.sigma_add()))
        sigma_props.append(float(model.sigma_prop()))
        eta_stds.append(np.asarray(model.eta, dtype=float).std(axis=0))

        fold_rows.append(
            {
                "fold": fold_index,
                "n_train_patients": n_train_patients,
                "sigma_add": float(model.sigma_add()),
                "sigma_prop": float(model.sigma_prop()),
            }
        )

        metrics = pd.read_parquet(fold_dir / "metrics.parquet").copy()
        metrics["fold"] = fold_index
        metrics_frames.append(metrics)

        for split_name, filename in (("train", "train_predictions.parquet"), ("test", "test_predictions.parquet")):
            prediction_path = fold_dir / filename
            if not prediction_path.is_file():
                parity_missing.append(str(prediction_path))
                continue

            predictions = pd.read_parquet(prediction_path).copy()
            required_columns = {"observed", "predicted", "outer_fold"}
            if not required_columns.issubset(predictions.columns):
                parity_missing.append(f"{prediction_path}: missing columns {sorted(required_columns - set(predictions.columns))}")
                continue

            if predictions.empty:
                parity_missing.append(f"{prediction_path}: empty")
                continue

            parity_frames[split_name].append(predictions)

    population_pre_array = np.stack(population_pre, axis=0)
    population_natural_array = np.stack(population_natural, axis=0)
    beta_array = np.stack(betas, axis=0)
    l_diag_array = np.stack(l_diags, axis=0)
    eta_std_array = np.stack(eta_stds, axis=0)
    metrics_frame = pd.concat(metrics_frames, ignore_index=True)

    population_parameter_table = pd.DataFrame(
        {
            "parameter": nlme.param_names,
            "transform": _parameter_transforms(),
            "population_mean": population_natural_array.mean(axis=0),
            "population_std": population_natural_array.std(axis=0),
        }
    )
    population_parameter_display_table = pd.DataFrame(
        {
            "parameter": population_parameter_table["parameter"],
            "transform": population_parameter_table["transform"],
            "mean": [_format_sig(value) for value in population_parameter_table["population_mean"]],
            "fold_sd": [_format_sig(value) for value in population_parameter_table["population_std"]],
        }
    )

    residual_error_table = pd.DataFrame(
        [
            {
                "parameter": "sigma_add",
                "mean": float(np.mean(sigma_adds)),
                "std": float(np.std(sigma_adds)),
            },
            {
                "parameter": "sigma_prop",
                "mean": float(np.mean(sigma_props)),
                "std": float(np.std(sigma_props)),
            },
        ]
    )
    residual_error_display_table = pd.DataFrame(
        {
            "parameter": residual_error_table["parameter"],
            "mean": [_format_sig(value) for value in residual_error_table["mean"]],
            "fold_sd": [_format_sig(value) for value in residual_error_table["std"]],
        }
    )

    mean_beta = beta_array.mean(axis=0)
    std_beta = beta_array.std(axis=0)
    fixed_effect_mean_table = pd.DataFrame(mean_beta, index=nlme.param_names, columns=nlme.covariate_names)
    fixed_effect_abs_mean_table = fixed_effect_mean_table.abs()
    fixed_effect_std_table = pd.DataFrame(std_beta, index=nlme.param_names, columns=nlme.covariate_names)

    dominant_covariate_rows = []
    for param_index, param_name in enumerate(nlme.param_names):
        ranked_indices = np.argsort(-np.abs(mean_beta[param_index]))
        top_cov_idx = ranked_indices[0]
        second_cov_idx = ranked_indices[1]
        dominant_covariate_rows.append(
            {
                "parameter": param_name,
                "dominant_covariate": nlme.covariate_names[top_cov_idx],
                "dominant_effect": _format_sig(mean_beta[param_index, top_cov_idx]),
                "second_covariate": nlme.covariate_names[second_cov_idx],
                "second_effect": _format_sig(mean_beta[param_index, second_cov_idx]),
                "max_abs_mean_beta": float(np.abs(mean_beta[param_index]).max()),
            }
        )
    dominant_covariates_table = pd.DataFrame(dominant_covariate_rows)

    random_effect_table = pd.DataFrame(
        {
            "parameter": nlme.param_names,
            "random_effect_sd_mean": l_diag_array.mean(axis=0),
            "random_effect_sd_std": l_diag_array.std(axis=0),
            "eta_train_sd_mean": eta_std_array.mean(axis=0),
            "eta_train_sd_std": eta_std_array.std(axis=0),
        }
    )
    random_effect_display_table = pd.DataFrame(
        {
            "parameter": random_effect_table["parameter"],
            "random_effect_sd": [_format_sig(value) for value in random_effect_table["random_effect_sd_mean"]],
            "eta_dispersion": [_format_sig(value) for value in random_effect_table["eta_train_sd_mean"]],
        }
    )

    metrics_summary_table = (
        metrics_frame.groupby("split", as_index=False)
        .agg(
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

    run_summary_table = pd.DataFrame(
        [
            {"field": "Experiment", "value": manifest["experiment_name"]},
            {"field": "Analysis", "value": manifest["analysis"]},
            {"field": "Run ID", "value": manifest["run_id"]},
            {"field": "Resolved run dir", "value": str(resolved_run_dir)},
            {"field": "Dataset", "value": Path(manifest["dataset_path"]).name},
            {"field": "Fold count", "value": len(fold_dirs)},
            {"field": "Covariates", "value": ", ".join(nlme.covariate_names)},
            {"field": "Epochs", "value": config["resolved_config"]["optimizer"]["epochs"]},
            {"field": "Learning rate", "value": config["resolved_config"]["optimizer"]["learning_rate"]},
        ]
    )

    equation_rows = []
    mean_population_pre = population_pre_array.mean(axis=0)
    for param_index, param_name in enumerate(nlme.param_names):
        transform = "sigmoid" if nlme.param_is_fraction[param_index] else "softplus"
        terms = [f"{mean_population_pre[param_index]:+.3f}"]
        for cov_index, covariate_name in enumerate(nlme.covariate_names):
            beta_value = mean_beta[param_index, cov_index]
            terms.append(f"{beta_value:+.3f}·{covariate_name}_scaled")
        linear_predictor = " ".join(terms).replace("+ -", "- ")
        equation_rows.append(
            {
                "parameter": param_name,
                "transform": transform,
                "equation": (
                    f"pre = {linear_predictor} + eta_i; "
                    f"{param_name}_i = {transform}(pre)"
                ),
            }
        )
    equation_table = pd.DataFrame(equation_rows)
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
        "equation_table": equation_table,
        "population_parameter_table": population_parameter_table,
        "population_parameter_display_table": population_parameter_display_table,
        "residual_error_table": residual_error_table,
        "residual_error_display_table": residual_error_display_table,
        "fixed_effect_mean_table": fixed_effect_mean_table.round(4).reset_index(names="parameter"),
        "fixed_effect_abs_mean_table": fixed_effect_abs_mean_table.round(4).reset_index(names="parameter"),
        "fixed_effect_std_table": fixed_effect_std_table.round(4).reset_index(names="parameter"),
        "dominant_covariates_table": dominant_covariates_table.sort_values(
            "max_abs_mean_beta", ascending=False, ignore_index=True
        ),
        "random_effect_table": random_effect_table,
        "random_effect_display_table": random_effect_display_table,
        "metrics_summary_table": metrics_summary_table,
        "metrics_display_table": metrics_display_table,
        "parity_data_available": parity_data_available,
        "parity_status_message": parity_status_message,
        "parity_missing_artifacts": parity_missing,
        "parity_prediction_frames": parity_data,
    }


def plot_nlme_double_cv_parity(nlme_results: dict[str, Any]) -> plt.Figure:
    """Plot side-by-side training and held-out test parity plots by outer fold.

    Parameters
    ----------
    nlme_results : dict[str, Any]
        Aggregated payload returned by ``load_nlme_results``.

    Returns
    -------
    plt.Figure
        Matplotlib figure with training and held-out test parity plots.
    """
    if not nlme_results.get("parity_data_available", False):
        raise ValueError(nlme_results.get("parity_status_message", "Parity data is unavailable."))

    train_frame = nlme_results["parity_prediction_frames"]["train"]
    test_frame = nlme_results["parity_prediction_frames"]["test"]
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


def _parameter_transforms() -> list[str]:
    return ["sigmoid" if is_fraction else "softplus" for is_fraction in nlme.param_is_fraction]


def _format_sig(value: float, digits: int = 3) -> str:
    return format(float(value), f".{digits}g")


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return payload
