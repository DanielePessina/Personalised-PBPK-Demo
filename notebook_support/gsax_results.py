"""Helpers to load and summarize standalone GSAX hybrid GSA bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr
import yaml

from run_paths import resolve_latest_analysis_run


ANALYSIS_NAME = "gsax_hybrid_gsa"


def resolve_latest_gsax_run(
    run_dir: str | Path | None = None,
    results_root: str | Path = "results",
) -> Path:
    """Resolve the standalone GSAX run directory to analyze."""
    return resolve_latest_analysis_run(
        analysis_name=ANALYSIS_NAME,
        run_dir=run_dir,
        results_root=results_root,
        display_name=ANALYSIS_NAME,
    )


def load_gsax_results(run_dir: str | Path | None = None) -> dict[str, Any]:
    """Load a standalone GSAX bundle and aggregate root/fold summaries."""
    resolved_run_dir = resolve_latest_gsax_run(run_dir=run_dir)
    config = _read_yaml(resolved_run_dir / "config.yaml")
    manifest = _read_yaml(resolved_run_dir / "run_manifest.yaml")
    source_hybrid = _read_yaml(resolved_run_dir / "source_hybrid_run.yaml")
    healthy_cohort = pd.read_parquet(resolved_run_dir / "healthy_phase1_cohort.parquet")
    shared_samples_frame = pd.read_parquet(resolved_run_dir / "sobol_samples.parquet")
    fold_dirs = sorted(path for path in resolved_run_dir.iterdir() if path.is_dir() and path.name.startswith("fold_"))
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories found under {resolved_run_dir}")

    fold_rows: list[dict[str, Any]] = []
    sobol_datasets: dict[int, xr.Dataset] = {}
    fold_metadata: dict[int, dict[str, Any]] = {}

    for fold_dir in fold_dirs:
        fold_index = int(fold_dir.name.split("_")[1])
        metadata = _read_yaml(fold_dir / "gsax_metadata.yaml")
        fold_metadata[fold_index] = metadata
        sobol_datasets[fold_index] = xr.load_dataset(fold_dir / "sobol_indices.nc")
        fold_rows.append(
            {
                "fold": fold_index,
                "source_hybrid_test_rmse": float(metadata["source_hybrid_test_rmse"]),
                "sobol_samples_unique": int(metadata["sobol_samples_unique"]),
                "sobol_expanded_total": int(metadata["sobol_expanded_total"]),
                "dense_points": int(metadata["dense_points"]),
            }
        )

    fold_overview_table = pd.DataFrame(fold_rows).sort_values("fold", ignore_index=True)
    default_source_fold = int(
        fold_overview_table.sort_values("source_hybrid_test_rmse", ignore_index=True).iloc[0]["fold"]
    )
    run_summary_table = pd.DataFrame(
        [
            {"field": "Experiment", "value": manifest["experiment_name"]},
            {"field": "Analysis", "value": manifest["analysis"]},
            {"field": "Run ID", "value": manifest["run_id"]},
            {"field": "Resolved run dir", "value": str(resolved_run_dir)},
            {"field": "Source hybrid run dir", "value": str(source_hybrid["resolved_run_dir"])},
            {"field": "Source hybrid run ID", "value": str(source_hybrid["source_run_id"])},
            {"field": "Cohort size", "value": int(len(shared_samples_frame))},
            {"field": "Default source fold", "value": default_source_fold},
            {"field": "Sobol n", "value": int(config["resolved_config"]["sobol_n"])},
            {"field": "Dense points", "value": int(config["resolved_config"]["dense_points"])},
        ]
    )
    return {
        "resolved_run_dir": resolved_run_dir,
        "manifest": manifest,
        "config": config,
        "source_hybrid_run": source_hybrid,
        "healthy_cohort": healthy_cohort,
        "shared_samples_frame": shared_samples_frame,
        "fold_directories": fold_dirs,
        "fold_overview_table": fold_overview_table,
        "run_summary_table": run_summary_table,
        "sobol_datasets": sobol_datasets,
        "fold_metadata": fold_metadata,
        "default_source_fold": default_source_fold,
    }


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return payload
