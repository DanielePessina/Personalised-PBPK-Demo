from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from notebook_support.gsax_results import load_gsax_results, resolve_latest_gsax_run


def test_resolve_latest_gsax_run_picks_expected_timestamped_bundle(tmp_path: Path) -> None:
    run_root = _create_gsax_fixture(tmp_path)
    older = run_root.parent / "20260313T101647Z"
    older.mkdir(parents=True)

    resolved = resolve_latest_gsax_run(results_root=tmp_path / "results")
    assert resolved == run_root
    assert resolved.parent.name == "gsax_hybrid_gsa"


def test_load_gsax_results_aggregates_root_and_fold_payloads(tmp_path: Path) -> None:
    run_dir = _create_gsax_fixture(tmp_path)
    payload = load_gsax_results(run_dir)

    assert len(payload["fold_directories"]) == 2
    assert len(payload["healthy_cohort"]) == 8
    assert len(payload["shared_samples_frame"]) == 8
    assert payload["default_source_fold"] == 0
    assert list(payload["fold_overview_table"].columns) == [
        "fold",
        "source_hybrid_test_rmse",
        "sobol_samples_unique",
        "sobol_expanded_total",
        "dense_points",
    ]
    assert set(payload["sobol_datasets"]) == {0, 1}
    assert "Source hybrid run ID" in payload["run_summary_table"]["field"].tolist()


def _create_gsax_fixture(tmp_path: Path) -> Path:
    run_dir = tmp_path / "results" / "remifentanil_benchmark" / "gsax_hybrid_gsa" / "20260314T120000Z"
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_yaml(
        run_dir / "config.yaml",
        {
            "analysis": "gsax_hybrid_gsa",
            "resolved_config": {
                "sobol_n": 64,
                "dense_points": 5,
            },
        },
    )
    _write_yaml(
        run_dir / "run_manifest.yaml",
        {
            "experiment_name": "remifentanil_benchmark",
            "analysis": "gsax_hybrid_gsa",
            "run_id": "20260314T120000Z",
        },
    )
    _write_yaml(
        run_dir / "source_hybrid_run.yaml",
        {
            "resolved_run_dir": str(tmp_path / "results" / "remifentanil_benchmark" / "hybrid_fixed_hparams" / "20260313T160742Z"),
            "source_run_id": "20260313T160742Z",
        },
    )
    pd.DataFrame(
        {
            "SampleID": range(8),
            "age_years": np.linspace(25, 45, 8),
            "height_cm": np.linspace(168, 176, 8),
            "bmi_kg_m2": np.linspace(22, 28, 8),
            "dose_rate": np.linspace(100, 200, 8),
            "dose_duration": np.linspace(5, 15, 8),
        }
    ).to_parquet(run_dir / "sobol_samples.parquet", index=False)
    (run_dir / "sobol_samples.json").write_text("{}", encoding="utf-8")
    pd.DataFrame(
        {
            "subject_id": range(1, 9),
            "sex": ["Male"] * 8,
            "age_years": np.linspace(25, 45, 8),
            "height_cm": np.linspace(168, 176, 8),
            "bmi_kg_m2": np.linspace(22, 28, 8),
            "weight_kg": np.linspace(68, 82, 8),
            "bsa_m2": np.linspace(1.8, 2.0, 8),
            "dose_rate": np.linspace(100, 200, 8),
            "dose_duration": np.linspace(5, 15, 8),
        }
    ).to_parquet(run_dir / "healthy_phase1_cohort.parquet", index=False)

    times = np.linspace(0.0, 20.0, 5)
    params = ["age_years", "height_cm", "bmi_kg_m2", "dose_rate", "dose_duration"]
    for fold_index, rmse in enumerate((1.25, 1.75)):
        fold_dir = run_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        _write_yaml(
            fold_dir / "gsax_metadata.yaml",
            {
                "source_hybrid_test_rmse": rmse,
                "sobol_samples_unique": 32,
                "sobol_expanded_total": 56,
                "dense_points": 5,
            },
        )
        xr.Dataset(
            data_vars={
                "S1": (("time", "output", "param"), np.full((5, 1, len(params)), 0.1 + fold_index)),
                "ST": (("time", "output", "param"), np.full((5, 1, len(params)), 0.2 + fold_index)),
            },
            coords={
                "time": times,
                "output": ["concentration"],
                "param": params,
            },
        ).to_netcdf(fold_dir / "sobol_indices.nc")

    return run_dir


def _write_yaml(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
