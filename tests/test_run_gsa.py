from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.schemas import FoldArtifact, FoldCounts, FoldSpec
from experiments.summary import fingerprint_file, load_remifentanil_dataset
from run_gsa import (
    ANALYSIS_NAME,
    dense_time_grid,
    gsax_samples_to_frame,
    main,
    rebuild_fold_scaler,
    resolve_source_run_dir,
    simulate_dense_batch_vmap,
)
from experiments.adapters.hybrid_fixed_hparams import HybridMLP
from pharmacokinetics import remifentanil


@pytest.fixture(scope="session")
def experiment_dataset():
    dataset_path = Path(__file__).resolve().parents[1] / "nlme-remifentanil.xlsx"
    if not dataset_path.exists():
        pytest.skip("nlme-remifentanil.xlsx not found")
    return load_remifentanil_dataset(dataset_path)


def test_gsax_samples_to_frame_derives_cohort_columns():
    samples = np.array(
        [
            [40.0, 172.0, 25.5, 100.0, 10.0],
            [30.0, 180.0, 30.0, 200.0, 20.0],
        ],
        dtype=np.float64,
    )
    cohort = gsax_samples_to_frame(samples)

    assert len(cohort) == 2
    assert set(cohort["sex"]) == {"Male"}
    np.testing.assert_allclose(
        cohort["weight_kg"].to_numpy(),
        cohort["bmi_kg_m2"].to_numpy() * np.square(cohort["height_cm"].to_numpy() / 100.0),
    )
    np.testing.assert_allclose(
        cohort["bsa_m2"].to_numpy(),
        np.sqrt(cohort["height_cm"].to_numpy() * cohort["weight_kg"].to_numpy() / 3600.0),
    )


def test_rebuild_fold_scaler_matches_manual_fit(experiment_dataset):
    fold_spec = FoldSpec(
        fold_index=0,
        train_patient_ids=[int(patient.id) for patient in experiment_dataset.raw_patients[:6]],
        test_patient_ids=[int(patient.id) for patient in experiment_dataset.raw_patients[6:8]],
        counts=FoldCounts(train=6, test=2),
    )
    rebuilt = rebuild_fold_scaler(experiment_dataset, fold_spec)

    rows = []
    for patient_id in fold_spec.train_patient_ids:
        patient = experiment_dataset.physio_by_id[int(patient_id)]
        rows.append([patient.age, patient.weight, patient.height, patient.bsa, patient.dose_rate, patient.dose_duration])
    manual = np.asarray(rows, dtype=np.float64)

    np.testing.assert_allclose(rebuilt.data_min_, manual.min(axis=0))
    np.testing.assert_allclose(rebuilt.data_max_, manual.max(axis=0))


def test_simulate_dense_batch_vmap_matches_loop(experiment_dataset):
    patients = experiment_dataset.physio_patients[:2]
    _, default_params = remifentanil.get_default_parameters()
    kinetics = jnp.stack([default_params, default_params * 1.05], axis=0)
    grid = jnp.linspace(0.0, 15.0, 6, dtype=jnp.float64)

    vmapped = simulate_dense_batch_vmap(patients, kinetics, grid)
    looped = []
    for patient, kinetic in zip(patients, kinetics, strict=True):
        _, concentration = remifentanil.simulate_patient_dense(patient, kinetic, grid)
        looped.append(concentration)
    expected = jnp.stack(looped, axis=0)

    np.testing.assert_allclose(np.asarray(vmapped), np.asarray(expected), rtol=1e-6, atol=1e-6)


def test_resolve_source_run_dir_uses_latest_visible_bundle(tmp_path):
    root = tmp_path / "results" / "remifentanil_benchmark" / "hybrid_fixed_hparams"
    older = root / "20260313T120000Z"
    latest = root / "20260314T120000Z"
    older.mkdir(parents=True)
    latest.mkdir(parents=True)

    settings = type(
        "Settings",
        (),
        {
            "source_run_dir": None,
            "output_root": str(tmp_path / "results"),
        },
    )()
    assert resolve_source_run_dir(settings) == latest


def test_main_runs_and_writes_expected_bundle(tmp_path, experiment_dataset):
    source_run_dir = _create_hybrid_source_bundle(tmp_path, experiment_dataset, run_id="20260313T160742Z")
    output_root = tmp_path / "out_results"

    exit_code = main(
        [
            "--source-run-dir",
            str(source_run_dir),
            "--output-root",
            str(output_root),
            "--experiment-name",
            "remifentanil_benchmark",
            "--seed",
            "7",
            "--sobol-n",
            "8",
            "--dense-points",
            "5",
        ]
    )

    assert exit_code == 0

    analysis_root = output_root / "remifentanil_benchmark" / ANALYSIS_NAME
    bundles = [path for path in analysis_root.iterdir() if path.is_dir()]
    assert len(bundles) == 1
    bundle_dir = bundles[0]
    assert (bundle_dir / "config.yaml").exists()
    assert (bundle_dir / "run_manifest.yaml").exists()
    assert (bundle_dir / "source_hybrid_run.yaml").exists()
    assert (bundle_dir / "healthy_phase1_cohort.parquet").exists()
    assert (bundle_dir / "sobol_samples.parquet").exists()
    assert (bundle_dir / "sobol_samples.json").exists()

    cohort = pd.read_parquet(bundle_dir / "healthy_phase1_cohort.parquet")
    samples = pd.read_parquet(bundle_dir / "sobol_samples.parquet")
    assert len(cohort) == len(samples)
    assert set(cohort["sex"]) == {"Male"}
    np.testing.assert_allclose(cohort["age_years"].to_numpy(), samples["age_years"].to_numpy())
    np.testing.assert_allclose(cohort["height_cm"].to_numpy(), samples["height_cm"].to_numpy())
    np.testing.assert_allclose(cohort["bmi_kg_m2"].to_numpy(), samples["bmi_kg_m2"].to_numpy())
    np.testing.assert_allclose(cohort["dose_rate"].to_numpy(), samples["dose_rate"].to_numpy())
    np.testing.assert_allclose(cohort["dose_duration"].to_numpy(), samples["dose_duration"].to_numpy())

    config = yaml.safe_load((bundle_dir / "config.yaml").read_text())
    assert "cohort_size" not in config["resolved_config"]
    assert config["shared_cohort_definition"]["source"] == "derived from sampling_result.samples"

    fold_dirs = sorted(path for path in bundle_dir.iterdir() if path.is_dir() and path.name.startswith("fold_"))
    assert [path.name for path in fold_dirs] == ["fold_0", "fold_1"]
    for fold_dir in fold_dirs:
        assert (fold_dir / "gsax_metadata.yaml").exists()
        assert (fold_dir / "status.yaml").exists()
        assert (fold_dir / "sobol_indices.nc").exists()
        metadata = yaml.safe_load((fold_dir / "gsax_metadata.yaml").read_text())
        assert metadata["shared_sampling_path"].endswith("sobol_samples.parquet")
        assert metadata["shared_cohort_path"].endswith("healthy_phase1_cohort.parquet")
        assert metadata["sobol_samples_unique"] == len(cohort)
        assert not (fold_dir / "sobol_samples.parquet").exists()
        assert not (fold_dir / "sobol_samples.json").exists()


def test_main_reuses_one_shared_design_across_all_folds(tmp_path, experiment_dataset):
    source_run_dir = _create_hybrid_source_bundle(tmp_path, experiment_dataset, run_id="20260313T160742Z")
    output_root = tmp_path / "out_results"

    assert main(
        [
            "--source-run-dir",
            str(source_run_dir),
            "--output-root",
            str(output_root),
            "--experiment-name",
            "remifentanil_benchmark",
            "--seed",
            "11",
            "--sobol-n",
            "8",
            "--dense-points",
            "5",
        ]
    ) == 0

    bundle_dir = next((output_root / "remifentanil_benchmark" / ANALYSIS_NAME).iterdir())
    cohort = pd.read_parquet(bundle_dir / "healthy_phase1_cohort.parquet")
    for fold_dir in sorted(path for path in bundle_dir.iterdir() if path.is_dir() and path.name.startswith("fold_")):
        metadata = yaml.safe_load((fold_dir / "gsax_metadata.yaml").read_text())
        assert metadata["sobol_samples_unique"] == len(cohort)


def _create_hybrid_source_bundle(tmp_path: Path, experiment_dataset, *, run_id: str) -> Path:
    run_dir = tmp_path / "source_results" / "remifentanil_benchmark" / "hybrid_fixed_hparams" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    patient_ids = [int(patient.id) for patient in experiment_dataset.raw_patients[:8]]
    folds = FoldArtifact(
        experiment_name="remifentanil_benchmark",
        dataset_path=str(experiment_dataset.dataset_path),
        dataset_fingerprint=fingerprint_file(experiment_dataset.dataset_path),
        outer_fold_count=2,
        outer_seed=123,
        generated_at="2026-03-14T12:00:00+00:00",
        folds=[
            FoldSpec(
                fold_index=0,
                train_patient_ids=patient_ids[:4],
                test_patient_ids=patient_ids[4:6],
                counts=FoldCounts(train=4, test=2),
            ),
            FoldSpec(
                fold_index=1,
                train_patient_ids=patient_ids[2:6],
                test_patient_ids=patient_ids[6:8],
                counts=FoldCounts(train=4, test=2),
            ),
        ],
    )

    _write_yaml(
        run_dir / "run_manifest.yaml",
        {
            "experiment_name": "remifentanil_benchmark",
            "analysis": "hybrid_fixed_hparams",
            "run_id": run_id,
            "dataset_path": str(experiment_dataset.dataset_path),
        },
    )
    _write_yaml(
        run_dir / "config.yaml",
        {
            "resolved_config": {
                "model": {"width_size": 4, "depth": 1},
                "optimizer": {"learning_rate": 0.01, "epochs": 5, "report_every": 1},
            }
        },
    )
    _write_yaml(run_dir / "folds.yaml", folds.to_dict())

    for fold_index in range(2):
        fold_dir = run_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        _write_yaml(
            fold_dir / "hybrid_metadata.yaml",
            {
                "resolved_hyperparameters": {
                    "model": {"width_size": 4, "depth": 1},
                    "optimizer": {"learning_rate": 0.01, "epochs": 5, "report_every": 1},
                },
                "covariate_columns": ["age", "weight", "height", "bsa", "dose_rate", "dose_duration"],
            },
        )
        metrics = pd.DataFrame(
            [
                {"split": "train", "mse": 1.0, "mae": 1.0, "rmse": 1.0, "r2": 0.5, "n_points": 10},
                {"split": "test", "mse": 2.0, "mae": 1.5, "rmse": 2.5 + fold_index, "r2": 0.4, "n_points": 10},
            ]
        )
        metrics.to_parquet(fold_dir / "metrics.parquet", index=False)

        model = HybridMLP(width_size=4, depth=1, key=jax.random.PRNGKey(fold_index))
        model = jax.tree_util.tree_map(
            lambda leaf: jnp.zeros_like(leaf) if eqx.is_inexact_array(leaf) else leaf,
            model,
        )
        with (fold_dir / "hybrid_model.eqx").open("wb") as handle:
            eqx.tree_serialise_leaves(handle, model)

    return run_dir


def _write_yaml(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
