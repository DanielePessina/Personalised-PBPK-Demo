"""Standalone GSAX-based global sensitivity analysis for the saved hybrid model."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from sklearn.preprocessing import MinMaxScaler

from experiments.adapters.hybrid_fixed_hparams import HybridMLP
from experiments.schemas import FoldArtifact, FoldSpec
from experiments.summary import load_remifentanil_dataset
from pharmacokinetics import remifentanil
from run_paths import resolve_latest_analysis_run, resolve_repo_dataset_path

import gsax


ANALYSIS_NAME = "gsax_hybrid_gsa"
DEFAULT_EXPERIMENT_NAME = "remifentanil_benchmark"
DEFAULT_OUTPUT_ROOT = "results"
DEFAULT_SEED = 42
DEFAULT_SOBOL_N = 16384
DEFAULT_DENSE_POINTS = 500
DEFAULT_TIME_END_MINUTES = 100
MODEL_OUTPUT_NAME = "concentration"
FIXED_SEX_LABEL = "Male"

AGE_MEAN = 40.0
AGE_STD = 12.0
HEIGHT_MEAN_CM = 172.0
HEIGHT_STD_CM = 8.0
BMI_MEAN = 25.5
BMI_STD = 4.0
DOSE_RATE_MEAN = 226.13953846153845
DOSE_RATE_STD = 50.69854336444573
DOSE_DURATION_MEAN = 10.401538461538461
DOSE_DURATION_STD = 3.464320087779756


@dataclass(slots=True)
class RunSettings:
    """Configuration for one standalone GSA execution."""

    source_run_dir: str | None
    output_root: str
    experiment_name: str
    seed: int
    sobol_n: int
    dense_points: int


@dataclass(slots=True)
class RunManifest:
    """YAML-serializable manifest for a standalone GSA run."""

    experiment_name: str
    analysis: str
    run_id: str
    dataset_path: str
    dataset_fingerprint: str
    source_hybrid_run_dir: str
    source_hybrid_run_id: str
    status: str
    started_at: str
    completed_at: str | None
    wall_clock_seconds: float | None
    failure_reason: str | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse standalone script arguments."""
    parser = argparse.ArgumentParser(description="Run GSAX Sobol analysis on the saved hybrid remifentanil model.")
    parser.add_argument("--source-run-dir", default=None, help="Explicit hybrid run directory. Defaults to latest visible run.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Root directory for result bundles.")
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME, help="Experiment name under the output root.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base random seed.")
    parser.add_argument("--sobol-n", type=int, default=DEFAULT_SOBOL_N, help="GSAX base Sobol sample count.")
    parser.add_argument("--dense-points", type=int, default=DEFAULT_DENSE_POINTS, help="Shared dense time grid size.")
    return parser.parse_args(argv)


def build_settings(args: argparse.Namespace) -> RunSettings:
    """Convert parsed CLI arguments into a typed settings object."""
    return RunSettings(
        source_run_dir=args.source_run_dir,
        output_root=str(args.output_root),
        experiment_name=str(args.experiment_name),
        seed=int(args.seed),
        sobol_n=int(args.sobol_n),
        dense_points=int(args.dense_points),
    )


def write_yaml(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a mapping to YAML using the repo's block-style conventions."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)
    return output_path


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return payload


def load_fold_artifact(path: str | Path) -> FoldArtifact:
    """Load a fold artifact from YAML."""
    payload = load_yaml(path)
    return FoldArtifact.from_dict(payload)


def resolve_source_run_dir(settings: RunSettings) -> Path:
    """Resolve the source hybrid bundle used by the GSA."""
    return resolve_latest_analysis_run(
        analysis_name="hybrid_fixed_hparams",
        run_dir=settings.source_run_dir,
        results_root=settings.output_root,
        display_name="hybrid_fixed_hparams",
    )


def build_run_dir(settings: RunSettings, run_id: str) -> Path:
    """Return the output directory for the standalone GSA bundle."""
    return Path(settings.output_root).resolve() / settings.experiment_name / ANALYSIS_NAME / run_id


def dense_time_grid(dense_points: int) -> jnp.ndarray:
    """Return the shared dense time grid in minutes."""
    if dense_points <= 1:
        raise ValueError(f"dense_points must be > 1, got {dense_points}")
    return jnp.linspace(0.0, DEFAULT_TIME_END_MINUTES, dense_points, dtype=jnp.float64)


def derive_weight_kg(height_cm: np.ndarray, bmi_kg_m2: np.ndarray) -> np.ndarray:
    """Convert height and BMI to body weight in kilograms."""
    return bmi_kg_m2 * np.square(height_cm / 100.0)


def derive_bsa_m2(height_cm: np.ndarray, weight_kg: np.ndarray) -> np.ndarray:
    """Compute Mosteller body surface area in square meters."""
    return np.sqrt((height_cm * weight_kg) / 3600.0)


def healthy_problem() -> gsax.Problem:
    """Build the GSAX problem describing the standalone Sobol inputs."""
    return gsax.Problem.from_dict(
        {
            "age_years": {
                "dist": "gaussian",
                "mean": AGE_MEAN,
                "variance": AGE_STD**2,
                "low": 0.0,
            },
            "height_cm": {
                "dist": "gaussian",
                "mean": HEIGHT_MEAN_CM,
                "variance": HEIGHT_STD_CM**2,
                "low": 0.0,
            },
            "bmi_kg_m2": {
                "dist": "gaussian",
                "mean": BMI_MEAN,
                "variance": BMI_STD**2,
                "low": 0.0,
            },
            "dose_rate": {
                "dist": "gaussian",
                "mean": DOSE_RATE_MEAN,
                "variance": DOSE_RATE_STD**2,
                "low": 0.0,
            },
            "dose_duration": {
                "dist": "gaussian",
                "mean": DOSE_DURATION_MEAN,
                "variance": DOSE_DURATION_STD**2,
                "low": 0.0,
            },
        },
        output_names=(MODEL_OUTPUT_NAME,),
    )


def gsax_samples_to_frame(samples: np.ndarray) -> pd.DataFrame:
    """Convert the unique GSAX Sobol sample matrix into a patient cohort frame."""
    age_years = np.asarray(samples[:, 0], dtype=np.float64)
    height_cm = np.asarray(samples[:, 1], dtype=np.float64)
    bmi_kg_m2 = np.asarray(samples[:, 2], dtype=np.float64)
    dose_rate = np.asarray(samples[:, 3], dtype=np.float64)
    dose_duration = np.asarray(samples[:, 4], dtype=np.float64)
    weight_kg = derive_weight_kg(height_cm, bmi_kg_m2)
    bsa_m2 = derive_bsa_m2(height_cm, weight_kg)
    return pd.DataFrame(
        {
            "subject_id": np.arange(1, samples.shape[0] + 1, dtype=int),
            "sex": FIXED_SEX_LABEL,
            "age_years": age_years,
            "height_cm": height_cm,
            "bmi_kg_m2": bmi_kg_m2,
            "weight_kg": weight_kg,
            "bsa_m2": bsa_m2,
            "dose_rate": dose_rate,
            "dose_duration": dose_duration,
        }
    )


def model_covariate_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Build the saved-hybrid covariate matrix from a cohort frame."""
    return frame.loc[
        :,
        ["age_years", "weight_kg", "height_cm", "bsa_m2", "dose_rate", "dose_duration"],
    ].to_numpy(dtype=np.float64)


def build_physio_patients(frame: pd.DataFrame, dense_grid: jnp.ndarray) -> list[remifentanil.PhysiologicalParameters]:
    """Convert a patient frame into male synthetic physiological patients."""
    patients: list[remifentanil.PhysiologicalParameters] = []
    zeros = jnp.zeros_like(dense_grid)
    mask = jnp.ones_like(dense_grid, dtype=bool)
    for row in frame.to_dict(orient="records"):
        raw_patient = remifentanil.RawPatient(
            id=int(row["subject_id"]),
            t_meas=dense_grid,
            c_meas=zeros,
            mask=mask,
            dose_rate=float(row["dose_rate"]),
            dose_duration=float(row["dose_duration"]),
            age=float(row["age_years"]),
            weight=float(row["weight_kg"]),
            height=float(row["height_cm"]),
            sex=True,
            bsa=float(row["bsa_m2"]),
        )
        patients.append(remifentanil.create_physiological_parameters(raw_patient))
    return patients


def source_fold_training_frame(dataset: Any, fold_spec: FoldSpec) -> pd.DataFrame:
    """Build the source-fold training covariate frame used to reconstruct the scaler."""
    rows: list[dict[str, float]] = []
    for patient_id in fold_spec.train_patient_ids:
        patient = dataset.physio_by_id[int(patient_id)]
        rows.append(
            {
                "age_years": float(patient.age),
                "height_cm": float(patient.height),
                "weight_kg": float(patient.weight),
                "bsa_m2": float(patient.bsa),
                "dose_rate": float(patient.dose_rate),
                "dose_duration": float(patient.dose_duration),
            }
        )
    return pd.DataFrame(rows)


def rebuild_fold_scaler(dataset: Any, fold_spec: FoldSpec) -> MinMaxScaler:
    """Rebuild the MinMax scaler used by the saved hybrid fold."""
    training_frame = source_fold_training_frame(dataset, fold_spec)
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    scaler.fit(
        training_frame.loc[
            :,
            ["age_years", "weight_kg", "height_cm", "bsa_m2", "dose_rate", "dose_duration"],
        ].to_numpy(dtype=np.float64)
    )
    return scaler


def load_hybrid_model(fold_dir: Path) -> HybridMLP:
    """Deserialize a saved hybrid fold checkpoint from disk."""
    metadata = load_yaml(fold_dir / "hybrid_metadata.yaml")
    model_cfg = metadata["resolved_hyperparameters"]["model"]
    template = HybridMLP(
        width_size=int(model_cfg["width_size"]),
        depth=int(model_cfg["depth"]),
        key=jax.random.PRNGKey(0),
    )
    with (fold_dir / "hybrid_model.eqx").open("rb") as handle:
        return eqx.tree_deserialise_leaves(handle, template)


def simulate_dense_batch_vmap(
    physio_patients: list[remifentanil.PhysiologicalParameters],
    kinetic_matrix: jnp.ndarray,
    dense_grid: jnp.ndarray,
) -> jnp.ndarray:
    """Simulate dense plasma concentration for a batch of patients using vmap."""
    stacked = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *physio_patients)

    def simulate_single(physio_params: remifentanil.PhysiologicalParameters, kinetic_vec: jnp.ndarray) -> jnp.ndarray:
        _, concentration = remifentanil.simulate_patient_dense(physio_params, kinetic_vec, dense_grid)
        return concentration

    return jax.vmap(simulate_single, in_axes=(0, 0))(stacked, kinetic_matrix)


def source_fold_test_rmse(fold_dir: Path) -> float:
    """Read the held-out test RMSE from a saved hybrid fold."""
    metrics = pd.read_parquet(fold_dir / "metrics.parquet")
    test_row = metrics.loc[metrics["split"] == "test"].iloc[0]
    return float(test_row["rmse"])


def source_run_id(source_run_dir: Path) -> str:
    """Return the source run identifier."""
    return source_run_dir.name


def root_config_payload(settings: RunSettings) -> dict[str, Any]:
    """Build the saved configuration payload."""
    return {
        "analysis": ANALYSIS_NAME,
        "resolved_config": asdict(settings),
        "shared_cohort_definition": {
            "source": "derived from sampling_result.samples",
            "sex": FIXED_SEX_LABEL,
            "age_years": {"dist": "gaussian", "mean": AGE_MEAN, "std": AGE_STD, "low": 0.0},
            "height_cm": {"dist": "gaussian", "mean": HEIGHT_MEAN_CM, "std": HEIGHT_STD_CM, "low": 0.0},
            "bmi_kg_m2": {"dist": "gaussian", "mean": BMI_MEAN, "std": BMI_STD, "low": 0.0},
            "dose_rate": {"dist": "gaussian", "mean": DOSE_RATE_MEAN, "std": DOSE_RATE_STD, "low": 0.0},
            "dose_duration": {"dist": "gaussian", "mean": DOSE_DURATION_MEAN, "std": DOSE_DURATION_STD, "low": 0.0},
        },
        "sobol": {
            "n_samples": settings.sobol_n,
            "calc_second_order": False,
            "output_names": [MODEL_OUTPUT_NAME],
        },
        "dense_grid": {
            "points": settings.dense_points,
            "time_start_minutes": 0.0,
            "time_end_minutes": DEFAULT_TIME_END_MINUTES,
        },
    }


def source_hybrid_payload(source_run_dir: Path, fold_dirs: list[Path]) -> dict[str, Any]:
    """Build root metadata describing the source hybrid bundle."""
    manifest = load_yaml(source_run_dir / "run_manifest.yaml")
    return {
        "resolved_run_dir": str(source_run_dir),
        "source_run_id": source_run_id(source_run_dir),
        "manifest": manifest,
        "fold_test_rmse": {
            fold_dir.name: source_fold_test_rmse(fold_dir)
            for fold_dir in fold_dirs
        },
    }


def fold_dirs_in_run(source_run_dir: Path) -> list[Path]:
    """List fold directories in a saved hybrid bundle."""
    fold_dirs = sorted(path for path in source_run_dir.iterdir() if path.is_dir() and path.name.startswith("fold_"))
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories found under {source_run_dir}")
    return fold_dirs


def run_fold_gsa(
    *,
    dataset: Any,
    fold_spec: FoldSpec,
    fold_dir: Path,
    output_fold_dir: Path,
    sampling_result: gsax.SamplingResult,
    cohort_frame: pd.DataFrame,
    dense_grid: jnp.ndarray,
) -> None:
    """Execute one fold-local Sobol GSA using the shared sample cohort."""
    scaler = rebuild_fold_scaler(dataset, fold_spec)
    model = load_hybrid_model(fold_dir)
    scaled_covariates = scaler.transform(model_covariate_matrix(cohort_frame))
    kinetic_matrix = jax.vmap(model)(jnp.asarray(scaled_covariates, dtype=jnp.float64))
    physio_patients = build_physio_patients(cohort_frame, dense_grid)
    concentration = simulate_dense_batch_vmap(physio_patients, kinetic_matrix, dense_grid)
    outputs = concentration[:, :, None]
    sobol_result = gsax.analyze(sampling_result, outputs, prenormalize=False)

    output_fold_dir.mkdir(parents=True, exist_ok=True)
    dataset_view = sobol_result.to_dataset(time_coords=np.asarray(dense_grid, dtype=np.float64))
    dataset_view.to_netcdf(output_fold_dir / "sobol_indices.nc")
    write_yaml(
        output_fold_dir / "gsax_metadata.yaml",
        {
            "source_fold_index": fold_spec.fold_index,
            "source_fold_dir": str(fold_dir),
            "source_hybrid_test_rmse": source_fold_test_rmse(fold_dir),
            "train_patient_ids": [int(value) for value in fold_spec.train_patient_ids],
            "test_patient_ids": [int(value) for value in fold_spec.test_patient_ids],
            "sobol_samples_unique": int(np.asarray(sampling_result.samples).shape[0]),
            "sobol_expanded_total": int(sampling_result.expanded_n_total),
            "shared_sampling_path": str(output_fold_dir.parent / "sobol_samples.parquet"),
            "shared_cohort_path": str(output_fold_dir.parent / "healthy_phase1_cohort.parquet"),
            "dense_points": int(dense_grid.shape[0]),
            "output_names": [MODEL_OUTPUT_NAME],
        },
    )
    write_yaml(
        output_fold_dir / "status.yaml",
        {
            "status": "completed",
            "failure_reason": None,
        },
    )


def execute(settings: RunSettings) -> Path:
    """Run the standalone GSA and return the output bundle path."""
    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ")
    output_run_dir = build_run_dir(settings, run_id)
    output_run_dir.mkdir(parents=True, exist_ok=True)

    source_run_dir = resolve_source_run_dir(settings)
    fold_artifact = load_fold_artifact(source_run_dir / "folds.yaml")
    dataset = load_remifentanil_dataset(resolve_repo_dataset_path(fold_artifact.dataset_path))
    dense_grid = dense_time_grid(settings.dense_points)
    fold_dirs = fold_dirs_in_run(source_run_dir)
    problem = healthy_problem()
    sampling_result = gsax.sample(
        problem,
        n_samples=settings.sobol_n,
        calc_second_order=False,
        seed=settings.seed,
        verbose=False,
    )
    cohort = gsax_samples_to_frame(np.asarray(sampling_result.samples))

    manifest = RunManifest(
        experiment_name=settings.experiment_name,
        analysis=ANALYSIS_NAME,
        run_id=run_id,
        dataset_path=str(dataset.dataset_path),
        dataset_fingerprint=dataset.dataset_fingerprint,
        source_hybrid_run_dir=str(source_run_dir),
        source_hybrid_run_id=source_run_id(source_run_dir),
        status="running",
        started_at=started.isoformat(),
        completed_at=None,
        wall_clock_seconds=None,
    )
    write_yaml(output_run_dir / "config.yaml", root_config_payload(settings))
    write_yaml(output_run_dir / "run_manifest.yaml", asdict(manifest))
    write_yaml(output_run_dir / "source_hybrid_run.yaml", source_hybrid_payload(source_run_dir, fold_dirs))
    sampling_result.save(str(output_run_dir / "sobol_samples"), format="parquet")
    cohort.to_parquet(output_run_dir / "healthy_phase1_cohort.parquet", index=False)

    try:
        for fold_spec, fold_dir in zip(fold_artifact.folds, fold_dirs, strict=True):
            run_fold_gsa(
                dataset=dataset,
                fold_spec=fold_spec,
                fold_dir=fold_dir,
                output_fold_dir=output_run_dir / f"fold_{fold_spec.fold_index}",
                sampling_result=sampling_result,
                cohort_frame=cohort,
                dense_grid=dense_grid,
            )
    except Exception as exc:
        failed = asdict(manifest)
        failed["status"] = "failed"
        failed["failure_reason"] = str(exc)
        failed["completed_at"] = datetime.now(timezone.utc).isoformat()
        failed["wall_clock_seconds"] = (
            datetime.now(timezone.utc) - started
        ).total_seconds()
        write_yaml(output_run_dir / "run_manifest.yaml", failed)
        raise

    completed = asdict(manifest)
    completed["status"] = "completed"
    completed["completed_at"] = datetime.now(timezone.utc).isoformat()
    completed["wall_clock_seconds"] = (
        datetime.now(timezone.utc) - started
    ).total_seconds()
    write_yaml(output_run_dir / "run_manifest.yaml", completed)
    return output_run_dir


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    settings = build_settings(args)
    execute(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
