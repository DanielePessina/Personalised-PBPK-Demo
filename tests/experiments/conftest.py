"""Fixtures for the experiment orchestration regression suite."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from experiments.schemas import ExperimentConfig, SplitConfig
from experiments.summary import load_remifentanil_dataset


@pytest.fixture(scope="session")
def experiment_dataset_path() -> Path:
    """Return the canonical remifentanil workbook for experiment tests."""
    path = Path(__file__).resolve().parents[2] / "nlme-remifentanil.xlsx"
    if not path.exists():
        pytest.skip("nlme-remifentanil.xlsx not found")
    return path


@pytest.fixture(scope="session")
def experiment_dataset(experiment_dataset_path):
    """Load the canonical remifentanil dataset through the experiment layer."""
    return load_remifentanil_dataset(experiment_dataset_path)


@pytest.fixture()
def stub_experiment_config(experiment_dataset_path, tmp_path) -> ExperimentConfig:
    """Provide a lightweight experiment configuration for stubbed runner tests."""
    return ExperimentConfig(
        version=1,
        experiment_name="stub_experiment",
        dataset_path=str(experiment_dataset_path),
        output_root=str(tmp_path / "results"),
        analyses=["stub"],
        split=SplitConfig(
            outer_folds=5,
            outer_seed=123,
        ),
        model_configs={"stub": {"source_config_path": "inline"}},
    )


@pytest.fixture()
def prediction_frame_template() -> pd.DataFrame:
    """Return a minimal valid prediction frame for bundle validation tests."""
    return pd.DataFrame(
        [
            {
                "experiment_name": "exp",
                "analysis": "stub",
                "run_id": "run-1",
                "outer_fold": 0,
                "inner_split": "train",
                "split": "train",
                "patient_id": 1,
                "measurement_index": 0,
                "time": 0.0,
                "observed": 1.0,
                "predicted": 1.1,
                "dose_rate": 2.0,
                "dose_duration": 3.0,
                "age": 40.0,
            }
        ]
    )
