from __future__ import annotations

from pathlib import Path

import numpy as np

from notebook_support.remifentanil_eda import compute_patient_summary, load_patient_covariate_frame


DATASET_PATH = Path(__file__).resolve().parents[2] / "nlme-remifentanil.xlsx"


def test_load_patient_covariate_frame_has_expected_columns_and_rows() -> None:
    patient_frame = load_patient_covariate_frame(DATASET_PATH)

    assert len(patient_frame) == 65
    assert list(patient_frame.columns) == [
        "patient_id",
        "n_samples",
        "final_measurement_time",
        "age",
        "weight",
        "height",
        "sex",
        "bsa",
    ]
    assert patient_frame["patient_id"].is_unique
    assert np.all(np.isfinite(patient_frame["n_samples"]))
    assert np.all(np.isfinite(patient_frame["final_measurement_time"]))
    assert np.all(patient_frame["n_samples"] > 0)
    assert np.all(patient_frame["final_measurement_time"] > 0.0)


def test_compute_patient_summary_matches_notebook_contract() -> None:
    patient_frame = load_patient_covariate_frame(DATASET_PATH)
    summary = compute_patient_summary(patient_frame)

    assert set(summary) == {
        "patient_count",
        "average_sampled_points",
        "average_final_measurement_time",
        "average_age",
        "average_weight",
        "average_height",
        "average_bsa",
        "sex_counts",
    }
    assert summary["patient_count"] == 65
    assert summary["average_sampled_points"] > 0.0
    assert summary["average_final_measurement_time"] > 0.0
    assert summary["average_age"] > 0.0
    assert summary["average_weight"] > 0.0
    assert summary["average_height"] > 0.0
    assert summary["average_bsa"] > 0.0
    assert sum(summary["sex_counts"].values()) == 65
