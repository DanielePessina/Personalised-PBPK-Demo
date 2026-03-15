"""Common pytest fixtures for the remifentanil-focused test suite."""

from pathlib import Path

import pytest

from pharmacokinetics import remifentanil

REPO_ROOT = Path(__file__).resolve().parents[2]
REMIFENTANIL_DATASET = REPO_ROOT / "nlme-remifentanil.xlsx"


@pytest.fixture(scope="session")
def remifentanil_default_params():
    """Return the default remifentanil kinetic parameter names and values."""
    return remifentanil.get_default_parameters()


@pytest.fixture(scope="session")
def remifentanil_dataset_path() -> Path:
    """Return the canonical remifentanil dataset path used in tests."""
    if not REMIFENTANIL_DATASET.exists():
        pytest.skip("nlme-remifentanil.xlsx not found")
    return REMIFENTANIL_DATASET


@pytest.fixture(scope="session")
def remifentanil_raw_patients(remifentanil_dataset_path):
    """Load raw remifentanil patients from the canonical workbook."""
    return remifentanil.import_patients(str(remifentanil_dataset_path))


@pytest.fixture(scope="session")
def remifentanil_physio_patients(remifentanil_raw_patients):
    """Convert raw remifentanil patients into physiological parameter objects."""
    return [remifentanil.create_physiological_parameters(p) for p in remifentanil_raw_patients]


@pytest.fixture(scope="session")
def remifentanil_patients(remifentanil_raw_patients, remifentanil_physio_patients):
    """Return both raw and physiological remifentanil patient representations."""
    return remifentanil_raw_patients, remifentanil_physio_patients


@pytest.fixture
def small_remifentanil_sample(remifentanil_raw_patients, remifentanil_physio_patients):
    """Return a small remifentanil sample for tests that do not need the full cohort."""
    n_patients = min(3, len(remifentanil_raw_patients))
    return remifentanil_raw_patients[:n_patients], remifentanil_physio_patients[:n_patients]
