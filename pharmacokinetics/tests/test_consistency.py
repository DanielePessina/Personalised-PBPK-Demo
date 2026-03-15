"""API regression tests for the remifentanil-focused package surface."""

import pharmacokinetics
from pharmacokinetics import remifentanil


def test_package_exports_remifentanil_only():
    """The package surface should expose only the active remifentanil modules."""
    assert pharmacokinetics.__all__ == ["remifentanil", "remifentanil_node", "nlme"]
    assert len(pharmacokinetics.__all__) == 3


def test_current_remifentanil_public_api():
    """The maintained remifentanil API should use separated physiological parameters."""
    required_functions = [
        "import_patients",
        "import_patients_from_nmexcel",
        "create_physiological_parameters",
        "simulate_patient_separated",
        "vectorized_simulate_separated",
        "simulate_patient_dense",
        "vectorized_simulate_dense",
        "create_patient_batches",
        "get_random_patient_batch",
        "get_default_parameters",
        "get_abbiati_parameters",
    ]

    for func_name in required_functions:
        assert hasattr(remifentanil, func_name), f"remifentanil missing {func_name}"
        assert callable(getattr(remifentanil, func_name))


def test_physiological_parameters_keep_measurement_surface(remifentanil_patients):
    """Raw-to-physiological conversion should preserve the measurement arrays."""
    raw_patients, physio_patients = remifentanil_patients
    raw_patient = raw_patients[0]
    physio_patient = physio_patients[0]

    assert physio_patient.id == raw_patient.id
    assert physio_patient.t_meas.shape == raw_patient.t_meas.shape
    assert physio_patient.c_meas.shape == raw_patient.c_meas.shape
    assert physio_patient.mask.shape == raw_patient.mask.shape
    assert physio_patient.to_nlme_covariates().shape == (6,)
