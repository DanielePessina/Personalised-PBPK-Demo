"""Tests for the active remifentanil PBPK module."""

import jax
import jax.numpy as jnp
import numpy as np

from pharmacokinetics import remifentanil


class TestRemifentanilParameters:
    """Test parameter and constant definitions."""

    def test_get_default_parameters(self, remifentanil_default_params):
        """Default parameter names should match the published constant ordering."""
        param_names, default_params = remifentanil_default_params

        assert param_names == list(remifentanil.KINETIC_PARAMETER_NAMES)
        assert isinstance(default_params, jnp.ndarray)
        assert default_params.shape == (8,)
        assert jnp.all(default_params > 0)
        assert jnp.all(jnp.isfinite(default_params))

    def test_get_abbiati_parameters(self):
        """The Abbiati reference parameters should share the same ordering."""
        param_names, abbiati_params = remifentanil.get_abbiati_parameters()

        assert param_names == list(remifentanil.KINETIC_PARAMETER_NAMES)
        assert abbiati_params.shape == (8,)
        assert jnp.all(abbiati_params > 0)


class TestRemifentanilPatients:
    """Test dataset loading and physiological parameter construction."""

    def test_import_patients(self, remifentanil_raw_patients):
        """The canonical dataset should load into RawPatient objects."""
        patient = remifentanil_raw_patients[0]

        assert len(remifentanil_raw_patients) > 0
        assert isinstance(patient, remifentanil.RawPatient)
        assert isinstance(patient.id, int)
        assert isinstance(patient.age, float)
        assert isinstance(patient.weight, float)
        assert isinstance(patient.height, float)
        assert isinstance(patient.sex, bool)
        assert isinstance(patient.dose_rate, float)
        assert isinstance(patient.dose_duration, float)
        assert patient.t_meas.shape == patient.c_meas.shape
        assert patient.t_meas.shape == patient.mask.shape
        assert patient.mask.dtype == jnp.bool_

    def test_create_physiological_parameters(self, remifentanil_patients):
        """Physiological parameter creation should produce positive organ flows and volumes."""
        raw_patients, physio_patients = remifentanil_patients
        raw_patient = raw_patients[0]
        physio_patient = physio_patients[0]

        assert isinstance(physio_patient, remifentanil.PhysiologicalParameters)
        assert physio_patient.id == raw_patient.id
        assert physio_patient.q_HA > 0
        assert physio_patient.q_PV > 0
        assert physio_patient.q_K > 0
        assert physio_patient.v_P > 0
        assert physio_patient.v_L > 0
        assert physio_patient.v_HP > 0
        assert physio_patient.to_nlme_covariates().shape == (6,)


class TestRemifentanilSimulation:
    """Test the separated simulation entrypoints."""

    def test_simulate_patient_separated(self, remifentanil_physio_patients, remifentanil_default_params):
        """Single-patient simulation should return finite concentrations on the measurement grid."""
        _, default_params = remifentanil_default_params
        patient = remifentanil_physio_patients[0]

        times, concentrations = remifentanil.simulate_patient_separated(patient, default_params)

        assert times.shape == concentrations.shape
        assert times.shape == patient.t_meas.shape
        assert jnp.all(jnp.isfinite(times))
        assert jnp.all(jnp.isfinite(concentrations))
        assert jnp.all(concentrations >= 0)
        assert jnp.all(times[patient.mask][1:] >= times[patient.mask][:-1])
        np.testing.assert_allclose(times, patient.t_meas, rtol=0, atol=0)

    def test_vectorized_simulate_separated(self, small_remifentanil_sample, remifentanil_default_params):
        """Vectorized simulation should agree with the individual solver outputs."""
        _, default_params = remifentanil_default_params
        _, physio_patients = small_remifentanil_sample

        batch_concentrations = remifentanil.vectorized_simulate_separated(physio_patients, default_params)

        assert batch_concentrations.shape[0] == len(physio_patients)
        assert batch_concentrations.shape[1] == physio_patients[0].t_meas.shape[0]
        assert jnp.all(jnp.isfinite(batch_concentrations))
        assert jnp.all(batch_concentrations >= 0)

        for index, patient in enumerate(physio_patients):
            _, concentrations = remifentanil.simulate_patient_separated(patient, default_params)
            np.testing.assert_allclose(batch_concentrations[index], concentrations, rtol=1e-8, atol=1e-10)

    def test_simulate_patient_dense(self, remifentanil_physio_patients, remifentanil_default_params):
        """Dense simulation should honor the requested time grid."""
        _, default_params = remifentanil_default_params
        patient = remifentanil_physio_patients[0]
        valid_end = float(patient.t_meas[patient.mask][-1])
        dense_grid = jnp.linspace(0.0, valid_end, 25)

        times, concentrations = remifentanil.simulate_patient_dense(patient, default_params, dense_grid)

        np.testing.assert_allclose(times, dense_grid)
        assert concentrations.shape == dense_grid.shape
        assert jnp.all(jnp.isfinite(concentrations))
        assert jnp.all(concentrations >= 0)


class TestRemifentanilBatching:
    """Test helper utilities used by hybrid and NLME workflows."""

    def test_create_patient_batches(self, remifentanil_physio_patients):
        """Batch construction should preserve patient-count and covariate alignment."""
        physio_patients = remifentanil_physio_patients[:8]
        covariates = jnp.stack([patient.to_nlme_covariates() for patient in physio_patients])

        batched_physio, batched_covariates = remifentanil.create_patient_batches(
            physio_patients,
            covariates,
            batch_size=3,
            key=jax.random.PRNGKey(0),
        )

        assert len(batched_physio) == len(batched_covariates)
        assert sum(len(batch) for batch in batched_physio) == len(physio_patients)
        assert all(batch_cov.shape[1] == covariates.shape[1] for batch_cov in batched_covariates)
        assert all(len(batch) <= 3 for batch in batched_physio)

    def test_get_random_patient_batch(self, remifentanil_physio_patients):
        """Random batching should return aligned patient and covariate subsets."""
        physio_patients = remifentanil_physio_patients[:8]
        covariates = jnp.stack([patient.to_nlme_covariates() for patient in physio_patients])

        batch_physio, batch_covariates = remifentanil.get_random_patient_batch(
            physio_patients,
            covariates,
            batch_size=4,
            key=jax.random.PRNGKey(42),
        )

        assert len(batch_physio) == 4
        assert batch_covariates.shape == (4, covariates.shape[1])


class TestRemifentanilAPI:
    """Guard the exported names used by the maintained codebase."""

    def test_module_all_contains_current_symbols(self):
        """The module export list should advertise the current separated API."""
        expected_exports = {
            "RawPatient",
            "PhysiologicalParameters",
            "import_patients",
            "import_patients_from_nmexcel",
            "create_physiological_parameters",
            "simulate_patient_separated",
            "vectorized_simulate_separated",
            "simulate_patient_dense",
            "vectorized_simulate_dense",
            "get_default_parameters",
            "get_abbiati_parameters",
            "create_patient_batches",
            "get_random_patient_batch",
        }

        assert expected_exports.issubset(set(remifentanil.__all__))
