"""Gradient tests for the remifentanil PBPK solver."""

import jax
import jax.numpy as jnp
import numpy as np

from pharmacokinetics import remifentanil

jax.config.update("jax_enable_x64", True)


def _patient_mse_loss(kinetic: jnp.ndarray, patient: remifentanil.PhysiologicalParameters) -> jnp.ndarray:
    """Return the masked mean-squared error for one patient."""
    _, concentrations = remifentanil.simulate_patient_separated(patient, kinetic)
    measured = patient.c_meas[patient.mask]
    predicted = concentrations[patient.mask]
    return jnp.mean((measured - predicted) ** 2)


class TestRemifentanilGradients:
    """Verify autodifferentiability of the active separated API."""

    def test_simulation_gradient_is_finite(self, remifentanil_physio_patients, remifentanil_default_params):
        """The plasma concentration trajectory should be differentiable in the kinetic parameters."""
        _, default_params = remifentanil_default_params
        patient = remifentanil_physio_patients[0]
        kinetic_vec = jnp.array(default_params)

        def summed_concentrations(kinetic):
            _, concentrations = remifentanil.simulate_patient_separated(patient, kinetic)
            return jnp.sum(concentrations)

        gradients = jax.grad(summed_concentrations)(kinetic_vec)

        assert gradients.shape == kinetic_vec.shape
        assert jnp.all(jnp.isfinite(gradients))
        assert not jnp.allclose(gradients, 0.0)

    def test_simulation_jacobian_shape(self, remifentanil_physio_patients, remifentanil_default_params):
        """The Jacobian should map kinetic parameters to measurement-grid concentrations."""
        _, default_params = remifentanil_default_params
        patient = remifentanil_physio_patients[0]
        kinetic_vec = jnp.array(default_params)

        def concentrations_only(kinetic):
            _, concentrations = remifentanil.simulate_patient_separated(patient, kinetic)
            return concentrations

        jacobian = jax.jacrev(concentrations_only)(kinetic_vec)

        assert jacobian.shape == (patient.t_meas.shape[0], kinetic_vec.shape[0])
        assert jnp.all(jnp.isfinite(jacobian))

    def test_patient_loss_gradient_is_finite(self, remifentanil_physio_patients, remifentanil_default_params):
        """The single-patient MSE loss should produce a finite parameter gradient."""
        _, default_params = remifentanil_default_params
        patient = remifentanil_physio_patients[0]
        kinetic_vec = jnp.array(default_params)

        loss_value, gradients = jax.value_and_grad(_patient_mse_loss)(kinetic_vec, patient)

        assert jnp.isfinite(loss_value)
        assert loss_value >= 0.0
        assert gradients.shape == kinetic_vec.shape
        assert jnp.all(jnp.isfinite(gradients))

    def test_vectorized_loss_gradient_is_finite(self, small_remifentanil_sample, remifentanil_default_params):
        """The batched separated API should support gradient-based fitting."""
        _, default_params = remifentanil_default_params
        _, physio_patients = small_remifentanil_sample
        kinetic_vec = jnp.array(default_params)

        def vectorized_loss(kinetic):
            predicted = remifentanil.vectorized_simulate_separated(physio_patients, kinetic)
            losses = []
            for index, patient in enumerate(physio_patients):
                measured = patient.c_meas[patient.mask]
                predicted_patient = predicted[index][patient.mask]
                losses.append(jnp.mean((measured - predicted_patient) ** 2))
            return jnp.mean(jnp.stack(losses))

        loss_value, gradients = jax.value_and_grad(vectorized_loss)(kinetic_vec)

        assert jnp.isfinite(loss_value)
        assert loss_value >= 0.0
        assert gradients.shape == kinetic_vec.shape
        assert jnp.all(jnp.isfinite(gradients))

    def test_gradient_matches_finite_difference_subset(self, remifentanil_physio_patients, remifentanil_default_params):
        """A subset of analytical gradients should agree with central finite differences."""
        _, default_params = remifentanil_default_params
        patient = remifentanil_physio_patients[0]
        kinetic_vec = jnp.array(default_params)
        analytical = jax.grad(_patient_mse_loss)(kinetic_vec, patient)

        eps = 1e-5
        indices = [0, 1, 4]
        finite_difference = []
        analytical_subset = []

        for index in indices:
            loss_plus = _patient_mse_loss(kinetic_vec.at[index].add(eps), patient)
            loss_minus = _patient_mse_loss(kinetic_vec.at[index].add(-eps), patient)
            finite_difference.append((loss_plus - loss_minus) / (2 * eps))
            analytical_subset.append(analytical[index])

        np.testing.assert_allclose(
            np.asarray(analytical_subset),
            np.asarray(finite_difference),
            rtol=1e-1,
            atol=1e-3,
        )
