"""Loss-function tests for remifentanil parameter-estimation workflows."""

import jax
import jax.numpy as jnp
import numpy as np

from pharmacokinetics import remifentanil

jax.config.update("jax_enable_x64", True)


def _rmse_loss(kinetic: jnp.ndarray, physio_patients) -> jnp.ndarray:
    """Return the cohort RMSE used by several remifentanil fitting scripts."""
    predicted = remifentanil.vectorized_simulate_separated(physio_patients, kinetic)
    residual_sums = []

    for index, patient in enumerate(physio_patients):
        measured = patient.c_meas[patient.mask]
        predicted_patient = predicted[index][patient.mask]
        residual_sums.append((measured - predicted_patient) ** 2)

    squared_errors = jnp.concatenate(residual_sums)
    return jnp.sqrt(jnp.mean(squared_errors))


def _weighted_nll_loss(kinetic: jnp.ndarray, physio_patients, scale: float = 0.1) -> jnp.ndarray:
    """Return a weighted negative log-likelihood style objective."""
    predicted = remifentanil.vectorized_simulate_separated(physio_patients, kinetic)
    total_log_likelihood = 0.0

    for index, patient in enumerate(physio_patients):
        measured = patient.c_meas[patient.mask]
        predicted_patient = predicted[index][patient.mask]
        residuals = predicted_patient - measured
        variances = jnp.clip((scale * measured) ** 2, 1e-8, None)
        total_log_likelihood += jnp.sum(
            -0.5 * (jnp.log(2 * jnp.pi * variances) + (residuals**2) / variances)
        )

    return -total_log_likelihood


class TestRemifentanilLossFunctions:
    """Test losses used by Bayesian optimization and gradient-based estimation."""

    def test_rmse_loss_gradient(self, small_remifentanil_sample, remifentanil_default_params):
        """The cohort RMSE loss should be finite and differentiable."""
        _, default_params = remifentanil_default_params
        _, physio_patients = small_remifentanil_sample
        kinetic_vec = jnp.array(default_params)

        loss_value, gradients = jax.value_and_grad(_rmse_loss)(kinetic_vec, physio_patients)

        assert jnp.isfinite(loss_value)
        assert loss_value >= 0.0
        assert gradients.shape == kinetic_vec.shape
        assert jnp.all(jnp.isfinite(gradients))

    def test_weighted_nll_gradient(self, small_remifentanil_sample, remifentanil_default_params):
        """The weighted negative log-likelihood objective should be finite and differentiable."""
        _, default_params = remifentanil_default_params
        _, physio_patients = small_remifentanil_sample
        kinetic_vec = jnp.array(default_params)

        loss_value, gradients = jax.value_and_grad(_weighted_nll_loss)(kinetic_vec, physio_patients)

        assert jnp.isfinite(loss_value)
        assert gradients.shape == kinetic_vec.shape
        assert jnp.all(jnp.isfinite(gradients))

    def test_batched_rmse_matches_weighted_individual_loss(self, small_remifentanil_sample, remifentanil_default_params):
        """The vectorized RMSE should equal the point-count-weighted cohort MSE."""
        _, default_params = remifentanil_default_params
        _, physio_patients = small_remifentanil_sample
        kinetic_vec = jnp.array(default_params)
        predicted = remifentanil.vectorized_simulate_separated(physio_patients, kinetic_vec)

        individual_mse = []
        individual_counts = []
        for index, patient in enumerate(physio_patients):
            measured = patient.c_meas[patient.mask]
            predicted_patient = predicted[index][patient.mask]
            individual_mse.append(jnp.mean((measured - predicted_patient) ** 2))
            individual_counts.append(measured.shape[0])

        rmse_squared = _rmse_loss(kinetic_vec, physio_patients) ** 2
        weighted_mean_mse = jnp.average(
            jnp.stack(individual_mse),
            weights=jnp.asarray(individual_counts, dtype=jnp.float64),
        )

        np.testing.assert_allclose(
            np.asarray(rmse_squared),
            np.asarray(weighted_mean_mse),
            rtol=1e-6,
            atol=1e-8,
        )
