"""Regression tests for hybrid-model covariate preprocessing."""

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.preprocessing import MinMaxScaler

jax.config.update("jax_enable_x64", True)


def _split_train_test(physio_patients):
    """Deterministically split patients into train and test cohorts."""
    indices = jax.random.permutation(jax.random.PRNGKey(333), len(physio_patients))
    split_index = int(0.8 * len(physio_patients))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    train_patients = [physio_patients[int(index)] for index in train_indices]
    test_patients = [physio_patients[int(index)] for index in test_indices]
    return train_patients, test_patients


def _covariates_with_bsa(physio_patients):
    """Mirror the covariate surface used in ``Remi_HybridModel.py``."""
    return jnp.stack(
        [
            jnp.array(
                [patient.age, patient.weight, patient.height, patient.bsa, patient.dose_rate, patient.dose_duration]
            )
            for patient in physio_patients
        ]
    )


def _covariates_with_sex(physio_patients):
    """Mirror the covariate surface used in ``Remi_HybridModel_OptunaTuning.py``."""
    return jnp.stack(
        [
            jnp.array(
                [patient.age, patient.weight, patient.height, patient.sex, patient.dose_rate, patient.dose_duration]
            )
            for patient in physio_patients
        ]
    )


def _scale(train_covariates, test_covariates):
    """Fit MinMax scaling on the training cohort and apply it to both splits."""
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_scaled = jnp.array(scaler.fit_transform(np.asarray(train_covariates)))
    test_scaled = jnp.array(scaler.transform(np.asarray(test_covariates)))
    return train_scaled, test_scaled


def test_bsa_covariates_scale_to_expected_range(remifentanil_physio_patients):
    """BSA-based hybrid covariates should scale to ``[-1, 1]`` on the training set."""
    train_patients, test_patients = _split_train_test(remifentanil_physio_patients)
    train_covariates = _covariates_with_bsa(train_patients)
    test_covariates = _covariates_with_bsa(test_patients)
    train_scaled, test_scaled = _scale(train_covariates, test_covariates)

    assert train_scaled.shape == train_covariates.shape
    assert test_scaled.shape == test_covariates.shape
    np.testing.assert_allclose(np.asarray(jnp.min(train_scaled, axis=0)), -1.0, rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.asarray(jnp.max(train_scaled, axis=0)), 1.0, rtol=0, atol=1e-10)
    assert jnp.all(jnp.isfinite(test_scaled))


def test_sex_covariates_scale_to_expected_range(remifentanil_physio_patients):
    """Sex-based hybrid covariates should scale to ``[-1, 1]`` on the training set."""
    train_patients, test_patients = _split_train_test(remifentanil_physio_patients)
    train_covariates = _covariates_with_sex(train_patients)
    test_covariates = _covariates_with_sex(test_patients)
    train_scaled, test_scaled = _scale(train_covariates, test_covariates)

    assert train_scaled.shape == train_covariates.shape
    assert test_scaled.shape == test_covariates.shape
    np.testing.assert_allclose(np.asarray(jnp.min(train_scaled, axis=0)), -1.0, rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.asarray(jnp.max(train_scaled, axis=0)), 1.0, rtol=0, atol=1e-10)
    assert jnp.all(jnp.isfinite(test_scaled))


def test_stop_gradient_blocks_backpropagation(remifentanil_physio_patients):
    """The scaled covariates should remain detached when explicitly wrapped in ``stop_gradient``."""
    train_patients, _ = _split_train_test(remifentanil_physio_patients)
    train_covariates = _covariates_with_bsa(train_patients)
    train_scaled, _ = _scale(train_covariates, train_covariates)
    sample = train_scaled[:5]

    def loss_fn(x):
        return jnp.sum(jax.lax.stop_gradient(x) ** 2)

    gradients = jax.grad(loss_fn)(sample)
    np.testing.assert_allclose(np.asarray(gradients), 0.0, rtol=0, atol=1e-12)
