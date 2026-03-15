"""Test cases for symbolic regression functionality."""

import pytest
import jax
import jax.numpy as jnp
import tempfile
from pharmacokinetics import remifentanil

# # Only import PySR if available
# pysr = pytest.importorskip("pysr", reason="PySR not available")


# class TestSymbolicRegression:
#     """Test symbolic regression with PySR."""

#     @pytest.fixture
#     def synthetic_data(self, small_remifentanil_sample):
#         """Generate synthetic data for symbolic regression testing."""
#         jax.config.update("jax_enable_x64", True)

#         raw_patients, patient_params = small_remifentanil_sample

#         # Get patient covariates
#         covariates = []
#         for p in patient_params:
#             covariates.append(jnp.array([p.age, p.weight, p.height, p.bsa, p.dose_rate, p.dose_duration]))
#         covariates = jnp.stack(covariates)

#         # Get default parameters for ranges
#         param_names, init_kin = remifentanil.get_default_parameters()

#         # Create synthetic dataset for symbolic regression
#         rng = jax.random.PRNGKey(42)
#         n_samples = 20  # Small sample for testing

#         # Parameter ranges for synthesis
#         param_ranges = {
#             "k_TP": (init_kin[0] * 0.5, init_kin[0] * 2.0),
#             "k_PT": (init_kin[1] * 0.5, init_kin[1] * 2.0),
#             "k_EL_Pl": (init_kin[4] * 0.5, init_kin[4] * 2.0),
#         }

#         # Generate synthetic parameters
#         sampled_params = []
#         patient_indices = jax.random.randint(rng, (n_samples,), 0, len(covariates))

#         for _ in range(n_samples):
#             rng, subkey = jax.random.split(rng)
#             params = {}
#             for param_name, (low, high) in param_ranges.items():
#                 params[param_name] = jax.random.uniform(subkey, (), minval=low, maxval=high)
#                 rng, subkey = jax.random.split(rng)
#             sampled_params.append(params)

#         # Convert to arrays
#         param_matrix = jnp.array([[params["k_TP"], params["k_PT"], params["k_EL_Pl"]] for params in sampled_params])

#         covariate_matrix = covariates[patient_indices]
#         variable_names = ["age", "weight", "height", "bsa", "dose_rate", "dose_duration"]

#         return covariate_matrix, param_matrix, variable_names, param_names

#     def test_pysr_basic_functionality(self, synthetic_data):
#         """Test basic PySR functionality with minimal settings."""
#         covariate_matrix, param_matrix, variable_names, param_names = synthetic_data

#         with tempfile.TemporaryDirectory() as tempdir:
#             # Create a minimal PySR regressor for testing
#             symbolic_regressor = pysr.PySRRegressor(
#                 niterations=2,  # Minimal for testing
#                 ncycles_per_iteration=50,
#                 populations=1,
#                 population_size=15,
#                 binary_operators=["+", "*"],
#                 unary_operators=["square"],
#                 maxsize=7,  # Minimum required by PySR
#                 maxdepth=2,
#                 model_selection="best",
#                 progress=False,  # Disable progress for clean test output
#                 tempdir=tempdir,
#                 temp_equation_file=True,
#                 tournament_selection_n=10,
#             )

#             # Test fitting just one parameter (k_TP)
#             symbolic_regressor.fit(covariate_matrix, param_matrix[:, 0], variable_names=variable_names)

#             # Verify that equations were generated
#             assert hasattr(symbolic_regressor, "equations_")
#             assert len(symbolic_regressor.equations_) > 0

#             # Test prediction capability
#             predictions = symbolic_regressor.predict(covariate_matrix)
#             assert predictions.shape == (len(covariate_matrix),)
#             assert jnp.all(jnp.isfinite(predictions))

#     def test_pysr_multiple_parameters(self, synthetic_data):
#         """Test symbolic regression for multiple parameters."""
#         covariate_matrix, param_matrix, variable_names, param_names = synthetic_data

#         # Test that we can create separate regressors for different parameters
#         regressors = {}

#         with tempfile.TemporaryDirectory() as tempdir:
#             for i, param_name in enumerate(["k_TP", "k_PT", "k_EL_Pl"]):
#                 regressor = pysr.PySRRegressor(
#                     niterations=1,  # Very minimal for testing
#                     ncycles_per_iteration=25,
#                     populations=1,
#                     population_size=15,  # Increased to avoid tournament selection issue
#                     binary_operators=["+", "*"],
#                     maxsize=7,  # Minimum required by PySR
#                     maxdepth=2,
#                     model_selection="best",
#                     progress=False,
#                     tempdir=tempdir,
#                     temp_equation_file=True,
#                     tournament_selection_n=10,  # Smaller than population_size
#                 )

#                 # Fit the regressor
#                 regressor.fit(covariate_matrix, param_matrix[:, i], variable_names=variable_names)
#                 regressors[param_name] = regressor

#                 # Test basic functionality
#                 predictions = regressor.predict(covariate_matrix)
#                 assert predictions.shape == (len(covariate_matrix),)
#                 assert jnp.all(jnp.isfinite(predictions))

#         # Verify we created regressors for all parameters
#         assert len(regressors) == 3
#         assert all(param in regressors for param in ["k_TP", "k_PT", "k_EL_Pl"])

#     def test_pysr_custom_operators(self, synthetic_data):
#         """Test PySR with custom operators."""
#         covariate_matrix, param_matrix, variable_names, param_names = synthetic_data

#         with tempfile.TemporaryDirectory() as tempdir:
#             # Test with different operator combinations
#             symbolic_regressor = pysr.PySRRegressor(
#                 niterations=1,
#                 ncycles_per_iteration=25,
#                 populations=1,
#                 population_size=15,  # Increased to avoid tournament selection issue
#                 binary_operators=["+", "-", "*", "/"],
#                 unary_operators=["square", "sqrt", "log"],
#                 maxsize=7,  # Minimum required by PySR
#                 maxdepth=2,
#                 model_selection="best",
#                 progress=False,
#                 tempdir=tempdir,
#                 temp_equation_file=True,
#                 tournament_selection_n=10,  # Smaller than population_size
#             )

#             # Test fitting
#             symbolic_regressor.fit(covariate_matrix, param_matrix[:, 0], variable_names=variable_names)

#             # Verify functionality
#             assert hasattr(symbolic_regressor, "equations_")
#             predictions = symbolic_regressor.predict(covariate_matrix)
#             assert predictions.shape == (len(covariate_matrix),)

#     def test_symbolic_regression_integration(self, remifentanil_default_params):
#         """Test integration with remifentanil module."""
#         param_names, default_params = remifentanil_default_params

#         # Verify that we have the expected parameter names for symbolic regression
#         expected_params = ["k_TP", "k_PT", "k_PHP", "k_HPP", "k_EL_Pl", "Eff_kid", "Eff_hep", "k_EL_Tis"]
#         assert param_names == expected_params

#         # Verify parameter indices for the ones we typically use in symbolic regression
#         assert param_names[0] == "k_TP"  # Index 0
#         assert param_names[1] == "k_PT"  # Index 1
#         assert param_names[4] == "k_EL_Pl"  # Index 4

#         # Verify parameters are positive and finite
#         assert jnp.all(default_params > 0)
#         assert jnp.all(jnp.isfinite(default_params))

#     @pytest.mark.slow
#     def test_symbolic_regression_longer_run(self, synthetic_data):
#         """Test symbolic regression with more iterations (marked as slow)."""
#         covariate_matrix, param_matrix, variable_names, param_names = synthetic_data

#         with tempfile.TemporaryDirectory() as tempdir:
#             symbolic_regressor = pysr.PySRRegressor(
#                 niterations=5,  # More iterations for better results
#                 ncycles_per_iteration=100,
#                 populations=2,
#                 population_size=20,
#                 binary_operators=["+", "*", "-", "/"],
#                 unary_operators=["square", "sqrt"],
#                 maxsize=8,
#                 maxdepth=3,
#                 model_selection="best",
#                 progress=False,
#                 tempdir=tempdir,
#                 temp_equation_file=True,
#                 tournament_selection_n=10,
#             )

#             # Test with k_TP parameter
#             symbolic_regressor.fit(covariate_matrix, param_matrix[:, 0], variable_names=variable_names)

#             # Verify we get reasonable results
#             predictions = symbolic_regressor.predict(covariate_matrix)

#             # Check that predictions are in a reasonable range
#             actual_values = param_matrix[:, 0]
#             relative_error = jnp.abs(predictions - actual_values) / actual_values

#             # At least some predictions should be reasonable (within 50% relative error)
#             reasonable_predictions = jnp.sum(relative_error < 0.5)
#             assert reasonable_predictions > 0, "No reasonable predictions found"
