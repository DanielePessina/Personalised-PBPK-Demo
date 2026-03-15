"""Tests for remifentanil_node data processing functions."""

import pytest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.preprocessing import MinMaxScaler

from pharmacokinetics import remifentanil_node
from pharmacokinetics.remifentanil import RawPatient


@pytest.fixture
def mock_raw_patients():
    """Create mock RawPatient objects for testing."""
    # Patient 1: Simple case with 3 measurements
    patient1 = RawPatient(
        id=1,
        age=30.0,
        weight=70.0,
        height=175.0,
        sex=False,  # Male
        bsa=1.8,  # Body surface area
        dose_rate=1.0,
        dose_duration=60.0,
        t_meas=jnp.array([0.0, 30.0, 60.0, 60.0, 60.0]),  # Padded with zeros
        c_meas=jnp.array([0.0, 5.0, 2.5, 0.0, 0.0]),    # Padded with zeros
        mask=jnp.array([True, True, True, False, False])
    )

    # Patient 2: Different case with 2 measurements
    patient2 = RawPatient(
        id=2,
        age=45.0,
        weight=80.0,
        height=180.0,
        sex=True,   # Female
        bsa=2.0,  # Body surface area
        dose_rate=1.5,
        dose_duration=90.0,
        t_meas=jnp.array([0.0, 45.0, 45.0, 45.0, 45.0]),   # Padded with zeros
        c_meas=jnp.array([0.0, 7.0, 0.0, 0.0, 0.0]),    # Padded with zeros
        mask=jnp.array([True, True, False, False, False])
    )

    return [patient1, patient2]


@pytest.fixture
def empty_patients_list():
    """Empty list of patients for error testing."""
    return []


class TestPrepareDataset:
    """Test the prepare_dataset function."""

    def test_prepare_dataset_basic_functionality(self, mock_raw_patients):
        """Test basic functionality of prepare_dataset."""
        processed_data, patient_ids, scalers = remifentanil_node.prepare_dataset(mock_raw_patients)

        # Check return types and lengths
        assert isinstance(processed_data, list)
        assert len(processed_data) == 2
        assert isinstance(patient_ids, jnp.ndarray)
        assert len(patient_ids) == 2
        assert isinstance(scalers, dict)

        # Check patient IDs
        assert patient_ids[0] == 1
        assert patient_ids[1] == 2

        # Check scalers
        assert "concentration_scaler" in scalers
        assert "global_max_time" in scalers
        assert isinstance(scalers["concentration_scaler"], MinMaxScaler)
        assert scalers["global_max_time"] == 60.0  # Max time from patient 1

    def test_prepare_dataset_data_structure(self, mock_raw_patients):
        """Test the structure of processed data objects."""
        processed_data, _, _ = remifentanil_node.prepare_dataset(mock_raw_patients)

        node_data = processed_data[0]
        assert isinstance(node_data, remifentanil_node.RemiNODEData)

        # Check all required attributes exist
        assert hasattr(node_data, 't_meas_scaled')
        assert hasattr(node_data, 'c_meas_scaled')
        assert hasattr(node_data, 'measurement_mask')
        assert hasattr(node_data, 'static_augmentations')
        assert hasattr(node_data, 'y0')

        # Check shapes are consistent (max_len should be 3 from patient 1)
        max_len = 3
        assert node_data.t_meas_scaled.shape == (max_len,)
        assert node_data.c_meas_scaled.shape == (max_len,)
        assert node_data.measurement_mask.shape == (max_len,)
        assert node_data.static_augmentations.shape == (6,)  # 6 static features
        assert node_data.y0.shape == (1,)

    def test_prepare_dataset_scaling(self, mock_raw_patients):
        """Test that scaling is applied correctly."""
        processed_data, _, scalers = remifentanil_node.prepare_dataset(mock_raw_patients)

        # Check time scaling
        node_data = processed_data[0]  # Patient 1
        expected_t_scaled = jnp.array([0.0, 30.0, 60.0]) / 60.0  # Scaled by global_max_time
        np.testing.assert_array_almost_equal(
            node_data.t_meas_scaled[:3], expected_t_scaled, decimal=6
        )

        # Check concentration scaling is in range [-1, 1]
        valid_mask = node_data.measurement_mask
        valid_concentrations = node_data.c_meas_scaled[valid_mask]
        assert jnp.all(valid_concentrations >= -1.0)
        assert jnp.all(valid_concentrations <= 1.0)

    def test_prepare_dataset_masking(self, mock_raw_patients):
        """Test that masking is applied correctly."""
        processed_data, _, _ = remifentanil_node.prepare_dataset(mock_raw_patients)

        # Patient 1 should have 3 valid measurements
        node_data_1 = processed_data[0]
        assert jnp.sum(node_data_1.measurement_mask) == 3

        # Patient 2 should have 2 valid measurements
        node_data_2 = processed_data[1]
        assert jnp.sum(node_data_2.measurement_mask) == 2

    def test_prepare_dataset_static_features(self, mock_raw_patients):
        """Test static features extraction."""
        processed_data, _, _ = remifentanil_node.prepare_dataset(mock_raw_patients)

        node_data_1 = processed_data[0]
        # Check that static features match patient 1 attributes
        expected_features = jnp.array([30.0, 70.0, 175.0, 0.0, 1.0, 60.0])  # sex converted to float
        np.testing.assert_array_almost_equal(
            node_data_1.static_augmentations, expected_features, decimal=6
        )

    def test_prepare_dataset_with_existing_scalers(self, mock_raw_patients):
        """Test using pre-fitted scalers."""
        # First call to fit scalers
        _, _, scalers_original = remifentanil_node.prepare_dataset(mock_raw_patients)

        # Second call with existing scalers
        processed_data_2, _, scalers_reused = remifentanil_node.prepare_dataset(
            mock_raw_patients, scalers=scalers_original
        )

        # Scalers should be the same object
        assert scalers_reused is scalers_original
        assert scalers_reused["global_max_time"] == scalers_original["global_max_time"]

    def test_prepare_dataset_empty_input(self, empty_patients_list):
        """Test error handling for empty input."""
        with pytest.raises(ValueError, match="Input `raw_patients` list cannot be empty"):
            remifentanil_node.prepare_dataset(empty_patients_list)

    def test_prepare_dataset_custom_static_features(self, mock_raw_patients):
        """Test with custom static feature names."""
        custom_features = ["age", "weight", "sex"]
        processed_data, _, _ = remifentanil_node.prepare_dataset(
            mock_raw_patients, static_feature_names=custom_features
        )

        node_data = processed_data[0]
        assert node_data.static_augmentations.shape == (3,)

        # Check values match the custom features
        expected_features = jnp.array([30.0, 70.0, 0.0])  # age, weight, sex (male=0)
        np.testing.assert_array_almost_equal(
            node_data.static_augmentations, expected_features, decimal=6
        )


class TestUnscalePredictions:
    """Test the unscale_predictions function."""

    def test_unscale_predictions_basic(self, mock_raw_patients):
        """Test basic unscaling functionality."""
        _, _, scalers = remifentanil_node.prepare_dataset(mock_raw_patients)

        # Test with some scaled predictions in range [-1, 1]
        scaled_predictions = jnp.array([-1.0, 0.0, 1.0])
        unscaled = remifentanil_node.unscale_predictions(scaled_predictions, scalers)

        # Should be a JAX array
        assert isinstance(unscaled, jnp.ndarray)
        assert unscaled.shape == (3,)

        # Check that 0.0 scaled maps to a concentration that makes sense
        # (depends on the scaler fit, but should be reasonable)
        assert jnp.all(unscaled >= 0.0)  # Concentrations should be non-negative

    def test_unscale_predictions_single_value(self, mock_raw_patients):
        """Test unscaling a single value."""
        _, _, scalers = remifentanil_node.prepare_dataset(mock_raw_patients)

        scaled_pred = jnp.array([0.5])
        unscaled = remifentanil_node.unscale_predictions(scaled_pred, scalers)

        assert unscaled.shape == (1,)
        assert jnp.isfinite(unscaled[0])


class TestUnscaleData:
    """Test the unscale_data function."""

    def test_unscale_data_basic(self, mock_raw_patients):
        """Test basic data unscaling functionality."""
        processed_data, _, scalers = remifentanil_node.prepare_dataset(mock_raw_patients)

        node_data = processed_data[0]  # Patient 1
        unscaled_dict = remifentanil_node.unscale_data(node_data, scalers)

        # Check return structure
        assert isinstance(unscaled_dict, dict)
        assert "time" in unscaled_dict
        assert "concentration" in unscaled_dict

        # Check that we get the right number of points (3 for patient 1)
        assert len(unscaled_dict["time"]) == 3
        assert len(unscaled_dict["concentration"]) == 3

        # Check time values are reasonable
        times = unscaled_dict["time"]
        assert jnp.all(times >= 0.0)
        assert times[0] == 0.0  # First measurement should be at time 0

    def test_unscale_data_values_match_original(self, mock_raw_patients):
        """Test that unscaled data matches original measurements."""
        processed_data, _, scalers = remifentanil_node.prepare_dataset(mock_raw_patients)

        # Get original data for patient 1
        original_patient = mock_raw_patients[0]
        original_mask = original_patient.mask
        original_times = original_patient.t_meas[original_mask]
        original_concentrations = original_patient.c_meas[original_mask]

        # Unscale processed data
        node_data = processed_data[0]
        unscaled_dict = remifentanil_node.unscale_data(node_data, scalers)

        # Compare (should be very close due to numerical precision)
        np.testing.assert_allclose(
            unscaled_dict["time"], original_times, rtol=1e-10
        )
        np.testing.assert_allclose(
            unscaled_dict["concentration"], original_concentrations, rtol=1e-5
        )


class TestRemiNODEData:
    """Test the RemiNODEData class."""

    def test_remi_node_data_creation(self):
        """Test creating a RemiNODEData object."""
        node_data = remifentanil_node.RemiNODEData(
            t_meas_scaled=jnp.array([0.0, 0.5, 1.0]),
            c_meas_scaled=jnp.array([-1.0, 0.0, 1.0]),
            measurement_mask=jnp.array([True, True, True]),
            static_augmentations=jnp.array([30.0, 70.0, 175.0, 0.0, 1.0, 60.0]),
            y0=jnp.array([-1.0])
        )

        # Check all attributes are accessible
        assert node_data.t_meas_scaled.shape == (3,)
        assert node_data.c_meas_scaled.shape == (3,)
        assert node_data.measurement_mask.shape == (3,)
        assert node_data.static_augmentations.shape == (6,)
        assert node_data.y0.shape == (1,)

    def test_remi_node_data_jax_compatibility(self):
        """Test that RemiNODEData is JAX-compatible."""
        node_data = remifentanil_node.RemiNODEData(
            t_meas_scaled=jnp.array([0.0, 0.5, 1.0]),
            c_meas_scaled=jnp.array([-1.0, 0.0, 1.0]),
            measurement_mask=jnp.array([True, True, True]),
            static_augmentations=jnp.array([30.0, 70.0, 175.0, 0.0, 1.0, 60.0]),
            y0=jnp.array([-1.0])
        )

        # Should be able to use with JAX tree operations
        tree_leaves = jax.tree_util.tree_leaves(node_data)
        assert len(tree_leaves) == 5  # 5 attributes

        # Should be able to stack multiple instances
        stacked = jax.tree_util.tree_map(lambda *x: jnp.stack(x), node_data, node_data)
        assert stacked.t_meas_scaled.shape == (2, 3)


def _build_augmented_state(
    *,
    augment_values: jnp.ndarray | None = None,
    dose_duration: float = 60.0,
) -> jnp.ndarray:
    """Construct an augmented NODE state with the default static feature layout."""
    augment_values = jnp.array([0.1, -0.2]) if augment_values is None else augment_values
    static_features = jnp.array([30.0, 70.0, 175.0, 0.0, 1.0, dose_duration], dtype=jnp.float64)
    return jnp.concatenate([jnp.array([-1.0], dtype=jnp.float64), augment_values, static_features])


def _build_constrained_model(*, global_max_time: float = 120.0) -> remifentanil_node.RemiNODE:
    """Construct a constrained RemiNODE with deterministic initialization."""
    return remifentanil_node.RemiNODE(
        augment_dim=2,
        static_dim=len(remifentanil_node.STATIC_FEATURE_NAMES),
        width_size=16,
        depth=2,
        global_max_time=global_max_time,
        dose_duration_index=remifentanil_node.STATIC_FEATURE_NAMES.index(remifentanil_node.DOSE_DURATION_FEATURE_NAME),
        constraint_mode=remifentanil_node.CONSTRAINT_MODE_SOFT_PRE_HARD_POST,
        constraint_buffer_fraction=0.5,
        constraint_pre_bias_scale=1.0,
        key=jax.random.PRNGKey(0),
    )


class TestRemiNODEConstraints:
    """Tests for the optional infusion-aware sign constraints."""

    def test_constraint_mode_none_preserves_unconstrained_vector_field(self):
        """The default mode should continue to use the legacy unconstrained vector field."""
        model = remifentanil_node.RemiNODE(
            augment_dim=2,
            static_dim=len(remifentanil_node.STATIC_FEATURE_NAMES),
            width_size=16,
            depth=2,
            key=jax.random.PRNGKey(0),
        )
        y_aug = _build_augmented_state()

        derivative = model.func(0.25, y_aug, None)

        assert isinstance(model.func, remifentanil_node.RemiNODEFunc)
        assert derivative.shape == y_aug.shape

    def test_constrained_mode_keeps_static_feature_derivatives_zero(self):
        """Static patient features should remain frozen even when the sign constraint is enabled."""
        model = _build_constrained_model()
        y_aug = _build_augmented_state()

        derivative = model.func(0.25, y_aug, None)

        np.testing.assert_allclose(
            np.asarray(derivative[-len(remifentanil_node.STATIC_FEATURE_NAMES):]),
            0.0,
            rtol=0.0,
            atol=1e-12,
        )

    def test_constrained_mode_enforces_negative_gradient_after_buffer(self):
        """The concentration derivative must become non-positive after the post-infusion buffer."""
        model = _build_constrained_model()
        y_aug = _build_augmented_state()
        _, _, transition_end = model.func._scaled_infusion_window(y_aug)

        derivative = model.func(transition_end + 1e-3, y_aug, None)

        assert derivative[0] <= 0.0

    def test_constrained_transition_is_finite_and_differentiable_around_switch(self):
        """The constrained head should stay numerically smooth around the infusion-end transition."""
        model = _build_constrained_model()
        y_aug = _build_augmented_state()
        dose_duration_scaled, buffer_scaled, transition_end = model.func._scaled_infusion_window(y_aug)

        eval_times = (
            dose_duration_scaled - 1e-3,
            dose_duration_scaled + 0.5 * buffer_scaled,
            transition_end + 1e-3,
        )

        for time_point in eval_times:
            value, grad = jax.value_and_grad(lambda t: model.func(t, y_aug, None)[0])(time_point)
            assert jnp.isfinite(value)
            assert jnp.isfinite(grad)

    def test_buffer_fraction_maps_to_one_and_a_half_durations_in_raw_time(self):
        """A 50% buffer should place hard negative enforcement at 1.5x the infusion duration."""
        model = _build_constrained_model(global_max_time=120.0)
        y_aug = _build_augmented_state(dose_duration=60.0)

        _, _, transition_end = model.func._scaled_infusion_window(y_aug)
        transition_end_raw_minutes = float(transition_end * model.func.global_max_time)

        assert transition_end_raw_minutes == pytest.approx(90.0)

    def test_constrained_make_step_returns_finite_loss_and_gradients(self, mock_raw_patients):
        """Two optimizer steps should produce finite losses and parameter gradients."""
        processed_data, _, scalers = remifentanil_node.prepare_dataset([mock_raw_patients[0]])
        batched_data = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *processed_data)

        model = remifentanil_node.RemiNODE(
            augment_dim=2,
            static_dim=len(remifentanil_node.STATIC_FEATURE_NAMES),
            width_size=16,
            depth=2,
            global_max_time=scalers["global_max_time"],
            dose_duration_index=remifentanil_node.STATIC_FEATURE_NAMES.index(remifentanil_node.DOSE_DURATION_FEATURE_NAME),
            constraint_mode=remifentanil_node.CONSTRAINT_MODE_SOFT_PRE_HARD_POST,
            constraint_buffer_fraction=0.5,
            constraint_pre_bias_scale=1.0,
            key=jax.random.PRNGKey(1),
        )
        optimizer = optax.adamw(1e-3)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        @eqx.filter_value_and_grad
        def loss_fn(current_model):
            predictions = jax.vmap(current_model, in_axes=(0, 0, 0))(
                batched_data.t_meas_scaled,
                batched_data.y0,
                batched_data.static_augmentations,
            )
            squared_error = (predictions - batched_data.c_meas_scaled) ** 2
            masked_error = jnp.where(batched_data.measurement_mask, squared_error, 0.0)
            return jnp.sum(masked_error) / jnp.sum(batched_data.measurement_mask)

        loss_value, grads = loss_fn(model)
        grad_leaves = [
            leaf
            for leaf in jax.tree_util.tree_leaves(grads)
            if eqx.is_inexact_array(leaf)
        ]

        assert jnp.isfinite(loss_value)
        assert grad_leaves
        for leaf in grad_leaves:
            assert jnp.all(jnp.isfinite(leaf))

        loss_step_1, model, opt_state = remifentanil_node.make_step(
            model,
            optimizer,
            opt_state,
            batched_data.t_meas_scaled,
            batched_data.y0,
            batched_data.static_augmentations,
            batched_data.c_meas_scaled,
            batched_data.measurement_mask,
        )
        loss_step_2, _, _ = remifentanil_node.make_step(
            model,
            optimizer,
            opt_state,
            batched_data.t_meas_scaled,
            batched_data.y0,
            batched_data.static_augmentations,
            batched_data.c_meas_scaled,
            batched_data.measurement_mask,
        )

        assert jnp.isfinite(loss_step_1)
        assert jnp.isfinite(loss_step_2)
