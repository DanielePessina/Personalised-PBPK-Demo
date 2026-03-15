"""
Remifentanil Neural ODE Model

This module contains the data structures and functions for training Neural Ordinary
Differential Equation (NODE) models to predict remifentanil plasma concentrations.

Key functionalities:
- `RemiNODEData`: Data container for processed patient data
- `prepare_dataset`: Data preprocessing and scaling pipeline
- `RemiNODE`: Neural ODE model architecture with system augmentation
- `train_remifentanil_node`: Main training function with curriculum learning

The main function for users is `train_remifentanil_node` which handles the complete
training pipeline from raw patient data to trained model.
"""
from typing import List, Dict, Tuple, Any, Callable, Literal
import time

import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from rich.console import Console
from sklearn.preprocessing import MinMaxScaler

from .remifentanil import RawPatient


# --- Configuration ---

# Default static features to include as time-invariant inputs to the Neural ODE
STATIC_FEATURE_NAMES = ["age", "weight", "height", "sex", "dose_rate", "dose_duration"]
DOSE_DURATION_FEATURE_NAME = "dose_duration"

CONSTRAINT_MODE_NONE = "none"
CONSTRAINT_MODE_SOFT_PRE_HARD_POST = "soft_pre_hard_post"
ConstraintMode = Literal["none", "soft_pre_hard_post"]


def resolve_hyperparams(hyperparams: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of the NODE hyperparameters with optional constraint defaults filled in.

    Args:
        hyperparams: Mapping with ``model`` and ``training`` sections.

    Returns:
        A new dictionary with resolved model defaults.

    Raises:
        KeyError: If the required ``model`` or ``training`` sections are missing.
        ValueError: If constraint-related settings are invalid.
    """
    model_params = {
        "constraint_mode": CONSTRAINT_MODE_NONE,
        "constraint_buffer_fraction": 0.5,
        "constraint_pre_bias_scale": 1.0,
        **dict(hyperparams["model"]),
    }
    training_params = {
        "logging_steps": 25,
        **dict(hyperparams["training"]),
    }

    constraint_mode = str(model_params["constraint_mode"])
    if constraint_mode not in {CONSTRAINT_MODE_NONE, CONSTRAINT_MODE_SOFT_PRE_HARD_POST}:
        raise ValueError(
            "Unsupported constraint mode. Expected one of "
            f"{CONSTRAINT_MODE_NONE!r} or {CONSTRAINT_MODE_SOFT_PRE_HARD_POST!r}, "
            f"got {constraint_mode!r}."
        )

    buffer_fraction = float(model_params["constraint_buffer_fraction"])
    if buffer_fraction < 0.0:
        raise ValueError("`constraint_buffer_fraction` must be non-negative.")

    pre_bias_scale = float(model_params["constraint_pre_bias_scale"])
    if pre_bias_scale < 0.0:
        raise ValueError("`constraint_pre_bias_scale` must be non-negative.")

    model_params["constraint_mode"] = constraint_mode
    model_params["constraint_buffer_fraction"] = buffer_fraction
    model_params["constraint_pre_bias_scale"] = pre_bias_scale
    return {"model": model_params, "training": training_params}


# --- Data Structures ---

class RemiNODEData(eqx.Module):
    """
    A container for a single patient's processed and scaled data, ready for NODE training.

    This class is an Equinox Module, making it compatible with JAX transformations
    like jit, vmap, and grad. All time-series arrays are right-padded to ensure
    consistent shapes for batching.

    Attributes:
        t_meas_scaled: Measurement times, scaled to [0, 1]. Shape: (max_len,)
        c_meas_scaled: Measured concentrations, scaled to [-1, 1]. Shape: (max_len,)
        measurement_mask: Boolean mask for valid measurements. Shape: (max_len,)
        static_augmentations: Static features for the NODE. Shape: (num_static_features,)
        y0: Initial condition for dynamic state. Shape: (1,)
    """
    # Dynamic data (Padded Arrays)
    t_meas_scaled: jnp.ndarray      # Measurement times, scaled to [0, 1]. Shape: (max_len,)
    c_meas_scaled: jnp.ndarray      # Measured concentrations, scaled to [-1, 1]. Shape: (max_len,)
    measurement_mask: jnp.ndarray   # Boolean mask for valid measurements. Shape: (max_len,)

    # Static, time-invariant features to be fed into the NODE's vector field.
    static_augmentations: jnp.ndarray # Shape: (num_static_features,)

    # The initial condition for the dynamic state variable (plasma concentration).
    # It's a 1D vector because we are predicting a single state.
    y0: jnp.ndarray                 # Shape: (1,)


# --- Data Processing Pipeline ---

def prepare_dataset(
    raw_patients: List[RawPatient],
    static_feature_names: List[str] = STATIC_FEATURE_NAMES,
    scalers: Dict[str, Any] | None = None,
) -> Tuple[List[RemiNODEData], jnp.ndarray, Dict[str, Any]]:
    """
    Processes a list of RawPatient objects into a scaled, padded, and structured dataset.

    This function performs global scaling on time and concentration data. If a `scalers`
    dictionary is provided, it uses them; otherwise, it fits new ones.

    Args:
        raw_patients: A list of `RawPatient` objects loaded from the dataset.
        static_feature_names: A list of strings specifying which patient attributes
            to include as static features for the model.
        scalers: (Optional) A pre-fitted scalers dictionary. If provided, no new
            scalers will be fitted.

    Returns:
        A tuple containing:
        - A list of `RemiNODEData` objects, one for each patient.
        - A JAX array of patient IDs, in the same order as the data list.
        - The scalers dictionary (either the one passed in or the newly fitted one).

    Raises:
        ValueError: If input `raw_patients` list is empty.
    """
    if not raw_patients:
        raise ValueError("Input `raw_patients` list cannot be empty.")

    # --- Fit new scalers if none are provided ---
    if scalers is None:
        # 1. Aggregate data for global scaling
        all_concentrations = []
        global_max_time = 0.0

        for p in raw_patients:
            valid_mask = p.mask
            valid_concentrations = p.c_meas[valid_mask]
            all_concentrations.append(np.array(valid_concentrations))
            if valid_concentrations.size > 0:
                current_max_time = jnp.max(p.t_meas[valid_mask])
                if current_max_time > global_max_time:
                    global_max_time = float(current_max_time)

        # Add a zero to the data to anchor the scaling range, ensuring that a
        # concentration of 0.0 is always part of the dataset for the scaler.
        all_concentrations.append(np.array([0.0]))

        if not all_concentrations or all(c.size == 0 for c in all_concentrations):
            all_concentrations_flat = np.array([]).reshape(-1, 1)
        else:
            all_concentrations_flat = np.concatenate(
                [c for c in all_concentrations if c.size > 0]
            ).reshape(-1, 1)

        # 2. Fit global scalers
        concentration_scaler = MinMaxScaler(feature_range=(-1, 1))
        if all_concentrations_flat.size > 0:
            concentration_scaler.fit(all_concentrations_flat)

        scalers = {
            "concentration_scaler": concentration_scaler,
            "global_max_time": global_max_time,
        }

    # --- Use existing or newly fitted scalers to process data ---
    concentration_scaler = scalers["concentration_scaler"]
    global_max_time = scalers["global_max_time"]

    max_len = 0
    for p in raw_patients:
        num_measurements = int(jnp.sum(p.mask))
        if num_measurements > max_len:
            max_len = num_measurements

    # 3. Process and pad each patient individually
    processed_data_list = []
    patient_ids = []
    for p in raw_patients:
        patient_ids.append(p.id)
        valid_mask_original = p.mask
        original_len = int(jnp.sum(valid_mask_original))
        t_meas_valid = p.t_meas[valid_mask_original]
        c_meas_valid = p.c_meas[valid_mask_original]

        t_meas_scaled = t_meas_valid / global_max_time if global_max_time > 0 else jnp.zeros_like(t_meas_valid)
        if c_meas_valid.size > 0:
            c_meas_scaled = concentration_scaler.transform(np.array(c_meas_valid).reshape(-1, 1)).flatten()
        else:
            c_meas_scaled = np.array([])

        t_pad_val = t_meas_scaled[-1] if original_len > 0 else 0
        t_meas_padded = jnp.pad(t_meas_scaled, (0, max_len - original_len), 'constant', constant_values=t_pad_val)
        c_meas_padded = jnp.pad(jnp.asarray(c_meas_scaled), (0, max_len - original_len), 'constant', constant_values=0)
        measurement_mask_padded = jnp.arange(max_len) < original_len

        static_features = []
        for feature_name in static_feature_names:
            feature_val = getattr(p, feature_name)
            if isinstance(feature_val, bool):
                feature_val = float(feature_val)
            static_features.append(feature_val)
        static_augmentations_vec = jnp.array(static_features, dtype=jnp.float64)

        node_data = RemiNODEData(
            t_meas_scaled=t_meas_padded,
            c_meas_scaled=c_meas_padded,
            measurement_mask=measurement_mask_padded,
            static_augmentations=static_augmentations_vec,
            y0=jnp.array([-1.0])
        )
        processed_data_list.append(node_data)

    return processed_data_list, jnp.array(patient_ids, dtype=jnp.int32), scalers


def unscale_predictions(
    scaled_predictions: jnp.ndarray,
    scalers: Dict[str, Any]
) -> jnp.ndarray:
    """
    Converts scaled model predictions back to the original concentration units.

    Args:
        scaled_predictions: Model predictions in scaled units [-1, 1]
        scalers: Dictionary containing fitted scalers

    Returns:
        Predictions in original concentration units
    """
    conc_scaler = scalers["concentration_scaler"]
    unscaled = conc_scaler.inverse_transform(np.array(scaled_predictions).reshape(-1, 1))
    return jnp.asarray(unscaled).flatten()


def unscale_data(
    node_data: RemiNODEData,
    scalers: Dict[str, Any]
) -> Dict[str, jnp.ndarray]:
    """
    Converts a RemiNODEData object's scaled data back to original units for verification.

    Args:
        node_data: Processed patient data in scaled units
        scalers: Dictionary containing fitted scalers

    Returns:
        Dictionary with 'time' and 'concentration' in original units
    """
    mask = node_data.measurement_mask
    t_unscaled = node_data.t_meas_scaled[mask] * scalers['global_max_time']
    c_scaled_valid = node_data.c_meas_scaled[mask]
    c_unscaled = unscale_predictions(c_scaled_valid, scalers)
    return {"time": t_unscaled, "concentration": c_unscaled}


# --- Neural ODE Model Definition ---

class RemiNODEFunc(eqx.Module):
    """The vector field of the Neural ODE, represented by an MLP."""
    mlp: eqx.nn.MLP
    num_static_features: int = eqx.field(static=True)

    def __init__(self, augment_dim: int, static_dim: int, width_size: int, depth: int, *, key: jax.Array):
        # The total input to the MLP is the sum of the dynamic state (1),
        # the learned augmentations, and the static patient features.
        in_size = 1 + augment_dim + static_dim
        self.mlp = eqx.nn.MLP(in_size, in_size, width_size, depth, activation=jnn.softplus, key=key)
        self.num_static_features = static_dim

    def __call__(self, t: float, y_aug: jnp.ndarray, args: Any) -> jnp.ndarray:
        """
        Computes the derivative of the augmented state.

        Args:
            t: time (unused, but required by the solver interface).
            y_aug: The augmented state [y_dynamic, y_learned_aug, y_static_aug].
            args: Any other arguments (unused).

        Returns:
            The derivative of the augmented state. The derivatives for the static
            features are explicitly set to zero.
        """
        derivatives = self.mlp(y_aug)
        # Zero out the derivatives for the static features
        return derivatives.at[-self.num_static_features:].set(0.0)


class ConstrainedRemiNODEFunc(eqx.Module):
    """
    Vector field with an infusion-aware sign-constrained head for ``dC/dt``.

    The concentration derivative is the only component that receives the explicit
    mechanistic prior. Learned augmentation coordinates remain free so the model
    can absorb residual structure without forcing a sign pattern on every state.
    The static patient features remain appended to the state purely as frozen
    conditioning variables; their derivatives are always set to zero.
    """

    mlp: eqx.nn.MLP
    augment_dim: int = eqx.field(static=True)
    num_static_features: int = eqx.field(static=True)
    dose_duration_index: int = eqx.field(static=True)
    global_max_time: float = eqx.field(static=True)
    constraint_buffer_fraction: float = eqx.field(static=True)
    constraint_pre_bias_scale: float = eqx.field(static=True)

    def __init__(
        self,
        augment_dim: int,
        static_dim: int,
        width_size: int,
        depth: int,
        *,
        dose_duration_index: int,
        global_max_time: float,
        constraint_buffer_fraction: float,
        constraint_pre_bias_scale: float,
        key: jax.Array,
    ):
        state_size = 1 + augment_dim + static_dim
        # The constrained head needs explicit timing context. We therefore append:
        #   1. the current scaled solver time t
        #   2. the phase relative to the end of infusion, normalized by the buffer
        #   3. the scaled infusion duration itself
        in_size = state_size + 3
        out_size = 4 + augment_dim
        self.mlp = eqx.nn.MLP(in_size, out_size, width_size, depth, activation=jnn.softplus, key=key)
        self.augment_dim = augment_dim
        self.num_static_features = static_dim
        self.dose_duration_index = dose_duration_index
        self.global_max_time = float(global_max_time)
        self.constraint_buffer_fraction = float(constraint_buffer_fraction)
        self.constraint_pre_bias_scale = float(constraint_pre_bias_scale)

    @staticmethod
    def _smootherstep(u: jnp.ndarray) -> jnp.ndarray:
        """
        Return the 5th-order smooth step used to blend the transition band.

        The polynomial has zero first derivative at both ends, which avoids a
        visible kink when the ODE switches from the pre-infusion bias to the
        post-buffer hard negative regime.
        """
        return u**3 * (u * (u * 6.0 - 15.0) + 10.0)

    def _scaled_infusion_window(self, y_aug: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Compute the scaled infusion end and post-infusion buffer from the static state.

        ``dose_duration`` is stored in raw minutes inside the static feature block.
        The ODE, however, is solved on the globally scaled time axis ``[0, 1]``.
        We therefore convert the patient-specific duration using the cohort-level
        ``global_max_time`` captured when the model is built.
        """
        dtype = y_aug.dtype
        if self.global_max_time <= 0.0:
            zero = jnp.array(0.0, dtype=dtype)
            return zero, zero, zero

        dose_duration = y_aug[-self.num_static_features + self.dose_duration_index]
        dose_duration = jnp.maximum(dose_duration.astype(dtype), 0.0)
        dose_duration_scaled = dose_duration / jnp.asarray(self.global_max_time, dtype=dtype)
        buffer_scaled = dose_duration_scaled * jnp.asarray(self.constraint_buffer_fraction, dtype=dtype)
        return dose_duration_scaled, buffer_scaled, dose_duration_scaled + buffer_scaled

    def __call__(self, t: float, y_aug: jnp.ndarray, args: Any) -> jnp.ndarray:
        """
        Compute the derivative of the augmented state with infusion-aware sign control.

        The first output is treated as the plasma concentration derivative:
        - before the infusion end we apply a soft positive bias
        - during the post-end transition band we blend through a free middle state
        - after the buffer we enforce an exact negative sign via ``-softplus(...)``
        """
        del args
        dtype = y_aug.dtype
        dose_duration_scaled, buffer_scaled, transition_end = self._scaled_infusion_window(y_aug)

        # When the buffer becomes very small we still need a safe denominator for
        # the relative phase features. The actual piecewise regime boundaries keep
        # using the true buffer width, so this epsilon only protects the division.
        safe_buffer = jnp.maximum(buffer_scaled, jnp.finfo(dtype).eps)
        phase = jnp.clip((jnp.asarray(t, dtype=dtype) - dose_duration_scaled) / safe_buffer, -2.0, 2.0)
        constrained_inputs = jnp.concatenate(
            [
                y_aug,
                jnp.array(
                    [jnp.asarray(t, dtype=dtype), phase, dose_duration_scaled],
                    dtype=dtype,
                ),
            ]
        )

        raw_outputs = self.mlp(constrained_inputs)
        pre_bias_raw, pre_free_raw, mid_raw, post_raw = raw_outputs[:4]
        augment_derivatives = raw_outputs[4:]

        pre_drift = pre_free_raw + jnp.asarray(self.constraint_pre_bias_scale, dtype=dtype) * jnn.softplus(pre_bias_raw)
        post_drift = -jnn.softplus(post_raw)

        u = jnp.clip((jnp.asarray(t, dtype=dtype) - dose_duration_scaled) / safe_buffer, 0.0, 1.0)
        blend = self._smootherstep(u)
        transition_drift = (
            (1.0 - blend) ** 2 * pre_drift
            + 2.0 * blend * (1.0 - blend) * mid_raw
            + blend**2 * post_drift
        )

        concentration_derivative = jnp.where(
            jnp.asarray(t, dtype=dtype) <= dose_duration_scaled,
            pre_drift,
            jnp.where(jnp.asarray(t, dtype=dtype) >= transition_end, post_drift, transition_drift),
        )

        static_derivatives = jnp.zeros(self.num_static_features, dtype=dtype)
        return jnp.concatenate(
            [
                jnp.atleast_1d(concentration_derivative),
                augment_derivatives,
                static_derivatives,
            ]
        )


class RemiNODE(eqx.Module):
    """A Neural ODE model for Remifentanil pharmacokinetics."""
    func: RemiNODEFunc | ConstrainedRemiNODEFunc
    solver: diffrax.AbstractSolver = eqx.field(static=True)
    augment_dim: int = eqx.field(static=True)
    static_dim: int = eqx.field(static=True)
    constraint_mode: ConstraintMode = eqx.field(static=True)
    constraint_buffer_fraction: float = eqx.field(static=True)
    constraint_pre_bias_scale: float = eqx.field(static=True)

    def __init__(
        self,
        augment_dim: int,
        static_dim: int,
        width_size: int,
        depth: int,
        *,
        key: jax.Array,
        global_max_time: float | None = None,
        dose_duration_index: int | None = None,
        constraint_mode: ConstraintMode = CONSTRAINT_MODE_NONE,
        constraint_buffer_fraction: float = 0.5,
        constraint_pre_bias_scale: float = 1.0,
    ):
        self.augment_dim = augment_dim
        self.static_dim = static_dim
        self.constraint_mode = constraint_mode
        self.constraint_buffer_fraction = float(constraint_buffer_fraction)
        self.constraint_pre_bias_scale = float(constraint_pre_bias_scale)

        # Keep the legacy autonomous vector field untouched unless the user
        # explicitly opts into the infusion-aware sign-constrained variant.
        if constraint_mode == CONSTRAINT_MODE_NONE:
            self.func = RemiNODEFunc(augment_dim, static_dim, width_size, depth, key=key)
        else:
            if dose_duration_index is None:
                raise ValueError("`dose_duration_index` is required when constraint mode is enabled.")
            if global_max_time is None:
                raise ValueError("`global_max_time` is required when constraint mode is enabled.")
            self.func = ConstrainedRemiNODEFunc(
                augment_dim,
                static_dim,
                width_size,
                depth,
                dose_duration_index=dose_duration_index,
                global_max_time=global_max_time,
                constraint_buffer_fraction=constraint_buffer_fraction,
                constraint_pre_bias_scale=constraint_pre_bias_scale,
                key=key,
            )
        self.solver = diffrax.Tsit5()  # A good general-purpose solver

    def __call__(self, ts: jnp.ndarray, y0: jnp.ndarray, static_features: jnp.ndarray) -> jnp.ndarray:
        """
        Solves the ODE for a given set of initial conditions and static features.

        Args:
            ts: The time points at which to save the solution.
            y0: The initial condition for the dynamic state (shape `(1,)`).
            static_features: The time-invariant patient features.

        Returns:
            The predicted trajectory for the dynamic state (plasma concentration).
        """
        # y0 is for the dynamic state. Learned augmentations also start at 0.
        y0_learned_aug = jnp.zeros(self.augment_dim)
        y0_full_augmented = jnp.concatenate([y0, y0_learned_aug, static_features])

        # Set up and solve the differential equation
        term = diffrax.ODETerm(self.func)
        solution = diffrax.diffeqsolve(
            term,
            self.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=None,  # Solver will determine initial step size
            y0=y0_full_augmented,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6)
        )

        # Return only the first dimension, which corresponds to the dynamic state
        return solution.ys[:, 0]


# --- Training and Evaluation ---

def dataloader(arrays: Tuple[jnp.ndarray, ...], batch_size: int, *, key: jax.Array):
    """A JAX-based infinite generator for creating mini-batches."""
    dataset_size = arrays[0].shape[0]
    if batch_size > dataset_size:
        batch_size = dataset_size
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        key, subkey = jr.split(key)
        perm = jr.permutation(subkey, indices)
        start = 0
        while start < dataset_size:
            end = start + batch_size
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end


@eqx.filter_jit
def make_step(
    model: RemiNODE,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    ts_batch: jnp.ndarray,
    y0_batch: jnp.ndarray,
    static_features_batch: jnp.ndarray,
    c_meas_batch: jnp.ndarray,
    mask_batch: jnp.ndarray,
):
    """Performs a single gradient update step."""

    @eqx.filter_value_and_grad
    def loss_fn(model, ts, y0, static, c_meas, mask):
        # Vmap the model over the batch dimension
        y_pred_batch = jax.vmap(model, in_axes=(0, 0, 0))(ts, y0, static)
        # Calculate masked mean squared error
        error = (y_pred_batch - c_meas) ** 2
        masked_error = jnp.where(mask, error, 0.)
        loss = jnp.sum(masked_error) / jnp.sum(mask)
        return loss

    loss, grads = loss_fn(model, ts_batch, y0_batch, static_features_batch, c_meas_batch, mask_batch)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


def train_remifentanil_node(
    raw_patients: List[RawPatient],
    hyperparams: Dict[str, Any],
    seed: int = 42
) -> Tuple[RemiNODE, Dict[str, Any]]:
    """
    Main function to orchestrate the training of the RemiNODE model.

    This function handles the complete training pipeline including data preprocessing,
    model initialization, and training with curriculum learning.

    Args:
        raw_patients: List of RawPatient objects containing patient data
        hyperparams: Dictionary containing model and training hyperparameters with keys:
            - 'model': dict with 'augment_dim', 'width_size', 'depth'
              and optional constraint settings:
              'constraint_mode', 'constraint_buffer_fraction',
              'constraint_pre_bias_scale'
            - 'training': dict with 'lr_strategy', 'steps_strategy', 'length_strategy', 'batch_size'
        seed: Random seed for reproducibility

    Returns:
        Tuple of (trained_model, scalers) where:
        - trained_model: The trained RemiNODE model
        - scalers: Dictionary containing fitted data scalers for future use

    Example:
        >>> from pharmacokinetics.remifentanil import import_patients
        >>> from pharmacokinetics.remifentanil_node import train_remifentanil_node
        >>>
        >>> # Load patient data
        >>> patients = import_patients("data.xlsx")
        >>>
        >>> # Define hyperparameters
        >>> hyperparams = {
        ...     "model": {
        ...         "width_size": 64,
        ...         "depth": 3,
        ...         "augment_dim": 4,
        ...     },
        ...     "training": {
        ...         "lr_strategy": (1e-3, 5e-4),
        ...         "steps_strategy": (500, 1000),
        ...         "length_strategy": (0.3, 1.0),
        ...         "batch_size": 32,
        ...     }
        ... }
        >>>
        >>> # Train the model
        >>> model, scalers = train_remifentanil_node(patients, hyperparams)
    """
    console = Console()
    key = jr.PRNGKey(seed)
    model_key, loader_key = jr.split(key)

    # 1. Prepare Data
    console.log("[bold cyan]Step 1: Preparing dataset...[/bold cyan]")
    processed_dataset, patient_ids, scalers = prepare_dataset(raw_patients)

    # Stack data for batching
    batched_data = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *processed_dataset)

    # 2. Initialize Model and Optimizer
    resolved_hyperparams = resolve_hyperparams(hyperparams)
    model_params = resolved_hyperparams['model']
    training_params = resolved_hyperparams['training']
    logging_steps = max(1, int(training_params.get("logging_steps", 25)))

    model = RemiNODE(
        augment_dim=model_params['augment_dim'],
        static_dim=len(STATIC_FEATURE_NAMES),
        width_size=model_params['width_size'],
        depth=model_params['depth'],
        global_max_time=scalers["global_max_time"],
        dose_duration_index=STATIC_FEATURE_NAMES.index(DOSE_DURATION_FEATURE_NAME),
        constraint_mode=model_params['constraint_mode'],
        constraint_buffer_fraction=model_params['constraint_buffer_fraction'],
        constraint_pre_bias_scale=model_params['constraint_pre_bias_scale'],
        key=model_key
    )

    # 3. Training Loop with Curriculum Learning
    console.log("[bold cyan]Step 2: Starting training...[/bold cyan]")
    total_steps = sum(training_params['steps_strategy'])
    console.log(
        "Training schedule: "
        f"total_steps={total_steps}, "
        f"phases={len(training_params['lr_strategy'])}, "
        f"batch_size={int(training_params['batch_size'])}, "
        f"logging_steps={logging_steps}"
    )

    completed_steps = 0

    # Curriculum learning phases
    for i, (lr, steps, length_frac) in enumerate(zip(
        training_params['lr_strategy'],
        training_params['steps_strategy'],
        training_params['length_strategy'],
        strict=True
    )):
        phase_start_time = time.perf_counter()
        optimizer = optax.adamw(lr)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        console.log(
            f"Phase {i + 1}/{len(training_params['lr_strategy'])}: "
            f"lr={lr:.3e} steps={steps} length_frac={length_frac:.3f}"
        )

        # Create a new dataloader for each phase, now including patient IDs
        train_loader = dataloader(
            (patient_ids, batched_data.t_meas_scaled, batched_data.y0, batched_data.static_augmentations, batched_data.c_meas_scaled, batched_data.measurement_mask),
            training_params['batch_size'],
            key=loader_key
        )

        for step_idx in range(steps):
            step_start_time = time.perf_counter()
            # Unpack the batch, ignoring the patient_id_b as it's not used in make_step
            _, ts_b, y0_b, sf_b, c_b, m_b = next(train_loader)

            # Slice the data for curriculum learning
            num_timesteps = ts_b.shape[1]
            slice_end = int(num_timesteps * length_frac)
            ts_slice = ts_b[:, :slice_end]
            c_slice = c_b[:, :slice_end]
            m_slice = m_b[:, :slice_end]

            if step_idx == 0:
                console.log(
                    f"Phase {i + 1} first batch: "
                    f"batch_shape={tuple(ts_b.shape)} "
                    f"slice_end={slice_end} "
                    f"ts_shape={tuple(ts_slice.shape)}"
                )

            loss, model, opt_state = make_step(model, optimizer, opt_state, ts_slice, y0_b, sf_b, c_slice, m_slice)

            step_elapsed = time.perf_counter() - step_start_time
            completed_steps += 1

            if step_idx == 0 or (step_idx + 1) % logging_steps == 0 or (step_idx + 1) == steps:
                phase_elapsed = time.perf_counter() - phase_start_time
                console.log(
                    f"Phase {i + 1} step {step_idx + 1}/{steps} "
                    f"(global {completed_steps}/{total_steps}): "
                    f"loss={float(loss):.6e} "
                    f"step_time={step_elapsed:.2f}s "
                    f"phase_elapsed={phase_elapsed:.2f}s"
                )

    console.log("[bold green]✓ Training complete.[/bold green]")
    return model, scalers
