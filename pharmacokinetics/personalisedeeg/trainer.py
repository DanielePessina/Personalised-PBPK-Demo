"""
Training and evaluation utilities for the remifentanil Neural ODE.

This module orchestrates the end‑to‑end pipeline: splitting the
dataset into train/validation/test sets, fitting scalers and
dimensionality reducers, constructing the model, running the
optimisation loop, and computing basic evaluation metrics.  The
training logic follows the curriculum strategy used in the context
code: multiple phases with decreasing learning rates and increasing
sequence lengths.  Losses are computed using a masked mean squared
error over the padded sequences.

The key entry points are :func:`train_model`, which returns both
training losses and the trained model, and :func:`evaluate_model`,
which reports simple aggregate metrics on a prepared dataset.
"""

from __future__ import annotations

from typing import Tuple, List, Optional, Dict, Callable, Any

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .data import PatientRecord, PreparedData, split_dataset, prepare_dataset, prepare_dataset_with_concentration
from .config import ModelConfig, TrainingConfig, DimReductionConfig, DataSplitConfig, PBPKConfig
from .dim_reduction import PCAReducer, TSNEReducer, MLPReducer
from .model import NeuralODEModel, FullModel


def _apply_scalers_and_pad(
    records: List[PatientRecord],
    value_scaler,
    covar_scaler,
    global_max_time: float,
    T_max: int,
) -> PreparedData:
    """Apply pre‑fitted scalers to a set of patient records and pad sequences.

    This helper mirrors :func:`prepare_dataset` but uses externally supplied
    scalers and maximum time.  It ensures that the resulting arrays
    have the same ``T_max`` as the training data.  Padding beyond the
    last observation uses the 'safe_beyond' strategy.
    """
    N = len(records)
    t_pad = np.zeros((N, T_max), dtype=np.float32)
    y_pad = np.zeros((N, T_max), dtype=np.float32)
    mask = np.zeros((N, T_max), dtype=bool)
    covars_scaled = np.zeros((N, records[0].covariates.shape[0]), dtype=np.float32)
    dose_rate_arr = np.zeros((N,), dtype=np.float32)
    dose_duration_arr = np.zeros((N,), dtype=np.float32)
    lengths_arr = np.zeros(N, dtype=int)
    ids = np.zeros(N, dtype=int)
    for i, rec in enumerate(records):
        ids[i] = rec.id
        L = rec.times.shape[0]
        lengths_arr[i] = L
        # Scale covariates and dose fields together using the training scaler
        dr_val = float(getattr(rec, "dose_rate", 0.0))
        dd_val = float(getattr(rec, "dose_duration", 0.0))
        combo = np.concatenate([rec.covariates.astype(float), [dr_val, dd_val]], axis=0).reshape(1, -1)
        scaled_combo = covar_scaler.transform(combo).astype(np.float32).flatten()
        covars_scaled[i] = scaled_combo[:-2]
        dose_rate_arr[i] = scaled_combo[-2]
        dose_duration_arr[i] = scaled_combo[-1]
        if L > 0:
            # Scale time to [0,1]
            t_scaled = (rec.times / global_max_time).astype(np.float32)
            # Scale values
            v_scaled = value_scaler.transform(rec.values.reshape(-1, 1)).flatten().astype(np.float32)
            t_pad[i, :L] = t_scaled
            y_pad[i, :L] = v_scaled
            mask[i, :L] = True
            # Pad with safe_beyond
            if T_max > L:
                n_pad = T_max - L
                if L > 1:
                    dt = float(t_scaled[-1] - t_scaled[-2])
                else:
                    dt = 1.0 / max(1.0, global_max_time)
                pad_start = 1.0 + dt
                t_pad[i, L:] = pad_start + np.arange(n_pad, dtype=np.float32) * dt
                y_pad[i, L:] = v_scaled[-1]
                mask[i, L:] = False
        else:
            # No measurements: pad with uniform time grid and zeros
            if T_max > 0:
                t_pad[i, :] = np.linspace(0.0, 1.0, T_max, dtype=np.float32)
                y_pad[i, :] = 0.0
                mask[i, :] = False
    return PreparedData(
        t_pad=jnp.asarray(t_pad),
        y_pad=jnp.asarray(y_pad),
        mask=jnp.asarray(mask),
        covars=jnp.asarray(covars_scaled),
        dose_rate=jnp.asarray(dose_rate_arr),
        dose_duration=jnp.asarray(dose_duration_arr),
        lengths=jnp.asarray(lengths_arr),
        ids=jnp.asarray(ids),
        scalers={
            "value_scaler": value_scaler,
            "covar_scaler": covar_scaler,
            "global_max_time": global_max_time,
        },
        dense_ts=None,  # Initialize PBPK fields as None
        dense_cs=None,
    )


def _apply_scalers_and_pad_with_pbpk(
    records: List[PatientRecord],
    value_scaler,
    covar_scaler,
    global_max_time: float,
    T_max: int,
    pbpk_config: PBPKConfig,
    train_dense_ts: jnp.ndarray,
    concentration_scaler,
) -> PreparedData:
    """Apply scalers and add PBPK data for validation/test sets."""
    # First get the basic prepared data
    prepared_data = _apply_scalers_and_pad(records, value_scaler, covar_scaler, global_max_time, T_max)

    # Add PBPK data if needed
    if pbpk_config and pbpk_config.add_concentration:
        # Create PBPK models for patients that have kinetic parameters
        patients_with_kinetics = []
        kinetics_arrays = []
        record_indices = []  # Track which records have kinetics

        for i, record in enumerate(records):
            patient_id = record.id
            if patient_id in pbpk_config.nlme_kinetics_dict:
                # Create physiological parameters from the record
                from .. import remifentanil

                phys_params = remifentanil.PhysiologicalParameters.from_patient_record(record)
                patients_with_kinetics.append(phys_params)

                # Get the kinetic parameters for this patient
                kinetic_dict = pbpk_config.nlme_kinetics_dict[patient_id]
                param_names = ["k_TP", "k_PT", "k_PHP", "k_HPP", "k_EL_Pl", "Eff_kid", "Eff_hep", "k_EL_Tis"]
                kinetic_array = jnp.array([kinetic_dict[name] for name in param_names])
                kinetics_arrays.append(kinetic_array)
                record_indices.append(i)

        if patients_with_kinetics:
            # Stack kinetics for vectorized simulation
            individual_kinetics = jnp.stack(kinetics_arrays)

            # Dense simulation - use same time points as training
            t_dense = jnp.linspace(0.0, global_max_time, pbpk_config.dense_sim_points)

            # Vectorized dense simulation
            dense_cs = remifentanil.vectorized_simulate_dense(
                patients_with_kinetics,
                individual_kinetics,
                t_dense=t_dense,
            )

            # Scale concentrations using the same scaler as training
            if dense_cs.size > 0:
                dense_cs_flat = np.array(dense_cs).flatten()
                nonzero_mask = dense_cs_flat > 0
                if np.any(nonzero_mask):
                    dense_cs_flat_scaled = np.zeros_like(dense_cs_flat)
                    dense_cs_flat_scaled[nonzero_mask] = concentration_scaler.transform(
                        dense_cs_flat[nonzero_mask].reshape(-1, 1)
                    ).flatten()
                    dense_cs_scaled = dense_cs_flat_scaled.reshape(dense_cs.shape)
                else:
                    dense_cs_scaled = np.zeros_like(dense_cs)
            else:
                dense_cs_scaled = dense_cs

            # Create dense arrays for all patients
            n_total_patients = len(records)
            dense_cs_padded = jnp.zeros((n_total_patients, pbpk_config.dense_sim_points))

            # Fill in the computed concentrations
            for kinetics_idx, record_idx in enumerate(record_indices):
                dense_cs_padded = dense_cs_padded.at[record_idx].set(dense_cs_scaled[kinetics_idx])

            # Update the prepared data with PBPK arrays
            prepared_data.dense_ts = train_dense_ts  # Use same time points as training
            prepared_data.dense_cs = dense_cs_padded
            prepared_data.scalers["concentration_scaler"] = concentration_scaler

    return prepared_data


def dataloader(arrays, batch_size: int, *, key: jax.Array, drop_last: bool = True):
    """Construct an infinite generator yielding mini‑batches from aligned arrays.

    Args:
        arrays: Tuple of aligned arrays with shape (N, ...).
        batch_size: Number of samples per batch; if -1, uses full batch.
        key: PRNG key for random shuffling.

    Returns:
        A generator yielding tuples of sliced arrays corresponding to
        randomly shuffled mini‑batches.
    """
    dataset_size = arrays[0].shape[0]
    assert all(arr.shape[0] == dataset_size for arr in arrays), "All arrays must share leading dimension"
    indices = jnp.arange(dataset_size)

    def generator(key):
        current_key = key
        while True:
            current_key, subkey = jr.split(current_key)
            perm = jr.permutation(subkey, indices)
            start = 0
            B = dataset_size if batch_size == -1 else batch_size
            while start < dataset_size:
                end = min(start + B, dataset_size)
                if drop_last and (end - start) < B and B != dataset_size:
                    # Keep shapes static to avoid JIT recompiles on the final smaller batch
                    break
                batch_idx = perm[start:end]
                yield tuple(arr[batch_idx] for arr in arrays)
                start = end

    return generator(key)


def train_model(
    records: List[PatientRecord],
    model_config: ModelConfig,
    training_config: TrainingConfig,
    dr_config: DimReductionConfig,
    split_config: Optional[DataSplitConfig] = None,
    pbpk_config: Optional[PBPKConfig] = None,
    covariate_cols: Optional[List[str]] = None,
    *,
    key: Optional[jax.Array] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    show_internal_progress: bool = True,
) -> Tuple[jnp.ndarray, FullModel, PreparedData, PreparedData, PreparedData]:
    """Train the neural ODE model on the provided patient cohort.

    The cohort is split into training, validation and test subsets
    according to ``split_config``.  For the training set scalers are
    fitted and, if required, a non‑learnable dimensionality reduction
    is applied to the covariates.  A learnable MLP reducer is
    instantiated if selected.  The optimisation proceeds through
    multiple curriculum phases as configured in ``training_config``.

    Args:
        records: Complete list of patient records.
        model_config: Configuration for the neural ODE architecture.
        training_config: Configuration for the training loop.
        dr_config: Configuration for the dimensionality reduction.
        split_config: Optional configuration for data splitting; if
            ``None`` a default 70/15/15 split is used.
        key: Optional PRNG key; if ``None`` a new key is generated.

    Returns:
        A tuple containing:
            - An array of training losses.
            - The trained :class:`FullModel`.
            - The prepared training data (instance of :class:`PreparedData`).
            - The prepared validation data (or ``None`` if no validation records).
            - The prepared test data (or ``None`` if no test records).
            - The precomputed latent vectors for the training set if using a
              non‑learnable reducer, otherwise ``None``.
            - The precomputed latent vectors for the validation set if
              using a non‑learnable reducer, otherwise ``None``.
            - The precomputed latent vectors for the test set if using a
              non‑learnable reducer, otherwise ``None``.
            - The reducer object used (``PCAReducer``, ``TSNEReducer``,
              ``MLPReducer``), which can be inspected or reused.
    """
    # PRNG handling
    if key is None:
        key = jr.PRNGKey(training_config.seed)
    # Split dataset
    split_cfg = split_config or DataSplitConfig()
    train_records, val_records, test_records = split_dataset(records, split_cfg)
    assert len(train_records) > 0, "Training set must not be empty"

    # Prepare training data (fit scalers)
    if pbpk_config and pbpk_config.add_concentration:
        train_data = prepare_dataset_with_concentration(train_records, pbpk_config, covariate_cols)
        value_scaler = train_data.scalers["value_scaler"]
        covar_scaler = train_data.scalers["covar_scaler"]
        concentration_scaler = train_data.scalers["concentration_scaler"]
    else:
        train_data, value_scaler, covar_scaler = prepare_dataset(train_records)
        concentration_scaler = None
    # Determine T_max and global_max_time
    T_max = int(train_data.t_pad.shape[1])
    global_max_time = float(train_data.scalers["global_max_time"])
    # Prepare val/test data using train scalers
    val_data = None
    if len(val_records) > 0:
        if pbpk_config and pbpk_config.add_concentration:
            val_data = _apply_scalers_and_pad_with_pbpk(
                val_records,
                value_scaler,
                covar_scaler,
                global_max_time,
                T_max,
                pbpk_config,
                train_data.dense_ts,
                concentration_scaler,
            )
        else:
            val_data = _apply_scalers_and_pad(val_records, value_scaler, covar_scaler, global_max_time, T_max)
    test_data = None
    if len(test_records) > 0:
        if pbpk_config and pbpk_config.add_concentration:
            test_data = _apply_scalers_and_pad_with_pbpk(
                test_records,
                value_scaler,
                covar_scaler,
                global_max_time,
                T_max,
                pbpk_config,
                train_data.dense_ts,
                concentration_scaler,
            )
        else:
            test_data = _apply_scalers_and_pad(test_records, value_scaler, covar_scaler, global_max_time, T_max)
    # Dimensionality reduction on covariates
    reducer = None
    latent_const_train = None
    latent_const_val = None
    latent_const_test = None
    # Single source of truth for latent dimensionality comes from dr_config
    latent_dim = int(dr_config.latent_dim)
    if dr_config.method.lower() == "pca":
        reducer = PCAReducer(latent_dim)
        # Include dose features in reduction input
        train_in = np.concatenate(
            [
                np.asarray(train_data.covars),
                np.asarray(train_data.dose_rate)[:, None],
                np.asarray(train_data.dose_duration)[:, None],
            ],
            axis=1,
        )
        reducer.fit(train_in)
        latent_const_train = jnp.asarray(reducer.transform(train_in), dtype=jnp.float32)
        if val_data is not None:
            val_in = np.concatenate(
                [
                    np.asarray(val_data.covars),
                    np.asarray(val_data.dose_rate)[:, None],
                    np.asarray(val_data.dose_duration)[:, None],
                ],
                axis=1,
            )
            latent_const_val = jnp.asarray(reducer.transform(val_in), dtype=jnp.float32)
        if test_data is not None:
            test_in = np.concatenate(
                [
                    np.asarray(test_data.covars),
                    np.asarray(test_data.dose_rate)[:, None],
                    np.asarray(test_data.dose_duration)[:, None],
                ],
                axis=1,
            )
            latent_const_test = jnp.asarray(reducer.transform(test_in), dtype=jnp.float32)
    elif dr_config.method.lower() == "tsne":
        reducer = TSNEReducer(
            latent_dim, dr_config.tsne_perplexity, dr_config.tsne_learning_rate, dr_config.tsne_n_iter
        )
        # t‑SNE does not have a separate transform, so we fit on train and transform each set separately
        train_in = np.concatenate(
            [
                np.asarray(train_data.covars),
                np.asarray(train_data.dose_rate)[:, None],
                np.asarray(train_data.dose_duration)[:, None],
            ],
            axis=1,
        )
        reducer.fit(train_in)
        latent_const_train = jnp.asarray(reducer.transform(train_in), dtype=jnp.float32)
        if val_data is not None:
            val_in = np.concatenate(
                [
                    np.asarray(val_data.covars),
                    np.asarray(val_data.dose_rate)[:, None],
                    np.asarray(val_data.dose_duration)[:, None],
                ],
                axis=1,
            )
            latent_const_val = jnp.asarray(reducer.transform(val_in), dtype=jnp.float32)
        if test_data is not None:
            test_in = np.concatenate(
                [
                    np.asarray(test_data.covars),
                    np.asarray(test_data.dose_rate)[:, None],
                    np.asarray(test_data.dose_duration)[:, None],
                ],
                axis=1,
            )
            latent_const_test = jnp.asarray(reducer.transform(test_in), dtype=jnp.float32)
    elif dr_config.method.lower() == "mlp":
        # Create a learnable MLP reducer
        covar_dim = train_data.covars.shape[1]
        reducer_key, key = jr.split(key)
        reducer = MLPReducer(
            in_size=int(covar_dim + 2),  # append dose_rate and dose_duration
            latent_dim=int(latent_dim),
            mlp_width=int(dr_config.mlp_width),
            mlp_depth=int(dr_config.mlp_depth),
            activation=jax.nn.swish,
            key=reducer_key,
        )
        # No precomputed latent vectors; they will be computed on the fly
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {dr_config.method}")
    # Build the neural ODE model with the chosen latent dimension
    # Build the neural ODE model
    model_key, key = jr.split(key)
    node = NeuralODEModel(model_config, latent_dim=latent_dim, key=model_key)
    full_model = FullModel(node=node, reducer=(reducer if isinstance(reducer, MLPReducer) else None), latent_const=None)
    # Prepare arrays for training
    t_pad = train_data.t_pad  # [N, T_max]
    y_pad = train_data.y_pad[:, :, None]  # [N, T_max, 1]
    covars = train_data.covars  # [N, C]
    dose_rate = train_data.dose_rate  # [N]
    dose_duration = train_data.dose_duration  # [N]
    latent_const = latent_const_train  # [N, latent_dim] if not None
    lengths = train_data.lengths  # [N]
    # Optimiser state will be reset each phase
    losses = []
    # Progress / bookkeeping
    total_steps = int(sum(training_config.steps_strategy))
    global_step = 0
    if show_internal_progress:
        console = Console(width=100)
        training_progress = Progress(
            SpinnerColumn(),
            BarColumn(bar_width=20, complete_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.description}", justify="right"),
            console=console,
            auto_refresh=True,
            refresh_per_second=5,
            transient=not training_config.verbose,
        )
        progress_task = training_progress.add_task("[cyan]Training", total=total_steps)
    # Dataloader for training
    batch_size = train_data.t_pad.shape[0] if training_config.batch_size == -1 else training_config.batch_size

    # Outer loop over curriculum phases
    def _train_loop():
        nonlocal global_step, full_model
        for phase, (lr, steps, length_frac) in enumerate(
            zip(
                training_config.lr_strategy,
                training_config.steps_strategy,
                training_config.length_strategy,
                strict=False,
            )
        ):
            # Optimiser: reset for each phase
            # optim = optax.adamw(lr, weight_decay=1e-6)
            optim = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adabelief(
                    lr,
                ),
            )
            opt_state = optim.init(eqx.filter(full_model, eqx.is_inexact_array))
            # Sequence length for this phase
            slice_len = int(length_frac * T_max)
            # Slice arrays to current length
            ts_slice = t_pad[:, :slice_len]
            ys_slice = y_pad[:, :slice_len, :]
            covars_slice = covars
            latent_slice = latent_const[:, :latent_dim] if latent_const is not None else None
            lengths_slice = jnp.minimum(lengths, slice_len)
            dr_slice = dose_rate
            dd_slice = dose_duration
            dense_ts_slice = train_data.dense_ts
            dense_cs_slice = train_data.dense_cs
            # Dataloader
            data_gen = dataloader(
                (
                    ts_slice,
                    ys_slice,
                    covars_slice,
                    latent_slice if latent_const is not None else covars_slice,
                    lengths_slice,
                    dr_slice,
                    dd_slice,
                    dense_ts_slice,
                    dense_cs_slice,
                ),
                batch_size,
                key=key,
            )

            # Define loss and update functions
            # # # --- Original loss function commented out ---
            def loss_fn(model: FullModel, ts_b, y_b, cov_b, latent_b, len_b, dr_b, dd_b, dense_ts_b, dense_cs_b):
                # y_b: [B, T, 1]
                B, T, _ = y_b.shape
                # y0: initial state per sample
                y0_batch = y_b[:, 0, :]  # [B, 1]
                # latent input per sample
                if isinstance(reducer, MLPReducer):
                    latent_input = cov_b  # covariates used to compute latent via reducer
                else:
                    latent_input = latent_b  # precomputed latent vectors

                # Vmap over batch dimension; wrap to avoid kwargs in vmapped call
                def _predict_one(ts_i, y0_i, cov_i, lat_i, len_i, dr_i, dd_i, dense_ts_i, dense_cs_i):
                    return model(
                        ts_i,
                        y0_i,
                        cov_i,
                        lat_i,
                        len_i,
                        dose_rate=dr_i,
                        dose_duration=dd_i,
                        dense_ts=dense_ts_i,
                        dense_cs=dense_cs_i,
                        return_full_state=False,
                    )

                preds = jax.vmap(_predict_one, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0))(
                    ts_b, y0_batch, cov_b, latent_input, len_b, dr_b, dd_b, dense_ts_b, dense_cs_b
                )  # [B, T, 1]
                # Masked MSE
                mask = jnp.arange(T)[None, :] < len_b[:, None]
                diff = (y_b - preds) ** 2
                masked_diff = jnp.where(mask[:, :, None], diff, 0.0)
                sum_err = jnp.sum(masked_diff, axis=(1, 2))
                denom = jnp.maximum(len_b.astype(jnp.float32), 1.0)
                per_exp_mse = sum_err / denom
                return jnp.mean(per_exp_mse)

            # # --- Balanced loss function ---
            # def loss_fn(model: FullModel, ts_b, y_b, cov_b, latent_b, len_b, dr_b, dd_b):
            #     """
            #     Computes the mean squared error for each patient (masked),
            #     then averages across all patients for a balanced loss.
            #     Masking is applied to exclude padded values.
            #     """
            #     B, T, _ = y_b.shape
            #     y0_batch = y_b[:, 0, :]  # [B, 1]
            #     if isinstance(reducer, MLPReducer):
            #         latent_input = cov_b
            #     else:
            #         latent_input = latent_b
            #     def _predict_one(ts_i, y0_i, cov_i, lat_i, len_i, dr_i, dd_i):
            #         return model(
            #             ts_i,
            #             y0_i,
            #             cov_i,
            #             lat_i,
            #             len_i,
            #             dose_rate=dr_i,
            #             dose_duration=dd_i,
            #             return_full_state=False,
            #         )
            #     preds = jax.vmap(_predict_one, in_axes=(0, 0, 0, 0, 0, 0, 0))(
            #         ts_b, y0_batch, cov_b, latent_input, len_b, dr_b, dd_b
            #     )  # [B, T, 1]
            #     # Masking: mask out padded values
            #     mask = jnp.arange(T)[None, :] < len_b[:, None]  # [B, T]
            #     # Compute squared error, mask out padding
            #     sq_err = ((y_b - preds) ** 2)[:, :, 0]  # [B, T]
            #     masked_sq_err = jnp.where(mask, sq_err, 0.0)  # [B, T]
            #     # For each patient, average error over valid (unmasked) timesteps
            #     valid_counts = jnp.sum(mask, axis=1).astype(jnp.float32)  # [B]
            #     # Avoid division by zero
            #     per_patient_mse = jnp.where(valid_counts > 0, jnp.sum(masked_sq_err, axis=1) / valid_counts, 0.0)  # [B]
            #     # Final loss: average across all patients
            #     balanced_loss = jnp.mean(per_patient_mse)
            #     return balanced_loss
            # Create jit+grad function
            grad_fn = eqx.filter_value_and_grad(loss_fn)

            @eqx.filter_jit
            def update_step(model, opt_state, ts_b, y_b, cov_b, latent_b, len_b, dr_b, dd_b, dense_ts_b, dense_cs_b):
                loss, grads = grad_fn(model, ts_b, y_b, cov_b, latent_b, len_b, dr_b, dd_b, dense_ts_b, dense_cs_b)
                updates, opt_state_new = optim.update(grads, opt_state, model)
                model = eqx.apply_updates(model, updates)
                return loss, model, opt_state_new

            # Training loop for this phase
            for step in range(steps):
                ts_b, y_b, cov_b, lat_b, len_b, dr_b, dd_b, dense_ts_b, dense_cs_b = next(data_gen)
                loss, full_model, opt_state = update_step(
                    full_model, opt_state, ts_b, y_b, cov_b, lat_b, len_b, dr_b, dd_b, dense_ts_b, dense_cs_b
                )
                losses.append(loss)
                # Internal progress (if enabled)
                if show_internal_progress:
                    training_progress.update(
                        progress_task,
                        advance=1,
                        description=f"phase {phase + 1} loss={loss:.3e}",
                    )
                # External callback for richer UIs
                if progress_callback is not None:
                    try:
                        progress_callback(
                            {
                                "phase": int(phase + 1),
                                "phase_steps": int(steps),
                                "phase_length_frac": float(length_frac),
                                "slice_len": int(slice_len),
                                "lr": float(lr),
                                "step": int(step + 1),
                                "global_step": int(global_step + 1),
                                "total_steps": int(total_steps),
                                "loss": float(loss),
                            }
                        )
                    except Exception:
                        # Avoid breaking training due to UI issues
                        pass
                global_step += 1

    if show_internal_progress:
        with training_progress:
            _train_loop()
    else:
        _train_loop()
    # No additional processing is needed after training: full_model already
    # contains the trained parameters.  Latent vectors for the training,
    # validation and test sets are stored in latent_const_train,
    # latent_const_val and latent_const_test for non‑learnable reducers.
    return (
        jnp.array(losses),
        full_model,
        train_data,
        val_data,
        test_data,
        latent_const_train,
        latent_const_val,
        latent_const_test,
        reducer,
    )


def evaluate_model(
    model: FullModel,
    data: PreparedData,
    latent_const: Optional[jnp.ndarray] = None,
    pbpk_config: Optional[PBPKConfig] = None,
    *,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Compute simple error metrics on a prepared dataset.

    Args:
        model: Trained full model.
        data: PreparedData instance with time series, values, masks, and covariates.
        latent_const: If the reducer is non‑learnable, provide the latent vectors.
        batch_size: Batch size for evaluation; default of 64. Use -1 to evaluate
            the full dataset in a single batch.

    Returns:
        Dictionary of aggregate metrics: mean squared error, mean absolute error and root mean squared error.
    """
    N, T = data.t_pad.shape
    # Determine latent input
    if model.reducer is None:
        assert latent_const is not None, "latent_const must be provided for non‑learnable reducer"
    # Dataloader for evaluation
    # Resolve batch size (-1 means full batch) and use drop_last=False to cover all samples
    B = int(N) if int(batch_size) == -1 else int(batch_size)
    data_gen = dataloader(
        (
            data.t_pad,
            data.y_pad[:, :, None],
            data.covars,
            latent_const if latent_const is not None else data.covars,
            data.lengths,
            data.dose_rate,
            data.dose_duration,
            data.dense_ts,
            data.dense_cs,
        ),
        B,
        key=jr.PRNGKey(0),
        drop_last=False,
    )
    total_sq_err = 0.0
    total_abs_err = 0.0
    total_count = 0.0
    n_batches = 1 if B >= N else int(np.ceil(N / B))
    for _ in range(n_batches):
        ts_b, y_b, cov_b, lat_b, len_b, dr_b, dd_b, dense_ts_b, dense_cs_b = next(data_gen)
        B, T_slice, _ = y_b.shape
        y0_b = y_b[:, 0, :]
        if model.reducer is None:
            latent_in = lat_b
        else:
            latent_in = cov_b

        # Run model; note that FullModel expects ts, y0, covars, latent_input, length
        def _predict_one(ts_i, y0_i, cov_i, lat_i, len_i, dr_i, dd_i, dense_ts_i, dense_cs_i):
            return model(
                ts_i,
                y0_i,
                cov_i,
                lat_i,
                len_i,
                dose_rate=dr_i,
                dose_duration=dd_i,
                dense_ts=dense_ts_i,
                dense_cs=dense_cs_i,
                return_full_state=False,
            )

        preds = jax.vmap(_predict_one, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0))(
            ts_b, y0_b, cov_b, latent_in, len_b, dr_b, dd_b, dense_ts_b, dense_cs_b
        )  # [B, T_slice, 1]
        # Compute mask
        mask = jnp.arange(T_slice)[None, :] < len_b[:, None]
        diff = y_b - preds
        sq_err = jnp.sum((diff**2) * mask[:, :, None])
        abs_err = jnp.sum(jnp.abs(diff) * mask[:, :, None])
        count = jnp.sum(mask)
        total_sq_err += float(sq_err)
        total_abs_err += float(abs_err)
        total_count += float(count)
    mse = float(total_sq_err / jnp.maximum(total_count, 1.0))
    mae = float(total_abs_err / jnp.maximum(total_count, 1.0))
    rmse = float(jnp.sqrt(total_sq_err / jnp.maximum(total_count, 1.0)))
    return {"mse": mse, "mae": mae, "rmse": rmse}
