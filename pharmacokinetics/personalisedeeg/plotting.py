"""
Plotting utilities for visualizing model performance and simulation results.

This module provides a function to generate a parity (predicted vs true)
plot for the EEG Neural ODE models. It supports both the full wrapper
model (with reducer) and a raw NODE if latent vectors are supplied.

Features
- Scatter parity plot with y=x dashed line
- ±10% deviation reference lines (dotted)
- Dark2 color cycle and subtle grid styling
- On-plot metrics: RMSE, MSE, R², and MAPE
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt

from .data import PreparedData
from .model import FullModel, NeuralODEModel

import pyfonts

pyfonts.set_default_font(pyfonts.load_google_font("Roboto Slab"))

# Global plotting defaults (kept lightweight and neutral)
mpl.rcParams["figure.dpi"] = 500
mpl.rcParams["axes.grid"] = True
mpl.rcParams["axes.grid.which"] = "both"
mpl.rcParams["grid.linestyle"] = "--"
mpl.rcParams["grid.alpha"] = 0.4
mpl.rcParams["grid.linewidth"] = 0.6
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["ytick.minor.visible"] = True

# Use Dark2 as a color cycle to match request
try:
    from cycler import cycler

    dark2 = plt.get_cmap("Dark2").colors
    mpl.rcParams["axes.prop_cycle"] = cycler(color=dark2)
except Exception:
    pass


def _collect_predictions(
    model_or_node: FullModel | NeuralODEModel,
    data: PreparedData,
    latent_const: Optional[jnp.ndarray],
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the model over ``data`` and return (y_true, y_pred) in original units.

    For ``FullModel``, covariates are used directly when a learnable reducer
    is present; for ``NeuralODEModel`` callers must provide ``latent_const``
    with shape [N, latent_dim].
    """
    N, T_max = data.t_pad.shape
    # Decide how to provide latent input per sample
    use_full = isinstance(model_or_node, FullModel)
    if not use_full and latent_const is None:
        raise ValueError("latent_const must be provided when passing a raw NeuralODEModel")

    # Value scaler to de-standardize
    value_scaler = data.scalers.get("value_scaler", None)
    if value_scaler is None:
        raise ValueError("PreparedData.scalers must contain 'value_scaler' for destandardization")

    # Containers for concatenation
    true_all = []
    pred_all = []

    # Simple batched loop to avoid large vmaps when N is big
    num_batches = int(np.ceil(N / max(1, batch_size)))
    for b in range(num_batches):
        start = b * batch_size
        end = min(N, (b + 1) * batch_size)
        if start >= end:
            break
        ts_b = data.t_pad[start:end]  # [B, T]
        y_b = data.y_pad[start:end][:, :, None]  # [B, T, 1]
        cov_b = data.covars[start:end]  # [B, C]
        len_b = data.lengths[start:end]  # [B]
        dr_b = data.dose_rate[start:end]
        dd_b = data.dose_duration[start:end]
        y0_b = y_b[:, 0, :]

        if use_full:
            # For FullModel, latent input is covariates (reducer computes latent inside)
            # Check if we have PBPK data
            dense_ts_b = data.dense_ts[start:end] if data.dense_ts is not None else None
            dense_cs_b = data.dense_cs[start:end] if data.dense_cs is not None else None

            def _predict_one(ts_i, y0_i, cov_i, len_i, dr_i, dd_i, dense_ts_i, dense_cs_i):
                return model_or_node(
                    ts_i,
                    y0_i,
                    cov_i,
                    cov_i,  # latent_input unused when reducer is present
                    len_i,
                    dose_rate=dr_i,
                    dose_duration=dd_i,
                    dense_ts=dense_ts_i,
                    dense_cs=dense_cs_i,
                    return_full_state=False,
                )

            # Handle case where dense data might be None
            if dense_ts_b is not None and dense_cs_b is not None:
                preds = jax.vmap(_predict_one, in_axes=(0, 0, 0, 0, 0, 0, 0, 0))(
                    ts_b, y0_b, cov_b, len_b, dr_b, dd_b, dense_ts_b, dense_cs_b
                )
            else:
                # Fallback for models without PBPK data - pass None directly
                preds = jax.vmap(_predict_one, in_axes=(0, 0, 0, 0, 0, 0, None, None))(
                    ts_b, y0_b, cov_b, len_b, dr_b, dd_b, None, None
                )
        else:
            latent_b = latent_const[start:end]

            def _predict_one_node(ts_i, y0_i, lat_i, len_i):
                return model_or_node.node(ts_i, y0_i, lat_i, len_i) if isinstance(model_or_node, FullModel) else model_or_node(
                    ts_i, y0_i, lat_i, len_i
                )

            preds = jax.vmap(_predict_one_node, in_axes=(0, 0, 0, 0))(ts_b, y0_b, latent_b, len_b)

        # Mask for valid time points
        B, T_slice, _ = preds.shape
        mask = (jnp.arange(T_slice)[None, :] < len_b[:, None])[:, :, None]
        # Select only valid entries
        y_true_scaled = jnp.where(mask, y_b, jnp.nan).reshape(B * T_slice, 1)
        y_pred_scaled = jnp.where(mask, preds, jnp.nan).reshape(B * T_slice, 1)
        # Drop NaNs then inverse-transform
        sel = np.isfinite(np.asarray(y_true_scaled)).flatten()
        if np.any(sel):
            yt = value_scaler.inverse_transform(np.asarray(y_true_scaled)[sel].reshape(-1, 1)).flatten()
            yp = value_scaler.inverse_transform(np.asarray(y_pred_scaled)[sel].reshape(-1, 1)).flatten()
            true_all.append(yt)
            pred_all.append(yp)

    if not true_all:
        raise ValueError("No valid points found for parity plot (check masks/lengths)")
    y_true = np.concatenate(true_all, axis=0)
    y_pred = np.concatenate(pred_all, axis=0)
    return y_true, y_pred


def plot_parity(
    model_or_node: FullModel | NeuralODEModel,
    data: PreparedData,
    *,
    latent_const: Optional[jnp.ndarray] = None,
    batch_size: int = 64,
    title: str = "Parity Plot",
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Generate a parity plot (predicted vs true) with metrics.

    Args:
        model_or_node: Trained :class:`FullModel` or raw :class:`NeuralODEModel`.
                       If a raw NODE is provided, ``latent_const`` must be supplied.
        data: :class:`PreparedData` with scaled/padded arrays and scalers.
        latent_const: Precomputed latent vectors [N, latent_dim] when using a raw NODE
                      or a FullModel without a reducer.
        batch_size: Batch size for prediction loop.
        title: Plot title.
        save_path: If provided, the figure is saved to this path.

    Returns:
        The matplotlib Axes containing the plot.
    """
    # Collect predictions and truths in original units
    y_true, y_pred = _collect_predictions(model_or_node, data, latent_const, batch_size)

    # Compute metrics
    diff = y_pred - y_true
    mse = float(np.mean(diff**2)) if y_true.size > 0 else float("nan")
    rmse = float(np.sqrt(mse)) if np.isfinite(mse) else float("nan")
    float(np.mean(np.abs(diff))) if y_true.size > 0 else float("nan")
    # R^2 (guard against zero variance)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (np.sum(diff**2) / denom)) if denom > 0 else float("nan")
    # MAPE (avoid division by zero)
    eps = 1e-8
    mape = float(np.mean(np.abs(diff) / (np.maximum(np.abs(y_true), eps)))) * 100.0

    # Build plot
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    # Scatter points (use first Dark2 color)
    c = plt.get_cmap("Dark2")(0)
    ax.scatter(y_true, y_pred, s=14, alpha=0.9, color=c, edgecolors="none")

    # Reference lines
    mn = float(np.minimum(y_true.min(), y_pred.min()))
    mx = float(np.maximum(y_true.max(), y_pred.max()))
    pad = 0.05 * (mx - mn if mx > mn else 1.0)
    x_ref = np.array([mn - pad, mx + pad])
    # y=x dashed
    ax.plot(x_ref, x_ref, linestyle="--", color="black", linewidth=1.0, alpha=0.9)
    # ±10% dotted
    ax.plot(x_ref, 1.1 * x_ref, linestyle=":", color="gray", linewidth=0.9, alpha=0.9)
    ax.plot(x_ref, 0.9 * x_ref, linestyle=":", color="gray", linewidth=0.9, alpha=0.9)

    ax.set_xlim(mn - pad, mx + pad)
    ax.set_ylim(mn - pad, mx + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("True value (original units)")
    ax.set_ylabel("Predicted value (original units)")

    # Metrics box
    ax.text(
        0.98,
        0.02,
        f"RMSE: {rmse:.3e}\nMSE: {mse:.3e}\nR²: {r2:.3f}\nMAPE: {mape:.2f}%",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.75, "linewidth": 0.6},
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax


def plot_loss_curve(losses: np.ndarray | jnp.ndarray, *, title: str = "Training Loss", save_path: Optional[str] = None):
    """Plot the training loss curve over steps.

    Args:
        losses: Sequence of loss values over optimisation steps.
        title: Title for the figure.
        save_path: Optional path to save the figure.
    """
    vals = np.asarray(losses, dtype=float).reshape(-1)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(np.arange(vals.size), vals, lw=1.2, color=plt.get_cmap("Dark2")(1))
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(title)
    ax.grid(True, alpha=0.4)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return ax


def plot_metrics_bars(metrics_by_split: dict, *, title: str = "Dataset Errors", save_path: Optional[str] = None):
    """Plot a bar chart of MSE/MAE/RMSE across train/val/test splits.

    Args:
        metrics_by_split: Mapping like {"train": {"mse":..., "mae":..., "rmse":...}, ...}
        title: Title for the figure.
        save_path: Optional path to save the figure.
    """
    splits = [k for k in ["train", "val", "test"] if k in metrics_by_split]
    if not splits:
        splits = list(metrics_by_split.keys())
    mses = [metrics_by_split[s]["mse"] for s in splits]
    maes = [metrics_by_split[s]["mae"] for s in splits]
    rmses = [metrics_by_split[s]["rmse"] for s in splits]
    x = np.arange(len(splits))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - width, mses, width, label="MSE", color=plt.get_cmap("Dark2")(0))
    ax.bar(x, maes, width, label="MAE", color=plt.get_cmap("Dark2")(2))
    ax.bar(x + width, rmses, width, label="RMSE", color=plt.get_cmap("Dark2")(3))
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in splits])
    ax.set_ylabel("Error")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return ax


def plot_random_trajectories(
    model_or_node: FullModel | NeuralODEModel,
    data: PreparedData,
    n_patients: int,
    *,
    latent_const: Optional[jnp.ndarray] = None,
    variances: Optional[dict | list] = None,
    title: str = "Random Patient Trajectories",
    save_path: Optional[str] = None,
) -> plt.Axes:
    """
    Plots N random predictions and corresponding true trajectories.

    Args:
        model_or_node: Trained :class:`FullModel` or raw :class:`NeuralODEModel`.
        data: :class:`PreparedData` with scaled/padded arrays and scalers.
        n_patients: Number of random patients to plot.
        latent_const: Precomputed latent vectors [N, latent_dim] when using a raw NODE.
        variances: Optional dict or list of variance arrays (from Kalman filter) for uncertainty bands.
                  If dict, keys should be patient IDs. If list, should be in same order as data.
        title: Plot title.
        save_path: If provided, the figure is saved to this path.

    Returns:
        The matplotlib Axes containing the plot.
    """
    N = data.t_pad.shape[0]
    if n_patients > N:
        n_patients = N

    rng = np.random.default_rng(seed=42)
    patient_indices = rng.choice(N, size=n_patients, replace=False)

    # Value scaler to de-standardize
    value_scaler = data.scalers.get("value_scaler")
    if value_scaler is None:
        raise ValueError("PreparedData.scalers must contain 'value_scaler' for destandardization")

    fig, ax = plt.subplots(figsize=(8, 4.5))

    legend_handles = []

    for i, patient_idx in enumerate(patient_indices):
        ts = data.t_pad[patient_idx]
        y_true_scaled = data.y_pad[patient_idx]
        length = data.lengths[patient_idx]
        y0_scaled = y_true_scaled[0:1]

        # Get prediction
        use_full = isinstance(model_or_node, FullModel)
        if use_full:
            cov = data.covars[patient_idx]
            dr = data.dose_rate[patient_idx]
            dd = data.dose_duration[patient_idx]
            # Handle PBPK data if available
            dense_ts = data.dense_ts[patient_idx] if data.dense_ts is not None else None
            dense_cs = data.dense_cs[patient_idx] if data.dense_cs is not None else None
            y_pred_scaled = model_or_node(
                ts, y0_scaled, cov, cov, length,
                dose_rate=dr, dose_duration=dd,
                dense_ts=dense_ts, dense_cs=dense_cs,
                return_full_state=False
            )
        else:
            if latent_const is None:
                raise ValueError("latent_const must be provided for NeuralODEModel")
            lat = latent_const[patient_idx]
            y_pred_scaled = model_or_node(ts, y0_scaled, lat, length)

        # De-standardize and mask
        ts_valid = ts[:length]
        y_true = value_scaler.inverse_transform(y_true_scaled[:length].reshape(-1, 1)).flatten()
        y_pred = value_scaler.inverse_transform(y_pred_scaled[:length, 0].reshape(-1, 1)).flatten()

        color = plt.get_cmap("Dark2")(i % 8)

        # Plot uncertainty band if variances are provided
        if variances is not None:
            var = None
            if isinstance(variances, dict):
                # Variance dictionary keyed by patient ID
                patient_id = data.ids[patient_idx]
                # Convert JAX array to Python scalar if needed
                if hasattr(patient_id, 'item'):
                    patient_id = patient_id.item()
                elif hasattr(patient_id, '__array__'):
                    patient_id = int(patient_id)

                # print(f"DEBUG: Looking for patient_id {patient_id} in variance dict")

                if patient_id in variances:
                    var = variances[patient_id][:length]
                    # print(f"DEBUG: Found variance for patient {patient_id}, mean std: {np.sqrt(np.mean(var)):.4f}")
                else:
                    print(f"DEBUG: Patient {patient_id} not found in variance dictionary")
            elif isinstance(variances, list) and len(variances) > patient_idx:
                # Variance list in same order as data
                var = variances[patient_idx][:length]

            if var is not None:
                std = np.sqrt(var)

                # The Kalman filter variance applies to the true (observed) values,
                # not the model predictions. Apply uncertainty bands to y_true.
                std_orig = std
                # print(f"DEBUG: Applying uncertainty to TRUE values, std range: [{np.min(std_orig):.4f}, {np.max(std_orig):.4f}]")

                # Plot uncertainty band around TRUE values (±1 standard deviation)
                ax.fill_between(ts_valid, y_true - std_orig, y_true + std_orig,
                              alpha=0.3, color=color, label='±1σ Kalman uncertainty' if i == 0 else "")
                # print(f"DEBUG: Plotted uncertainty band around TRUE values for patient {patient_idx}")
            else:
                print(f"DEBUG: No variance data found for patient {patient_idx}")

        # Plotting
        p, = ax.plot(ts_valid, y_pred, color=color, linestyle="-",)
        ax.plot(ts_valid, y_true, color=color, linestyle="--")
        legend_handles.append(p)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    line_pred = Line2D([0], [0], color="k", linestyle="-", label="Prediction")
    line_true = Line2D([0], [0], color="k", linestyle="--", label="True")

    legend_items = [line_pred, line_true]

    # Add uncertainty band to legend if variances were provided
    if variances is not None:
        patch_uncertainty = Patch(facecolor='gray', alpha=0.3, label='±1σ Kalman uncertainty')
        legend_items.append(patch_uncertainty)

    patient_labels = [f"Patient {data.ids[i]}" for i in patient_indices]

    legend1 = ax.legend(legend_handles, patient_labels, title="Patients", loc="upper right")
    ax.add_artist(legend1)

    ax.legend(handles=legend_items, loc="lower right")

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value (original units)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax
