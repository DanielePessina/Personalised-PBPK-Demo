"""
Kalman filtering utilities for smoothing noisy EEG observations.

Implements a simple 1D linear-Gaussian state space model using JAX
and Equinox. The model assumes a random-walk latent state with
additive Gaussian measurement noise. We provide a convenience function
to apply smoothing across a list of PatientRecord instances.
"""

from __future__ import annotations

from typing import Iterable, List

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .data import PatientRecord


class KalmanSmoother1D(eqx.Module):
    """1D random-walk Kalman filter + RTS smoother.

    The latent state evolves as x_{t+1} = x_t + q_t, and observations
    are y_t = x_t + r_t, where q_t ~ N(0, Q_t) and r_t ~ N(0, R).
    We scale Q_t by the time step between observations to handle
    irregular sampling.

    Parameters are expressed relative to the variance of the observed
    sequence to make the defaults robust across different scales.

    Attributes:
        process_var_scale: Multiplier for per-unit-time process variance Q.
        obs_var_scale: Multiplier for measurement variance R.
        initial_var_scale: Multiplier for initial state variance P0.
        min_var: Small positive value to keep variances well-conditioned.
    """

    process_var_scale: float = 0.05
    obs_var_scale: float = 1.0
    initial_var_scale: float = 1.0
    min_var: float = 1e-6

    def smooth(self, times: jnp.ndarray, values: jnp.ndarray, return_variance: bool = False) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
        """Run Kalman filtering and Rauch–Tung–Striebel smoothing.

        Args:
            times: 1D array of monotonically increasing times, shape [T].
            values: 1D array of observations, shape [T].
            return_variance: If True, return both smoothed means and variances.

        Returns:
            If return_variance is False: 1D array of smoothed state means, shape [T].
            If return_variance is True: tuple of (smoothed means, smoothed variances), both shape [T].
        """
        # Handle edge cases
        T = values.shape[0]
        if T == 0:
            return values
        if T == 1:
            return values

        # Robust scale: use (unbiased) variance of observations
        var_y = jnp.var(values, dtype=values.dtype) + self.min_var
        q_base = self.process_var_scale * var_y
        r = self.obs_var_scale * var_y + self.min_var
        p0 = self.initial_var_scale * var_y + self.min_var

        # Per-step process variance Q_t scaled by dt (irregular sampling)
        # Use dt[0] = 1.0 as a nominal first step.
        dt = jnp.concatenate([
            jnp.array([1.0], dtype=values.dtype), jnp.diff(times)
        ])
        dt = jnp.clip(dt, a_min=self.min_var, a_max=jnp.inf)
        Q = q_base * dt

        # Forward pass: Kalman filter
        def kf_step(carry, inp):
            m_prev, P_prev = carry
            y_t, q_t = inp
            # Predict
            m_pred = m_prev
            P_pred = P_prev + q_t
            # Update
            S = P_pred + r
            K = jnp.where(S > 0.0, P_pred / S, 0.0)
            resid = y_t - m_pred
            m_filt = m_pred + K * resid
            P_filt = (1.0 - K) * P_pred
            return (m_filt, P_filt), (m_filt, P_filt, m_pred, P_pred)

        m0 = values[0]
        P0 = p0
        (_, _), (m_filt, P_filt, m_pred, P_pred) = jax.lax.scan(
            kf_step, (m0, P0), (values, Q)
        )

        # Backward pass: RTS smoother
        # Initialise smoothed means with filtered means; set last equal.
        m_smooth = jnp.zeros_like(m_filt)
        m_smooth = m_smooth.at[-1].set(m_filt[-1])

        if return_variance:
            # Also track smoothed variances for uncertainty quantification
            P_smooth = jnp.zeros_like(P_filt)
            P_smooth = P_smooth.at[-1].set(P_filt[-1])

        def bwd_body(i, carry):
            if return_variance:
                ms, Ps = carry
                # i runs from 0 .. T-2; map to t = T-2-i (reverse order)
                t = (T - 2) - i
                # Smoother gain J_t = P_filt[t] / P_pred[t+1]
                denom = P_pred[t + 1]
                J_t = jnp.where(denom > 0.0, P_filt[t] / denom, 0.0)
                # m_pred[t+1] is the one-step prediction from filtered t
                update = J_t * (ms[t + 1] - m_pred[t + 1])
                ms = ms.at[t].set(m_filt[t] + update)
                # Smoothed variance: P_smooth[t] = P_filt[t] + J_t * (P_smooth[t+1] - P_pred[t+1]) * J_t
                P_update = J_t * (Ps[t + 1] - P_pred[t + 1]) * J_t
                Ps = Ps.at[t].set(P_filt[t] + P_update)
                return (ms, Ps)
            else:
                ms = carry
                # i runs from 0 .. T-2; map to t = T-2-i (reverse order)
                t = (T - 2) - i
                # Smoother gain J_t = P_filt[t] / P_pred[t+1]
                denom = P_pred[t + 1]
                J_t = jnp.where(denom > 0.0, P_filt[t] / denom, 0.0)
                # m_pred[t+1] is the one-step prediction from filtered t
                update = J_t * (ms[t + 1] - m_pred[t + 1])
                ms = ms.at[t].set(m_filt[t] + update)
                return ms

        if return_variance:
            m_smooth, P_smooth = jax.lax.fori_loop(0, T - 1, bwd_body, (m_smooth, P_smooth))
            return m_smooth, P_smooth
        else:
            m_smooth = jax.lax.fori_loop(0, T - 1, bwd_body, m_smooth)
            return m_smooth


def smooth_patients_with_kalman(
    records: Iterable[PatientRecord], *, process_var_scale: float = 0.05, obs_var_scale: float = 1.0,
    initial_var_scale: float = 1.0
) -> List[PatientRecord]:
    """Apply 1D Kalman smoothing to each patient's EEG series.

    Args:
        records: Iterable of PatientRecord instances.
        process_var_scale: Multiplier for per-unit-time process variance.
        obs_var_scale: Multiplier for observation noise variance.
        initial_var_scale: Multiplier for initial state variance.

    Returns:
        A new list of PatientRecord with ``values`` replaced by smoothed values.
    """
    smoother = KalmanSmoother1D(
        process_var_scale=process_var_scale,
        obs_var_scale=obs_var_scale,
        initial_var_scale=initial_var_scale,
    )

    out: List[PatientRecord] = []
    for rec in records:
        times = jnp.asarray(rec.times)
        values = jnp.asarray(rec.values)
        if values.shape[0] <= 1:
            smoothed = values
        else:
            smoothed = smoother.smooth(times, values, return_variance=False)
        # Convert back to numpy for consistency with PatientRecord
        smoothed_np = np.asarray(smoothed, dtype=float)
        out.append(
            PatientRecord(
                id=rec.id,
                times=np.asarray(rec.times, dtype=float),
                values=smoothed_np,
                covariates=np.asarray(rec.covariates, dtype=float),
                dose_rate=float(getattr(rec, "dose_rate", 0.0)),
                dose_duration=float(getattr(rec, "dose_duration", 0.0)),
            )
        )
    return out


def smooth_patients_with_kalman_and_variance(
    records: Iterable[PatientRecord], *, process_var_scale: float = 0.05, obs_var_scale: float = 1.0,
    initial_var_scale: float = 1.0
) -> tuple[List[PatientRecord], List[np.ndarray]]:
    """Apply 1D Kalman smoothing to each patient's EEG series and return variances.

    Args:
        records: Iterable of PatientRecord instances.
        process_var_scale: Multiplier for per-unit-time process variance.
        obs_var_scale: Multiplier for observation noise variance.
        initial_var_scale: Multiplier for initial state variance.

    Returns:
        A tuple of (smoothed_records, variances_list) where:
        - smoothed_records: List of PatientRecord with ``values`` replaced by smoothed values.
        - variances_list: List of numpy arrays containing smoothed variances for each patient.
    """
    smoother = KalmanSmoother1D(
        process_var_scale=process_var_scale,
        obs_var_scale=obs_var_scale,
        initial_var_scale=initial_var_scale,
    )

    out_records: List[PatientRecord] = []
    out_variances: List[np.ndarray] = []

    for rec in records:
        times = jnp.asarray(rec.times)
        values = jnp.asarray(rec.values)

        if values.shape[0] <= 1:
            smoothed = values
            variances = jnp.zeros_like(values)
        else:
            smoothed, variances = smoother.smooth(times, values, return_variance=True)

        # Convert back to numpy for consistency with PatientRecord
        smoothed_np = np.asarray(smoothed, dtype=float)
        variances_np = np.asarray(variances, dtype=float)

        out_records.append(
            PatientRecord(
                id=rec.id,
                times=np.asarray(rec.times, dtype=float),
                values=smoothed_np,
                covariates=np.asarray(rec.covariates, dtype=float),
                dose_rate=float(getattr(rec, "dose_rate", 0.0)),
                dose_duration=float(getattr(rec, "dose_duration", 0.0)),
            )
        )
        out_variances.append(variances_np)

    return out_records, out_variances
