"""
Neural ODE architecture definitions.

This module contains the core neural ODE components used to model the
evolution of the EEG signal over time.  The state consists of the
measured signal, a number of learnable augmented dimensions and a
latent vector derived from patient covariates.  The latent vector
remains constant during integration, while the signal and augmented
dimensions evolve according to a learnable vector field implemented
as a multilayer perceptron.  The integration is performed using
Diffrax in a single solve over the full time grid, with dynamics
frozen per‑sample after the last valid observation.
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import diffrax

from .config import ModelConfig


class ODEFunc(eqx.Module):
    """Vector field for the neural ODE."""

    mlp: eqx.nn.MLP
    augment_dims: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    include_time: bool = eqx.field(static=True)
    add_concentration: bool = eqx.field(static=True)
    constrained: bool = eqx.field(static=True)
    data_size: int = eqx.field(static=True)

    def __init__(self, config: ModelConfig, latent_dim: int, *, key: jax.Array):
        super().__init__()
        self.augment_dims = int(config.augment_dims)
        self.latent_dim = int(latent_dim)
        self.include_time = bool(config.include_time)
        self.add_concentration = bool(config.add_concentration)
        self.constrained = bool(config.constrained)
        self.data_size = 1  # We predict a single EEG channel

        in_size = self.data_size + self.augment_dims + self.latent_dim
        if self.include_time:
            in_size += 1
        if self.add_concentration:
            in_size += 1

        out_size = self.data_size + self.augment_dims + self.latent_dim
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=int(config.mlp_width),
            depth=int(config.mlp_depth),
            activation=config.activation,
            key=key,
        )

    def __call__(self, t: jax.Array, y: jax.Array, args: Optional[tuple] = None) -> jax.Array:
        """Compute dy/dt at time ``t`` for state ``y``."""

        t_end, dense_ts, dense_cs = args

        mlp_in = [y]
        if self.include_time:
            mlp_in.append(jnp.atleast_1d(t))

        if self.add_concentration:
            c_at_t = jnp.interp(t, dense_ts, dense_cs)
            mlp_in.append(jnp.atleast_1d(c_at_t))

        out = self.mlp(jnp.concatenate(mlp_in, axis=-1))

        if self.constrained:
            out = out.at[0].set(-jnp.cos(out[0]))

        if self.latent_dim > 0:
            out = out.at[-self.latent_dim :].set(0.0)

        if t_end is not None:
            out = jax.lax.cond(t <= t_end, lambda: out, lambda: jnp.zeros_like(out))

        return out


class NeuralODEModel(eqx.Module):
    """Neural ODE model that integrates the ODEFunc over time."""

    func: ODEFunc
    solver: diffrax.AbstractSolver = eqx.field(static=True)
    augment_dims: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    include_time: bool = eqx.field(static=True)
    data_size: int = eqx.field(static=True)

    def __init__(self, config: ModelConfig, latent_dim: int, *, key: jax.Array):
        super().__init__()
        func_key, _ = jax.random.split(key)
        self.func = ODEFunc(config, latent_dim=latent_dim, key=func_key)
        self.solver = config.solver
        self.augment_dims = int(config.augment_dims)
        self.latent_dim = int(latent_dim)
        self.include_time = bool(config.include_time)
        self.data_size = 1

    def __call__(
        self,
        ts: jnp.ndarray,
        y0: jnp.ndarray,
        latent_vec: jnp.ndarray,
        length: jnp.ndarray,
        *,
        dense_ts: Optional[jnp.ndarray] = None,
        dense_cs: Optional[jnp.ndarray] = None,
        return_full_state: bool = False,
    ) -> jnp.ndarray:
        """Integrate the neural ODE over the supplied time grid."""
        assert ts.ndim == 1, "ts must be 1D"
        assert y0.ndim == 1 and y0.shape[0] == self.data_size, "y0 must have shape (1,)"
        assert latent_vec.ndim == 1 and latent_vec.shape[0] == self.latent_dim, (
            f"latent_vec must have shape ({self.latent_dim},)"
        )

        aug0 = jnp.zeros(self.augment_dims, dtype=y0.dtype)
        y_state0 = jnp.concatenate([y0, aug0, latent_vec], axis=-1)

        t0 = ts[0]
        t1 = ts[-1]
        t_end = ts[jnp.maximum(length - 1, 0)]

        args = (t_end, dense_ts, dense_cs)

        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-6)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            self.solver,
            t0,
            t1,
            None,
            y_state0,
            args,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=stepsize_controller,
            max_steps=int(1e12),
        )
        ys_full = sol.ys
        if return_full_state:
            return ys_full
        return ys_full[:, : self.data_size]


class FullModel(eqx.Module):
    """Wrapper combining a latent reducer and the neural ODE."""

    node: NeuralODEModel
    reducer: Optional[eqx.Module] = None
    latent_const: Optional[jnp.ndarray] = None

    def __init__(
        self,
        node: NeuralODEModel,
        reducer: Optional[eqx.Module] = None,
        latent_const: Optional[jnp.ndarray] = None,
    ):
        super().__init__()
        self.node = node
        self.reducer = reducer
        self.latent_const = latent_const

    def __call__(
        self,
        ts: jnp.ndarray,
        y0: jnp.ndarray,
        covars: jnp.ndarray,
        latent_input: jnp.ndarray,
        length: jnp.ndarray,
        *,
        dose_rate: jnp.ndarray | float | None = None,
        dose_duration: jnp.ndarray | float | None = None,
        dense_ts: Optional[jnp.ndarray] = None,
        dense_cs: Optional[jnp.ndarray] = None,
        return_full_state: bool = False,
    ) -> jnp.ndarray:
        """Forward pass through the full model."""
        if self.reducer is not None:
            if (dose_rate is not None) and (dose_duration is not None):
                dr = jnp.atleast_1d(jnp.asarray(dose_rate, dtype=jnp.float32))
                dd = jnp.atleast_1d(jnp.asarray(dose_duration, dtype=jnp.float32))
                reducer_in = jnp.concatenate([covars, dr, dd], axis=-1)
            else:
                reducer_in = covars

            if hasattr(self.reducer, "mlp"):
                latent_vec = self.reducer.mlp(reducer_in)
            else:
                latent_vec = self.reducer(reducer_in)
        else:
            latent_vec = latent_input

        assert latent_vec.ndim == 1 and latent_vec.shape[0] == self.node.latent_dim, (
            f"Reducer produced latent of shape {latent_vec.shape}, expected ({self.node.latent_dim},)"
        )

        return self.node(
            ts, y0, latent_vec, length,
            dense_ts=dense_ts, dense_cs=dense_cs,
            return_full_state=return_full_state
        )
