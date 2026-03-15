import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import equinox as eqx
from typing import Tuple

# --- Parameter configuration ---
param_names: Tuple[str, ...] = (
    "k_TP",
    "k_PT",
    "k_PHP",
    "k_HPP",
    "k_EL_Pl",
    "Eff_kid",
    "Eff_hep",
    "k_EL_Tis",
)
P = len(param_names)

# Mark which parameters are fractions (constrained to (0,1)) vs positive rates
param_is_fraction: Tuple[bool, ...] = (False, False, False, False, False, True, True, False)

# Covariates used for all parameters by default
covariate_names: Tuple[str, ...] = ("age", "weight", "height", "bsa", )
C = len(covariate_names)

# Initial population values (natural scale) used to build preimages
init_kin: jax.Array = jnp.array(
    [
        0.03538742575982093,  # k_TP
        1.970013557855141,  # k_PT
        1.8928271523721207,  # k_PHP
        0.17767543157739724,  # k_HPP
        5.525652727926651,  # k_EL_Pl
        0.5548170936796774,  # Eff_kid (fraction)
        0.4866315841958292,  # Eff_hep (fraction)
        0.7695315126895349,  # k_EL_Tis
    ],
    dtype=jnp.float64,
)

# --- Link helpers ---
EPS = 1e-8


def softplus_inv(x: jax.Array) -> jax.Array:
    return jnp.log(jnp.exp(x) - 1.0 + EPS)


def logit(x: jax.Array) -> jax.Array:
    x = jnp.clip(x, EPS, 1.0 - EPS)
    return jnp.log(x) - jnp.log(1.0 - x)


# Preimage init per parameter type
init_pop_pre_list = []
for idx, val in enumerate(init_kin):
    if param_is_fraction[idx]:
        init_pop_pre_list.append(logit(val))
    else:
        init_pop_pre_list.append(softplus_inv(val))
init_pop_pre: jax.Array = jnp.array(init_pop_pre_list)


# --- NLME Model ---
class NLMEModel(eqx.Module):
    # Trainable fields
    pop_pre: jax.Array  # (P,)
    beta: jax.Array  # (P, C)
    eta: jax.Array  # (N_train, P)
    L_diag_pre: jax.Array  # (P,)  -> L_diag = softplus(L_diag_pre)
    sigma_add_pre: jax.Array  # ()    -> sigma_add = softplus(sigma_add_pre)
    sigma_prop_pre: jax.Array  # ()    -> sigma_prop = softplus(sigma_prop_pre)

    # Static (non-trainable) metadata
    param_is_fraction: tuple
    param_names: tuple
    covariate_names: tuple

    def __init__(self, n_patients: int):
        self.pop_pre = init_pop_pre
        self.beta = jnp.zeros((P, C), dtype=jnp.float64)
        self.eta = jnp.zeros((n_patients, P), dtype=jnp.float64)
        self.L_diag_pre = jnp.full((P,), softplus_inv(0.1), dtype=jnp.float64)  # start with SD ~ 0.1
        self.sigma_add_pre = jnp.array(softplus_inv(0.05), dtype=jnp.float64)
        self.sigma_prop_pre = jnp.array(softplus_inv(0.15), dtype=jnp.float64)
        self.param_is_fraction = param_is_fraction
        self.param_names = tuple(param_names)
        self.covariate_names = tuple(covariate_names)

    # Forward link from preimage to natural scale (element-wise per parameter)
    def _forward_link(self, pre_vec: jax.Array) -> jax.Array:
        pos = jax.nn.softplus(pre_vec)
        frac = jax.nn.sigmoid(pre_vec)
        # Use boolean mask to select per-parameter transform
        mask = jnp.array(self.param_is_fraction)
        return jnp.where(mask, frac, pos)

    def natural_from_pre(self, pre_vec: jax.Array) -> jax.Array:
        return self._forward_link(pre_vec)

    def sigma_add(self) -> jax.Array:
        return jax.nn.softplus(self.sigma_add_pre) + EPS

    def sigma_prop(self) -> jax.Array:
        return jax.nn.softplus(self.sigma_prop_pre) + EPS

    def L_diag(self) -> jax.Array:
        # Diagonal Cholesky => Omega = diag(L_diag**2); SDs are L_diag
        return jax.nn.softplus(self.L_diag_pre) + EPS

    def individual_parameters(self, X: jax.Array, use_eta: bool = True) -> jax.Array:
        # X: (N, C)
        # Returns natural-scale kinetic parameters per patient: (N, P)
        N = X.shape[0]
        pop = jnp.broadcast_to(self.pop_pre, (N, P))
        cov = X @ self.beta.T  # (N, P)
        if use_eta:
            if self.eta.shape[0] != N:
                # In training, they match; in test, we pass use_eta=False
                raise ValueError("Shape mismatch: eta and X have different batch sizes.")
            pre = pop + cov + self.eta
        else:
            pre = pop + cov
        return self.natural_from_pre(pre)


def save_model(model: NLMEModel, path: str):
    """Saves the model to a file using Equinox serialization."""
    with open(path, "wb") as f:
        eqx.tree_serialise_leaves(f, model)


def load_model(path: str, template_model: NLMEModel) -> NLMEModel:
    """Loads the model from a file using a template model."""
    with open(path, "rb") as f:
        return eqx.tree_deserialise_leaves(f, template_model)


__all__ = ["NLMEModel", "save_model", "load_model", "param_names", "covariate_names"]
