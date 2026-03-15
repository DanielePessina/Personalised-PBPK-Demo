"""
Dimensionality reduction utilities for patient covariates.

This module defines three simple reducers: PCA, t‑SNE and a
learnable MLP.  The PCA and t‑SNE reducers wrap scikit‑learn
implementations and are therefore non‑trainable once fitted.  The
MLP reducer uses Equinox and produces latent vectors whose
parameters are jointly optimised with the Neural ODE during
training.

Each reducer exposes a common interface with ``fit`` and
``transform`` methods.  The MLP reducer additionally implements
``__call__`` for JAX/EQX compatibility.  The reducers are agnostic
to the eventual size of the covariate matrix and rely on the
calling code to ensure that only numeric inputs are passed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class BaseReducer:
    """Abstract base class for dimensionality reducers."""

    latent_dim: int

    def fit(self, X: np.ndarray) -> "BaseReducer":
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, X: Any) -> Any:
        return self.transform(X)


@dataclass
class PCAReducer(BaseReducer):
    """Principal component analysis (PCA) reducer using scikit‑learn.

    Attributes:
        latent_dim: Number of principal components to retain.
        pca: Internal sklearn PCA object.
    """

    latent_dim: int
    pca: Optional[PCA] = None

    def fit(self, X: np.ndarray) -> "PCAReducer":
        self.pca = PCA(n_components=self.latent_dim)
        self.pca.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("PCA reducer has not been fitted yet")
        return self.pca.transform(X).astype(np.float32)


@dataclass
class TSNEReducer(BaseReducer):
    """t‑Distributed Stochastic Neighbour Embedding (t‑SNE) reducer.

    Note that t‑SNE does not implement a separate transform operation in
    scikit‑learn; ``transform`` therefore re‑fits the model on the input
    data each time.  This yields embeddings that are not aligned
    across calls.  Use with caution if you wish to compare train and
    test embeddings directly.  The non‑learnable nature of t‑SNE
    makes it more suitable for visualisation rather than as a true
    conditioning signal.

    Attributes:
        latent_dim: Number of t‑SNE components to compute.
        perplexity: t‑SNE perplexity parameter.
        learning_rate: t‑SNE learning rate.
        n_iter: Number of optimisation iterations.
        tsne: Internal sklearn TSNE object.
    """

    latent_dim: int
    perplexity: float = 30.0
    learning_rate: float = 200.0
    n_iter: int = 1000
    tsne: Optional[TSNE] = None

    def fit(self, X: np.ndarray) -> "TSNEReducer":
        # Fit and transform in one step
        self.tsne = TSNE(
            n_components=self.latent_dim,
            perplexity=self.perplexity,
            learning_rate=self.learning_rate,
            n_iter=self.n_iter,
            init="random",
            method="barnes_hut",
            random_state=0,
        )
        self.tsne.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # For t‑SNE we call fit_transform on the input; note that
        # embeddings will differ between calls.
        tsne = TSNE(
            n_components=self.latent_dim,
            perplexity=self.perplexity,
            learning_rate=self.learning_rate,
            n_iter=self.n_iter,
            init="random",
            method="barnes_hut",
            random_state=0,
        )
        embedding = tsne.fit_transform(X)
        return embedding.astype(np.float32)


class MLPReducer(eqx.Module):
    """Learnable MLP reducer for covariates.

    This reducer defines an Equinox MLP that maps covariate vectors to
    latent vectors.  It exposes both a ``fit`` method (which is a
    no‑op since the MLP is initialised randomly) and a ``__call__``
    method that takes a JAX array and returns a JAX array.  The MLP
    parameters are intended to be trained jointly with the Neural ODE.

    Args:
        in_size: Dimensionality of the input covariate vector.
        latent_dim: Dimensionality of the output latent vector.
        mlp_width: Width of hidden layers.
        mlp_depth: Number of hidden layers.
        activation: Activation function for the hidden layers.
        key: JAX PRNGKey for initialising parameters.
    """

    mlp: eqx.nn.MLP

    def __init__(
        self,
        in_size: int,
        latent_dim: int,
        mlp_width: int,
        mlp_depth: int,
        *,
        activation=jax.nn.swish,
        key: jax.Array,
    ):
        super().__init__()
        # The Equinox MLP API requires specifying in_size, out_size, width_size and depth
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=latent_dim,
            width_size=mlp_width,
            depth=mlp_depth,
            activation=activation,
            key=key,
        )

    def fit(self, X: np.ndarray) -> "MLPReducer":
        # Nothing to fit for a learnable reducer; parameters are randomly initialised
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Convert to JAX array and forward through the MLP without grad
        X_j = jnp.asarray(X, dtype=jnp.float32)
        return jax.vmap(self.mlp)(X_j).astype(jnp.float32)

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(self.mlp)(X)