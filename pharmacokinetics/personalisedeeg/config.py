"""
Configuration dataclasses for the remifentanil neural ODE project.

These dataclasses hold hyperparameters and settings for the various
components of the training pipeline.  They are intentionally kept
separate from the implementation code to decouple configuration from
behaviour and to make it easy to switch between different model
architectures or training regimes.  The names and default values
mirror those in the provided context files where appropriate.

Use :class:`ModelConfig` to configure the ODE architecture, including
the number of augmented dimensions and the depth and width of the vector field
MLP.  The `include_time` flag determines whether the current time
should be concatenated onto the MLP input at each integration step,
and `constrained` can be set to enforce monotonicity on the first
output dimension via a softplus transform.

:class:`TrainingConfig` governs the training loop itself.  It follows
a simple curriculum of two phases: each phase has its own learning
rate, number of optimisation steps and fraction of the sequence to
expose to the ODE solver.  The batch size, random seed and verbosity
level can also be tuned here.

:class:`DimReductionConfig` describes how to compress the patient
covariates and dosing metadata into a low‑dimensional latent vector. The
`latent_dim` specified here is the single source of truth for the latent
vector size and therefore also the size of the constant block appended to
the NODE state. The `method` field accepts ``"pca"``, ``"tsne"``, or
``"mlp"``. The MLP reduction is implemented as an Equinox MLP whose width
and depth are specified via `mlp_width` and `mlp_depth`.

:class:`DataSplitConfig` defines how to split the cohort of patients
into training, validation and test sets.  By default it performs
a 70/15/15 split based on unique patient identifiers, but the
fractions and random seed can be changed as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Optional
import diffrax
import jax.nn as jnn


@dataclass
class ModelConfig:
    """Configuration for the neural ODE architecture.

    Attributes:
        augment_dims: Number of learnable auxiliary state dimensions to
            append to the dynamic state.  These dimensions are updated
            by the vector field and can capture unmodelled dynamics.
        mlp_width: Width of the hidden layers in the vector field MLP.
        mlp_depth: Number of hidden layers in the vector field MLP.
        include_time: Whether to concatenate the current time to the
            MLP input.  Including time can help the model learn
            non‑autonomous dynamics.
        add_concentration: Whether to include drug concentration as an
            additional input to the MLP. This enables PK/PD integration.
        constrained: If ``True``, the derivative of the first state
            component (the EEG signal) is transformed via a softplus to
            enforce negative drift.  This mirrors the practice in the
            provided context code to model monotonic decay.
        activation: Activation function used in the MLP.  Defaults to
            the ``swish`` nonlinearity.
        solver: Diffrax ODE solver used for integration.  Defaults to
            Tsit5, a reasonably good general‑purpose solver.
    """

    augment_dims: int = 1
    mlp_width: int = 100
    mlp_depth: int = 4
    include_time: bool = True
    add_concentration: bool = False
    constrained: bool = False
    activation: Callable = jnn.swish
    solver: diffrax.AbstractSolver = diffrax.Dopri8()


@dataclass
class TrainingConfig:
    """Configuration for the training loop.

    The training proceeds in phases, each with its own learning rate,
    number of optimisation steps and fraction of the sequence to use.
    This simple curriculum allows the model to first learn on shorter
    sequences before being exposed to the full data.

    Attributes:
        lr_strategy: Tuple of learning rates, one per phase.
        steps_strategy: Tuple of number of optimisation steps per phase.
        length_strategy: Tuple of fractions of the maximum sequence
            length to expose during each phase.  Values should be in
            (0, 1]; a value of 0.5 means only the first half of each
            sequence is used in that phase.
        batch_size: Mini‑batch size.  A value of -1 forces full batch
            training.
        verbose: If ``True``, prints a progress bar with loss values.
        print_every: Unused here but kept for parity with the context
            code; could be used to log intermediate metrics.
        seed: PRNG seed controlling initialisation and data shuffling.
    """

    lr_strategy: tuple[float, float] = (1e-3, 1e-4)
    steps_strategy: tuple[int, int] = (500, 500)
    length_strategy: tuple[float, float] = (0.5, 1.0)
    batch_size: int = 32
    verbose: bool = True
    print_every: int = 100
    seed: int = 42


@dataclass
class DimReductionConfig:
    """Configuration for covariate dimensionality reduction.

    The method field selects between non‑learnable reduction (PCA or t‑SNE)
    and a learnable MLP. The `latent_dim` here is the output dimensionality
    of the reducer and must match the size of the constant block appended to
    the NODE state.
    """

    method: str = "mlp"  # 'pca', 'tsne' or 'mlp'
    latent_dim: int = 3
    # Parameters for t‑SNE
    tsne_perplexity: float = 30.0
    tsne_learning_rate: float = 200.0
    tsne_n_iter: int = 1000
    # Parameters for MLP reducer
    mlp_width: int = 64
    mlp_depth: int = 2


@dataclass
class DataSplitConfig:
    """Configuration for splitting the cohort into train, validation and test sets.

    The splits are applied on unique patient IDs.  Fractions must sum
    to 1.0 and each be positive.  The random seed ensures
    reproducibility of the shuffle.
    """

    train_frac: float = 0.7
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 0

    def __post_init__(self):
        total = self.train_frac + self.val_frac + self.test_frac
        assert abs(total - 1.0) < 1e-6, "Split fractions must sum to 1.0"
        assert self.train_frac > 0 and self.val_frac >= 0 and self.test_frac >= 0, (
            "Split fractions must be non‑negative and the training fraction must be positive"
        )


@dataclass
class PBPKConfig:
    """Configuration for PBPK model integration.

    Attributes:
        add_concentration: If ``True``, the PK concentration will be
            added to the NODE input.
        nlme_model_path: Path to the saved NLME model file.
        dense_sim_points: Number of points for dense PK simulation.
        nlme_kinetics_dict: Dictionary mapping patient IDs to their kinetic parameters.
    """
    add_concentration: bool = False
    nlme_model_path: str | None = None
    dense_sim_points: int = 200
    nlme_kinetics_dict: dict = None
