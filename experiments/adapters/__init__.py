"""Adapter registry for experiment analyses."""

from .bayesianopt import BayesianOptimizationAdapter
from .base import AnalysisAdapter
from .differentialevo import DifferentialEvolutionAdapter
from .hybrid_fixed_hparams import HybridFixedHyperparametersAdapter
from .neural_ode_eeg import NeuralODEEEGAdapter
from .neural_ode_remifentanil import RemifentanilNODEAdapter
from .nlme_optax import NLMEOptaxAdapter

ADAPTER_REGISTRY: dict[str, type[AnalysisAdapter]] = {
    "bayesianopt": BayesianOptimizationAdapter,
    "differentialevo": DifferentialEvolutionAdapter,
    "nlme_optax": NLMEOptaxAdapter,
    "hybrid_fixed_hparams": HybridFixedHyperparametersAdapter,
    "neural_ode_remifentanil": RemifentanilNODEAdapter,
    "neural_ode_eeg": NeuralODEEEGAdapter,
}

__all__ = ["ADAPTER_REGISTRY", "AnalysisAdapter"]
