"""Regression tests for adapter registry and contract surface."""

from __future__ import annotations

import pytest

from experiments.analysis_names import PUBLIC_ANALYSIS_NAMES, resolve_public_analysis_name
from experiments.adapters import ADAPTER_REGISTRY
from experiments.adapters.neural_ode_eeg import NeuralODEEEGAdapter


def test_public_analysis_names_stay_stable():
    """The public experiment runner names should stay stable."""
    assert PUBLIC_ANALYSIS_NAMES == ("bayesianopt", "differentialevo", "nlme", "hybrid", "node")
    assert {resolve_public_analysis_name(name) for name in PUBLIC_ANALYSIS_NAMES} <= set(ADAPTER_REGISTRY)


def test_internal_adapter_registry_retains_private_eeg_adapter():
    """The internal registry can keep non-public adapters."""
    assert "neural_ode_eeg" in ADAPTER_REGISTRY


@pytest.mark.parametrize("analysis_name", sorted(ADAPTER_REGISTRY))
def test_adapter_contract_methods_exist(analysis_name: str):
    """Each adapter should implement the shared contract methods."""
    adapter = ADAPTER_REGISTRY[analysis_name]()
    for method_name in ("prepare_inputs", "fit", "predict", "evaluate", "save_artifacts"):
        assert hasattr(adapter, method_name)
        assert callable(getattr(adapter, method_name))


def test_eeg_adapter_fails_with_precise_message(tmp_path):
    """The EEG adapter should fail fast until its data pipeline is wired."""
    adapter = NeuralODEEEGAdapter()
    with pytest.raises(RuntimeError, match="EEG dataset not found"):
        adapter.prepare_inputs(None, None, {"dataset": {"path": str(tmp_path / "missing.csv")}})
