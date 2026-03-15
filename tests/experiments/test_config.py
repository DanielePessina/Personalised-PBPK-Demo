"""Regression tests for public experiment config loading."""

from __future__ import annotations

from pathlib import Path

from experiments.config import load_experiment_config, load_yaml


def test_default_config_uses_public_analysis_names():
    """The default YAML should expose only the supported public analysis names."""
    root = Path(__file__).resolve().parents[2]
    payload = load_yaml(root / "configs" / "experiments" / "default.yaml")

    assert payload["analyses"]["default"] == ["bayesianopt", "hybrid", "node"]
    assert set(payload["model_config_paths"]) == {"bayesianopt", "differentialevo", "nlme", "hybrid", "node"}
    assert "neural_ode_eeg" not in payload["model_config_paths"]


def test_load_experiment_config_keeps_public_keys_and_existing_files():
    """Config loading should keep public keys while resolving the canonical YAML files."""
    config = load_experiment_config()

    assert config.analyses == ["bayesianopt", "hybrid", "node"]
    assert set(config.model_configs) == {"bayesianopt", "differentialevo", "nlme", "hybrid", "node"}
    assert config.model_configs["bayesianopt"]["source_config_path"].endswith("configs/experiments/bayesianopt.yaml")
    assert config.model_configs["differentialevo"]["source_config_path"].endswith(
        "configs/experiments/differentialevo.yaml"
    )
    assert config.model_configs["nlme"]["source_config_path"].endswith("configs/experiments/nlme_optax.yaml")
    assert config.model_configs["hybrid"]["source_config_path"].endswith(
        "configs/experiments/hybrid_fixed_hparams.yaml"
    )
    assert config.model_configs["node"]["source_config_path"].endswith(
        "configs/experiments/neural_ode_remifentanil.yaml"
    )
