"""Configuration loading for experiment orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .analysis_names import PUBLIC_ANALYSIS_NAMES
from .schemas import ExperimentConfig, SplitConfig


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a block-style YAML file into a dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping at {path}, found {type(payload).__name__}.")
    return payload


def load_experiment_config(config_path: Path | None = None) -> ExperimentConfig:
    """Load the default experiment configuration and per-model YAML files."""
    root = Path(__file__).resolve().parents[1]
    config_dir = root / "configs" / "experiments"
    default_path = config_dir / "default.yaml" if config_path is None else config_path
    payload = load_yaml(default_path)

    configured_analyses = [str(value) for value in payload["analyses"]["default"]]
    invalid_default_analyses = sorted(set(configured_analyses) - set(PUBLIC_ANALYSIS_NAMES))
    if invalid_default_analyses:
        raise ValueError(f"Unsupported public analyses in default config: {invalid_default_analyses}")

    model_config_paths = {str(name): str(path) for name, path in payload["model_config_paths"].items()}
    invalid_model_config_names = sorted(set(model_config_paths) - set(PUBLIC_ANALYSIS_NAMES))
    if invalid_model_config_names:
        raise ValueError(f"Unsupported public model config keys: {invalid_model_config_names}")

    missing_model_config_names = sorted(set(configured_analyses) - set(model_config_paths))
    if missing_model_config_names:
        raise ValueError(f"Missing model configs for public analyses: {missing_model_config_names}")

    model_configs: dict[str, dict[str, Any]] = {}
    for analysis_name, relative_path in model_config_paths.items():
        model_path = (root / relative_path).resolve()
        model_payload = load_yaml(model_path)
        model_payload["source_config_path"] = str(model_path)
        model_configs[analysis_name] = model_payload

    split_config = SplitConfig(
        outer_folds=int(payload["split"]["outer_folds"]),
        outer_seed=int(payload["split"]["outer_seed"]),
    )
    return ExperimentConfig(
        version=int(payload["version"]),
        experiment_name=str(payload["experiment_name"]),
        dataset_path=str((root / payload["dataset"]["path"]).resolve()),
        output_root=str((root / payload["output"]["root_dir"]).resolve()),
        analyses=configured_analyses,
        split=split_config,
        model_configs=model_configs,
    )
