"""Fail-fast EEG Neural ODE adapter placeholder."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .base import AnalysisAdapter


class NeuralODEEEGAdapter(AnalysisAdapter):
    """Adapter that fails clearly until the EEG inputs are fully wired."""

    analysis_name = "neural_ode_eeg"

    def prepare_inputs(self, dataset: Any, fold_spec: Any, config: dict[str, Any]) -> Any:
        expected_path = Path(config["dataset"]["path"]).resolve()
        if not expected_path.exists():
            raise RuntimeError(
                f"EEG dataset not found at {expected_path}. The neural_ode_eeg adapter is not yet runnable."
            )
        raise RuntimeError(
            "The neural_ode_eeg adapter has not been implemented against a stable EEG training interface yet."
        )

    def fit(self, model_inputs: Any, config: dict[str, Any], reporter: Callable[[str], None]) -> Any:
        raise RuntimeError("The neural_ode_eeg adapter cannot fit without a runnable EEG data pipeline.")

    def predict(self, trained_state: Any, split_name: str) -> pd.DataFrame:
        raise RuntimeError("The neural_ode_eeg adapter cannot produce predictions yet.")

    def evaluate(self, long_predictions: pd.DataFrame) -> pd.DataFrame:
        raise RuntimeError("The neural_ode_eeg adapter cannot compute metrics yet.")

    def save_artifacts(self, trained_state: Any, bundle_writer: Any) -> None:
        raise RuntimeError("The neural_ode_eeg adapter cannot save artifacts yet.")
