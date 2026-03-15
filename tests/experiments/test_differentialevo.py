"""Regression tests for Differential Evolution helper behavior."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from experiments.adapters.differentialevo import (
    _decode_population,
    _evaluate_candidate_population,
    _parameter_spec_from_config,
    _stack_physio_patients,
    _weighted_negative_log_likelihood_scalar,
)


@pytest.mark.parametrize("jit_enabled", [False, True])
def test_population_objective_matches_scalar_weighted_nll(experiment_dataset, jit_enabled: bool) -> None:
    physio_patients = list(experiment_dataset.physio_by_id.values())[:3]
    stacked_physio = _stack_physio_patients(physio_patients)
    parameter_spec = _parameter_spec_from_config(_differentialevo_config())
    normalized_population = jnp.asarray(
        [
            [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.20, 0.70],
            [0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05],
        ],
        dtype=jnp.float64,
    )

    batched = np.asarray(
        _evaluate_candidate_population(
            normalized_population,
            stacked_physio,
            parameter_spec,
            0.1,
            candidate_batch_size=1,
            jit_enabled=jit_enabled,
        ),
        dtype=np.float64,
    )
    decoded = np.asarray(_decode_population(normalized_population, parameter_spec), dtype=np.float64)
    scalar = np.asarray(
        [
            _weighted_negative_log_likelihood_scalar(candidate, physio_patients, 0.1)
            for candidate in decoded
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(batched, scalar, rtol=1e-6, atol=1e-6)


def test_parameter_spec_requires_canonical_parameter_order() -> None:
    config = _differentialevo_config()
    config["parameters"][0]["name"] = "k_PT"

    try:
        _parameter_spec_from_config(config)
    except ValueError as exc:
        assert "KINETIC_PARAMETER_NAMES order" in str(exc)
    else:
        raise AssertionError("Expected ValueError for parameter-order mismatch.")


def _differentialevo_config() -> dict:
    return {
        "parameters": [
            {"name": "k_TP", "low": 1e-4, "high": 2.0, "scale": "linear"},
            {"name": "k_PT", "low": 1e-4, "high": 2.0, "scale": "linear"},
            {"name": "k_PHP", "low": 1e-4, "high": 2.0, "scale": "linear"},
            {"name": "k_HPP", "low": 1e-4, "high": 2.0, "scale": "linear"},
            {"name": "k_EL_Pl", "low": 1e-4, "high": 30.0, "scale": "linear"},
            {"name": "Eff_kid", "low": 1e-4, "high": 1.0, "scale": "linear"},
            {"name": "Eff_hep", "low": 1e-4, "high": 0.5, "scale": "linear"},
            {"name": "k_EL_Tis", "low": 1e-4, "high": 3.0, "scale": "linear"},
        ]
    }
