"""Public analysis names exposed by the experiment runner."""

from __future__ import annotations

from typing import Final

PUBLIC_TO_CANONICAL_ANALYSIS_NAME: Final[dict[str, str]] = {
    "bayesianopt": "bayesianopt",
    "differentialevo": "differentialevo",
    "nlme": "nlme_optax",
    "hybrid": "hybrid_fixed_hparams",
    "node": "neural_ode_remifentanil",
}
PUBLIC_ANALYSIS_NAMES: Final[tuple[str, ...]] = tuple(PUBLIC_TO_CANONICAL_ANALYSIS_NAME)


def resolve_public_analysis_name(public_name: str) -> str:
    """Map a public analysis name to the canonical internal adapter identifier."""
    try:
        return PUBLIC_TO_CANONICAL_ANALYSIS_NAME[public_name]
    except KeyError as exc:
        expected = ", ".join(PUBLIC_ANALYSIS_NAMES)
        raise KeyError(f"Unknown public analysis '{public_name}'. Expected one of: {expected}.") from exc
