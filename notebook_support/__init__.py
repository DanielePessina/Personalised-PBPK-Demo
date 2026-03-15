"""Notebook support helpers for remifentanil exploratory analysis."""

from .bayesianopt_results import (
    load_bayesianopt_results,
    plot_bayesianopt_double_cv_parity,
    resolve_latest_bayesianopt_run,
)
from .remifentanil_eda import compute_patient_summary, load_patient_covariate_frame, plot_covariate_corner
from .hybrid_results import load_hybrid_results, plot_hybrid_double_cv_parity, resolve_latest_hybrid_run
from .gsax_results import load_gsax_results, resolve_latest_gsax_run
from .node_results import load_node_results, plot_node_double_cv_parity, resolve_latest_node_run
from .style import use_mpl_style

__all__ = [
    "load_bayesianopt_results",
    "load_gsax_results",
    "load_hybrid_results",
    "load_node_results",
    "compute_patient_summary",
    "load_patient_covariate_frame",
    "plot_bayesianopt_double_cv_parity",
    "plot_hybrid_double_cv_parity",
    "plot_node_double_cv_parity",
    "plot_covariate_corner",
    "resolve_latest_bayesianopt_run",
    "resolve_latest_gsax_run",
    "resolve_latest_hybrid_run",
    "resolve_latest_node_run",
    "use_mpl_style",
]
