"""CLI setup for the experiment runner."""

from __future__ import annotations

import argparse

from rich.console import Console

from .analysis_names import PUBLIC_ANALYSIS_NAMES


def build_parser() -> argparse.ArgumentParser:
    """Build the public experiment runner CLI."""
    parser = argparse.ArgumentParser(description="Run remifentanil experiment benchmarks.")
    parser.add_argument(
        "--analysis",
        nargs="+",
        choices=PUBLIC_ANALYSIS_NAMES,
        help="Run only the selected public analyses.",
    )
    parser.add_argument("--experiment-name", default=None, help="Override the configured experiment name.")
    parser.add_argument("--output-dir", default=None, help="Override the configured output directory.")
    parser.add_argument("--folds-file", default=None, help="Load or save folds at a specific path.")
    parser.add_argument("--prepare-folds-only", action="store_true", help="Prepare and save folds, then exit.")
    parser.add_argument("--validate-only", action="store_true", help="Validate existing result bundles, then exit.")
    return parser


def make_console() -> Console:
    """Create the CLI console used for rich output."""
    return Console(width=120)
