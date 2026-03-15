"""Shared helpers for resolving showcase run directories and repo-local artifacts."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = REPO_ROOT / "nlme-remifentanil.xlsx"


def resolve_latest_analysis_run(
    *,
    analysis_name: str,
    run_dir: str | Path | None = None,
    results_root: str | Path = "results",
    display_name: str | None = None,
) -> Path:
    """Resolve an explicit or latest visible run directory for one analysis."""
    if run_dir is not None:
        resolved = Path(run_dir).resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"{display_name or analysis_name} run directory not found: {resolved}")
        return resolved

    root = Path(results_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Results directory not found: {root}")

    candidates: list[Path] = []
    for experiment_dir in sorted(root.iterdir()):
        if not experiment_dir.is_dir() or experiment_dir.name.startswith("."):
            continue
        analysis_dir = experiment_dir / analysis_name
        if not analysis_dir.is_dir():
            continue
        for candidate in sorted(analysis_dir.iterdir()):
            if candidate.is_dir() and not candidate.name.startswith("."):
                candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(f"No visible {display_name or analysis_name} runs found under {root}")
    return sorted(candidates, key=lambda path: path.name)[-1]


def resolve_repo_dataset_path(dataset_path: str | Path) -> Path:
    """Prefer the repo-local workbook over serialized machine-specific paths."""
    repo_candidate = DEFAULT_DATASET_PATH.resolve()
    if repo_candidate.is_file():
        return repo_candidate

    resolved = Path(dataset_path).expanduser().resolve()
    if resolved.is_file():
        return resolved

    basename_candidate = REPO_ROOT / Path(dataset_path).name
    if basename_candidate.is_file():
        return basename_candidate.resolve()

    raise FileNotFoundError(
        f"Dataset not found. Checked repo-local workbook {repo_candidate} and serialized path {resolved}."
    )
