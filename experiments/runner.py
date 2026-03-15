"""Experiment runner orchestrating summary, folds, adapters, and bundles."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess
import time
from typing import Any

import pandas as pd
from rich.panel import Panel

from .cli import build_parser, make_console
from .config import load_experiment_config
from .analysis_names import resolve_public_analysis_name
from .folds import create_fold_artifact, save_fold_artifact, validate_fold_artifact
from .results import BundleWriter
from .schemas import ManifestStatus, RunManifest
from .summary import compute_dataset_summary, dataset_summary_table, load_remifentanil_dataset
from .validate import find_analysis_bundles, validate_bundle
from .adapters import ADAPTER_REGISTRY


def _git_revision() -> str:
    """Return the current git revision, or 'unknown' if unavailable."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return completed.stdout.strip()


def _default_fold_path(output_root: Path, experiment_name: str) -> Path:
    """Return the canonical fold artifact path."""
    return output_root / experiment_name / "folds.yaml"


def _validate_existing_bundles(
    output_root: Path,
    experiment_name: str,
    analyses: list[tuple[str, str]],
) -> int:
    """Validate previously saved bundles under the configured output root."""
    console = make_console()
    failures = 0
    for public_analysis_name, canonical_analysis_name in analyses:
        analysis_root = output_root / experiment_name / canonical_analysis_name
        bundles = find_analysis_bundles(analysis_root)
        if not bundles:
            console.print(
                f"[red]No bundles found for analysis '{public_analysis_name}' in {analysis_root}.[/red]"
            )
            failures += 1
            continue
        for bundle in bundles:
            issues = validate_bundle(bundle)
            if issues:
                failures += 1
                console.print(Panel("\n".join(issues), title=f"Validation failed: {bundle.name}", border_style="red"))
            else:
                console.print(f"[green]Validated bundle:[/green] {bundle}")
    return 0 if failures == 0 else 1


def run(argv: list[str] | None = None) -> int:
    """Execute the configured experiment workflow from the CLI."""
    experiment_config = load_experiment_config()
    parser = build_parser()
    args = parser.parse_args(argv)
    console = make_console()

    experiment_name = args.experiment_name or experiment_config.experiment_name
    output_root = Path(args.output_dir or experiment_config.output_root).resolve()
    selected_public_analyses = args.analysis or experiment_config.analyses
    resolved_analyses = [
        (public_analysis_name, resolve_public_analysis_name(public_analysis_name))
        for public_analysis_name in selected_public_analyses
    ]

    if args.validate_only:
        return _validate_existing_bundles(output_root, experiment_name, resolved_analyses)

    dataset = load_remifentanil_dataset(experiment_config.dataset_path)
    fold_path = Path(args.folds_file).resolve() if args.folds_file else _default_fold_path(output_root, experiment_name)
    fold_artifact = create_fold_artifact(
        experiment_name=experiment_name,
        dataset=dataset,
        outer_fold_count=experiment_config.split.outer_folds,
        outer_seed=experiment_config.split.outer_seed,
    )
    save_fold_artifact(fold_artifact, fold_path)

    fold_errors = validate_fold_artifact(fold_artifact, dataset)
    if fold_errors:
        raise RuntimeError(f"Invalid fold artifact: {fold_errors}")

    summary = compute_dataset_summary(dataset, fold_artifact)
    console.print(dataset_summary_table(summary))

    if args.prepare_folds_only:
        console.print(f"[green]Saved folds to[/green] {fold_path}")
        return 0

    git_revision = _git_revision()
    now = datetime.now(timezone.utc)

    for public_analysis_name, canonical_analysis_name in resolved_analyses:
        if canonical_analysis_name not in ADAPTER_REGISTRY:
            raise KeyError(f"Unknown canonical analysis '{canonical_analysis_name}'.")
        if public_analysis_name not in experiment_config.model_configs:
            raise KeyError(f"No model config registered for public analysis '{public_analysis_name}'.")
        run_id = now.strftime("%Y%m%dT%H%M%SZ")
        writer = BundleWriter(
            output_root=output_root,
            experiment_name=experiment_name,
            analysis=canonical_analysis_name,
            run_id=run_id,
        )
        manifest = RunManifest(
            experiment_name=experiment_name,
            analysis=canonical_analysis_name,
            run_id=run_id,
            config_version=experiment_config.version,
            dataset_path=str(dataset.dataset_path),
            dataset_fingerprint=dataset.dataset_fingerprint,
            fold_file=str(fold_path),
            code_version=git_revision,
            status=ManifestStatus(
                status="running",
                started_at=now.isoformat(),
                completed_at=None,
                wall_clock_seconds=None,
            ),
        )
        analysis_config: dict[str, Any] = {
            **experiment_config.model_configs[public_analysis_name],
            "experiment_name": experiment_name,
            "run_id": run_id,
            "public_analysis_name": public_analysis_name,
            "canonical_analysis_name": canonical_analysis_name,
        }
        writer.write_config(
            {
                "version": experiment_config.version,
                "experiment_name": experiment_name,
                "analysis": canonical_analysis_name,
                "resolved_config": analysis_config,
            }
        )
        writer.write_dataset_summary(summary)
        writer.write_folds(fold_artifact)
        writer.write_manifest(manifest)

        adapter = ADAPTER_REGISTRY[canonical_analysis_name]()
        started = time.perf_counter()
        console.print(
            Panel.fit(
                f"Running analysis: {public_analysis_name} ({canonical_analysis_name})",
                border_style="cyan",
            )
        )
        try:
            for fold_spec in fold_artifact.folds:
                fold_started = time.perf_counter()
                prepared_inputs = adapter.prepare_inputs(dataset, fold_spec, analysis_config)
                trained_state = adapter.fit(prepared_inputs, analysis_config, console.log)
                writer.save_fold_metadata(
                    fold_spec.fold_index,
                    {
                        "analysis": canonical_analysis_name,
                        "fold_index": fold_spec.fold_index,
                    },
                )
                writer.save_fold_history(fold_spec.fold_index, getattr(trained_state, "history", []))
                metrics_frames: list[pd.DataFrame] = []
                for split_name in ("train", "test"):
                    frame = adapter.predict(trained_state, split_name)
                    metrics_result = adapter.evaluate(frame)
                    writer.save_fold_predictions(fold_spec.fold_index, split_name, frame)
                    writer.write_metrics(fold_spec.fold_index, split_name, metrics_result)
                    metrics_frames.append(metrics_result.assign(outer_fold=fold_spec.fold_index))
                writer.save_fold_metrics(
                    fold_spec.fold_index,
                    pd.concat(metrics_frames, ignore_index=True),
                )
                writer.save_fold_status(
                    fold_spec.fold_index,
                    {
                        "status": "completed",
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "failure_reason": None,
                        "wall_clock_seconds": time.perf_counter() - fold_started,
                    },
                )
                adapter.save_artifacts(trained_state, writer)
        except Exception as exc:
            manifest.status.status = "failed"
            manifest.status.failure_reason = str(exc)
            manifest.status.completed_at = datetime.now(timezone.utc).isoformat()
            manifest.status.wall_clock_seconds = time.perf_counter() - started
            writer.write_manifest(manifest)
            writer.save_fold_status(
                fold_spec.fold_index,
                {
                    "status": "failed",
                    "failure_reason": str(exc),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "wall_clock_seconds": time.perf_counter() - fold_started,
                },
            )
            raise

        manifest.status.status = "completed"
        manifest.status.completed_at = datetime.now(timezone.utc).isoformat()
        manifest.status.wall_clock_seconds = time.perf_counter() - started
        writer.write_manifest(manifest)

        bundle_errors = validate_bundle(writer.bundle_dir)
        if bundle_errors:
            raise RuntimeError(
                f"Bundle validation failed for {public_analysis_name} ({canonical_analysis_name}): {bundle_errors}"
            )

    return 0
