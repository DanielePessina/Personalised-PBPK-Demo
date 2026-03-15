"""CLI entrypoint for the experiment orchestration layer."""

from __future__ import annotations

from experiments import run


if __name__ == "__main__":
    raise SystemExit(run())
