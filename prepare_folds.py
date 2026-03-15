"""CLI entrypoint for preparing canonical experiment folds."""

from __future__ import annotations

from experiments import run


if __name__ == "__main__":
    raise SystemExit(run(["--prepare-folds-only"]))
