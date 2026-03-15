# PersonalisedPBPK

This repository is a remifentanil-focused PBPK showcase built around JAX-based
mechanistic simulation, curated benchmark results, a standalone GSAX sensitivity
analysis workflow, and a marimo notebook for presentation.

The public workflows in this repository are:

- `run_experiments.py` for the maintained benchmark analyses
- `run_gsa.py` for GSAX sensitivity analysis on a saved hybrid run
- `notebook_remifentanil.py` for the marimo showcase notebook

The repository already includes the main workbook, curated demo result bundles,
and the code needed to reproduce the showcased workflows.

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for Python environment and
dependency management. If you do not have `uv` installed, follow the official
installation instructions for your platform.

Once `uv` is available, clone the repository and sync the environment:

```bash
git clone <repository-url>
cd PersonalisedPBPK
uv sync --all-groups
```

This installs all dependencies including the marimo notebook runner, JAX-based
simulation tools, and test dependencies.

## Run The Marimo Notebook

The showcase notebook visualises the remifentanil benchmark results using the
curated data bundles in `results/`.

### App mode

Run the notebook as a static web application:

```bash
uv run marimo run notebook_remifentanil.py
```

### Edit mode

Run the notebook with an editable interface:

```bash
uv run marimo edit notebook_remifentanil.py
```

### Static HTML export

A pre-rendered static HTML file is provided for viewing the notebook output
without running any computation:

```bash
open notebook_remifentanil.html
```

Or open `notebook_remifentanil.html` directly in your browser. This file
contains the complete notebook output (tables, plots, and analysis) from the
curated result bundles.

## Repository Layout

```text
pharmacokinetics/               Remifentanil PBPK and NODE code
experiments/                    Experiment runner, configs, adapters, and bundle I/O
configs/experiments/            YAML configs for the maintained analyses
notebook_support/               Helpers used by the marimo notebook
results/                        Curated demo bundles for the showcase workflows
tests/                          Regression tests for runners and notebook loaders
run_experiments.py              Public benchmark entrypoint
run_gsa.py                      Public GSAX sensitivity-analysis entrypoint
notebook_remifentanil.py        Marimo showcase notebook
nlme-remifentanil.xlsx          Canonical local workbook used by the workflows
```

## Run The Benchmark Analyses

The default `run_experiments.py` configuration runs the showcased analyses
`bayesianopt`, `hybrid`, and `node`. You can also run them separately:

```bash
uv run python run_experiments.py --analysis bayesianopt
uv run python run_experiments.py --analysis hybrid
uv run python run_experiments.py --analysis node
```

Persisted outputs are written under:

```text
results/remifentanil_benchmark/<analysis>/<run_id>/
```

where the canonical analysis directories are:

- `bayesianopt`
- `hybrid_fixed_hparams`
- `neural_ode_remifentanil`

## Run The GSAX Analysis

`run_gsa.py` performs a standalone GSAX Sobol analysis using a saved hybrid
bundle as the source model. The curated showcase bundle uses:

```text
results/remifentanil_benchmark/hybrid_fixed_hparams/20260313T160742Z
```

Run it with:

```bash
uv run python run_gsa.py --source-run-dir results/remifentanil_benchmark/hybrid_fixed_hparams/20260313T160742Z
```

The GSAX outputs are written under:

```text
results/remifentanil_benchmark/gsax_hybrid_gsa/<run_id>/
```

## Additional Model Directions

Beyond the showcased BO, hybrid, and remifentanil NODE benchmarks, the repository
also contains:

- `differentialevo`: a differential-evolution parameter-estimation workflow
- `nlme_optax`: a nonlinear mixed-effects remifentanil model
- `neural_ode_eeg`: a Neural ODE direction that uses a time-dependent,
  EEG-derived signal to improve predictions

## Testing

Run the maintained test suites with:

```bash
uv run python -m pytest tests/experiments
uv run python -m pytest tests/notebook_support
uv run python -m pytest tests/test_run_gsa.py
uv run python -m pytest pharmacokinetics/tests
```

## Model Scope

The mechanistic remifentanil model follows Abbiati et al.:

- Abbiati, R. A.; Lamberti, G.; Grassi, M.; Trotta, F.; Manca, D.
  Definition and Validation of a Patient-Individualized Physiologically-Based
  Pharmacokinetic Model. *Computers & Chemical Engineering* 2016, 84, 394-408.
  [DOI](https://doi.org/10.1016/j.compchemeng.2015.09.018)
