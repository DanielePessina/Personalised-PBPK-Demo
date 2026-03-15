# Pharmacokinetics Test Suite

This directory contains the maintained pytest suite for the remifentanil-focused
`pharmacokinetics` package.

## Active Test Modules

- `test_consistency.py`: package-surface and API regression checks
- `test_remifentanil.py`: dataset loading, physiological parameter construction,
  PBPK simulation, and batching helpers
- `test_gradients.py`: autodifferentiability of the separated solver
- `test_loss_gradients.py`: cohort loss functions used by estimation workflows
- `test_hybrid_preprocessing.py`: scaling and covariate preprocessing used by the
  hybrid models
- `test_remifentanil_node.py`: Neural ODE preprocessing utilities

`test_symbolic_regression.py` remains as placeholder scaffolding for future PySR
coverage and is not currently active.

## Running Tests

Use `uv` from the repository root:

```bash
uv run python -m pytest
```

Pytest is configured through `pytest.ini` to collect tests from this directory.

## Dataset Assumptions

The maintained tests expect the canonical remifentanil workbook at:

```text
nlme-remifentanil.xlsx
```

The test fixtures resolve that path relative to the repository root, so the tests
do not depend on the current working directory beyond being run inside this repo.
