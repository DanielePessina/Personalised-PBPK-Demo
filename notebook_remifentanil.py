import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd

    from notebook_support import (
        compute_patient_summary,
        load_bayesianopt_results as load_aggregated_bayesianopt_results,
        load_gsax_results as load_aggregated_gsax_results,
        load_hybrid_results as load_aggregated_hybrid_results,
        load_node_results as load_aggregated_node_results,
        load_patient_covariate_frame,
        # load_nlme_results as load_aggregated_nlme_results,
        plot_bayesianopt_double_cv_parity,
        plot_covariate_corner,
        plot_hybrid_double_cv_parity,
        # plot_nlme_double_cv_parity,
        plot_node_double_cv_parity,
        use_mpl_style,
    )

    return (
        Path,
        compute_patient_summary,
        load_aggregated_bayesianopt_results,
        load_aggregated_gsax_results,
        load_aggregated_hybrid_results,
        load_aggregated_node_results,
        load_patient_covariate_frame,
        mo,
        np,
        pd,
        plot_bayesianopt_double_cv_parity,
        plot_covariate_corner,
        plot_hybrid_double_cv_parity,
        plot_node_double_cv_parity,
        use_mpl_style,
    )


@app.cell
def _(use_mpl_style):
    use_mpl_style()
    return


@app.cell
def _(mo):
    mo.md("""
    # Remifentanil dataset overview

    This notebook starts from the Remifentanil example dataset documented on the
    [MonolixSuite page](https://monolixsuite.slp-software.com/monolix/2024R1/remifentanil-data-set).
    The source cohort contains **65 healthy adults** who received remifentanil by
    continuous IV infusion at a constant rate, with dense remifentanil blood
    concentration measurements for the PK component and dense EEG measurements for
    the PD component.

    The Monolix description lists the source covariates `AGE`, `SEX`, `LBM`, and
    `TINFCAT`. In this repository, the active experiment workflow reads the curated
    local workbook `nlme-remifentanil.xlsx`, which carries patient-level covariates
    as `Age`, `Wt`, `Ht`, `Sex`, and `BSA`. The exploratory summary and plots below
    therefore use the local workbook shape consumed by `run_experiments.py`.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## The PBPK Model

    The mechanistic backbone used throughout this notebook is the remifentanil
    minimal-PBPK model implemented in `pharmacokinetics.remifentanil`, with the
    main source code living in `pharmacokinetics/remifentanil.py`. This module
    contains the full modelling pipeline from curated clinical records to
    simulation-ready physiological objects: `import_patients(...)` reads the local
    remifentanil workbook into `RawPatient` records, and
    `create_physiological_parameters(...)` transforms each subject into a
    `PhysiologicalParameters` object that couples subject anatomy, flows, dosing,
    and measurement grids to the PBPK solver.

    The underlying ODE right-hand side is implemented by
    `class _RemifentanilODE(eqx.Module)`. In the current code, the state vector has
    **11 states**: 8 concentration-like states and 3 cumulative elimination
    states. The dynamic compartments include gastric lumen, small-intestine lumen,
    large-intestine lumen, plasma, poorly perfused tissues, GI solid compartment,
    liver, and highly perfused organs. Mechanistically, remifentanil is infused
    directly into the **plasma** compartment, exchanged with the tissue
    compartments, and cleared through plasma, liver, and tissue-related
    elimination pathways. This is the key scientific architecture used throughout
    the notebook: a compact physiology-informed transport model with explicit
    compartment coupling and dose-duration-aware infusion switching, providing
    mass-balance guarantees, interpretable parameters, and a physics-grounded
    scaffold that can be extended or hybridised with data-driven components
    while retaining physiological plausibility.

    The patient-specific physiological quantities, such as organ volumes,
    cardiac-output-derived flows, sex-dependent fractions, and dosing metadata, are
    assembled in `create_physiological_parameters(...)`, while the 8 kinetic
    parameters remain explicit and separately identifiable through
    `get_default_parameters()` / `get_abbiati_parameters()`. Numerically, the ODE
    is solved with Diffrax in `_simulate_single_patient_separated(...)`, which
    constructs a `diffrax.ODETerm`, uses a `Tsit5` integrator, and inserts a jump
    at `dose_duration` so the IV infusion is switched off cleanly. At the package
    level, `pharmacokinetics/__init__.py` exposes this mechanistic workflow as
    `pharmacokinetics.remifentanil`, alongside `pharmacokinetics.remifentanil_node`
    for the Neural ODE workflow and `pharmacokinetics.nlme` for the mixed-effects
    utilities.
    """)
    return


@app.cell
def _(Path, compute_patient_summary, load_patient_covariate_frame):
    dataset_path = Path("nlme-remifentanil.xlsx")
    patient_frame = load_patient_covariate_frame(dataset_path)
    patient_summary = compute_patient_summary(patient_frame)
    return patient_frame, patient_summary


@app.cell
def _(mo, patient_summary, pd):
    summary_table = pd.DataFrame(
        [
            {"metric": "Patients", "value": patient_summary["patient_count"]},
            {"metric": "Average sampled points", "value": f"{patient_summary['average_sampled_points']:.2f}"},
            {
                "metric": "Average final measurement time",
                "value": f"{patient_summary['average_final_measurement_time']:.2f} min",
            },
            {"metric": "Average age", "value": f"{patient_summary['average_age']:.2f} years"},
            {"metric": "Average weight", "value": f"{patient_summary['average_weight']:.2f} kg"},
            {"metric": "Average height", "value": f"{patient_summary['average_height']:.2f} cm"},
            {"metric": "Average BSA", "value": f"{patient_summary['average_bsa']:.3f} m^2"},
        ]
    )
    sex_table = pd.DataFrame([{"sex": sex, "count": count} for sex, count in patient_summary["sex_counts"].items()])

    mo.vstack(
        [
            mo.md("## Cohort summary"),
            summary_table,
            mo.md("### Sex breakdown"),
            sex_table,
        ]
    )
    return


@app.cell
def _(patient_frame, plot_covariate_corner):
    corner_grid = plot_covariate_corner(patient_frame)
    corner_grid.figure
    return


@app.cell
def _(mo, pd):
    model_family_table = pd.DataFrame(
        [
            {
                "analysis": "nlme_optax",
                "type": "Population PK / NLME",
                "status": "Hidden for now",
            },
            {
                "analysis": "hybrid_fixed_hparams",
                "type": "PBPK + neural surrogate",
                "status": "Implemented below",
            },
            {
                "analysis": "neural_ode_remifentanil",
                "type": "Remifentanil Neural ODE",
                "status": "Implemented below",
            },
        ]
    )
    mo.vstack(
        [
            mo.md(
                """
                ## Model families in the saved experiments

                The current `results/` directory contains the Optax-trained NLME model,
                the population-only Bayesian optimisation workflow, the fixed-hyperparameter
                hybrid PBPK model, and the remifentanil Neural ODE. The visible notebook
                sections below focus on Bayesian optimisation, the hybrid model, and NODE.
                """
            ),
            model_family_table,
        ]
    )
    return


@app.cell
def _(load_aggregated_bayesianopt_results):
    def load_bayesianopt_results(run_dir=None):
        return load_aggregated_bayesianopt_results(run_dir=run_dir)

    return (load_bayesianopt_results,)


@app.cell
def _(load_aggregated_hybrid_results):
    def load_hybrid_results(run_dir=None):
        return load_aggregated_hybrid_results(run_dir=run_dir)

    return (load_hybrid_results,)


@app.cell
def _(load_aggregated_gsax_results):
    def load_gsax_results(run_dir=None):
        return load_aggregated_gsax_results(run_dir=run_dir)

    return (load_gsax_results,)


@app.cell
def _(load_bayesianopt_results):
    bayesianopt_results = load_bayesianopt_results()
    return (bayesianopt_results,)


@app.cell
def _(load_hybrid_results):
    hybrid_results = load_hybrid_results()
    return (hybrid_results,)


@app.cell
def _(load_gsax_results):
    gsa_results = load_gsax_results()
    return (gsa_results,)


@app.cell
def _(Path, load_aggregated_node_results):
    baseline_node_run_dir = Path("results/remifentanil_benchmark/neural_ode_remifentanil/20260314T154219Z")
    constrained_node_run_dir = Path("results/remifentanil_benchmark/neural_ode_remifentanil/20260314T154829Z")
    baseline_node_results = load_aggregated_node_results(run_dir=baseline_node_run_dir)
    constrained_node_results = load_aggregated_node_results(run_dir=constrained_node_run_dir)
    return baseline_node_results, constrained_node_results


@app.cell
def _():
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell
def _(bayesianopt_results, mo):
    available_shared_folds = [int(value) for value in bayesianopt_results["fold_overview_table"]["fold"].tolist()]
    selected_shared_fold = mo.ui.dropdown(
        options=available_shared_folds,
        value=available_shared_folds[2],
        label="Held-out fold",
    )
    selected_shared_fold
    return (selected_shared_fold,)


@app.cell
def _(Path, bayesianopt_results, mo, pd, selected_shared_fold):
    shared_run_dir = Path(bayesianopt_results["resolved_run_dir"])
    shared_fold_prediction_frame = pd.read_parquet(
        shared_run_dir / f"fold_{int(selected_shared_fold.value)}" / "test_predictions.parquet"
    ).copy()
    shared_patient_ids = sorted(int(value) for value in shared_fold_prediction_frame["patient_id"].drop_duplicates().tolist())[
        :3
    ]
    shared_patient_table = pd.DataFrame({"patient_id": shared_patient_ids})
    mo.vstack(
        [
            mo.md(
                """
                ## Held-out fold for sample predictions

                The Bayesian optimisation, hybrid, and NODE examples below all use the same
                selected held-out fold and the same first 3 sorted held-out patient IDs.
                """
            ),
            shared_patient_table,
        ]
    )
    return (shared_patient_ids,)


@app.cell
def _(mo):
    mo.md("""
    ## Bayesian Optimisation

    This subsection summarizes the population-level Bayesian optimisation workflow
    over the 8 PBPK kinetic parameters of `pharmacokinetics.remifentanil`.
    Bayesian optimisation is used here because each objective evaluation requires
    repeated simulation of an expensive mechanistic ODE model over a cohort of
    patients. The optimisation is framed as a weighted nonlinear least-squares
    problem under heteroscedastic measurement noise, and in this notebook we
    describe the measurement model with an assumed **5% proportional variance** to
    account for observation noise. The Gaussian-process surrogate is queried with
    an **Expected Improvement** acquisition function to balance exploration of the
    PBPK parameter space against exploitation of promising regions.

    Scientifically, this BO benchmark is intentionally restrictive: it identifies a
    single population-level kinetic vector and therefore tests how far the
    mechanistic model can be pushed before moving to more flexible patient-aware
    architectures. In practice, the parity plot and fold-specific examples show
    that this model family can fit the global trend, but it had trouble capturing
    the sharp peak injections that motivate the hybrid formulation introduced next.
    """)
    return


@app.cell
def _(bayesianopt_results):
    bo_bounds_table = bayesianopt_results["parameter_summary_table"].loc[:, ["parameter", "low", "high", "prior"]]
    bo_settings_table = (
        bayesianopt_results["run_summary_table"]
        .loc[
            bayesianopt_results["run_summary_table"]["field"].isin(
                ["Objective", "Objective scale", "n_calls", "n_initial_points"]
            )
        ]
        .reset_index(drop=True)
    )
    return bo_bounds_table, bo_settings_table


@app.cell
def _(bo_bounds_table, bo_settings_table, mo):
    mo.vstack(
        [
            mo.md("### Parameter bounds"),
            bo_bounds_table,
            mo.md("### BO settings"),
            bo_settings_table,
        ]
    )
    return


@app.cell
def _(bayesianopt_results, mo, pd, plot_bayesianopt_double_cv_parity):
    if bayesianopt_results["parity_data_available"]:
        _bo_parity_figure = plot_bayesianopt_double_cv_parity(bayesianopt_results)
        _bo_output = _bo_parity_figure
    else:
        _bo_missing_table = pd.DataFrame({"missing_artifact": bayesianopt_results["parity_missing_artifacts"]})
        _bo_output = mo.vstack(
            [
                mo.md("### Double-CV parity plot"),
                mo.md(bayesianopt_results["parity_status_message"]),
                _bo_missing_table,
            ]
        )
    _bo_output
    return


@app.cell
def _(bayesianopt_results, mo):
    mo.vstack(
        [
            mo.md(
                """
                ### Fold-aggregated error summary

                As with the other benchmark sections, `train` corresponds to the pooled
                4/5 of patients used inside each outer fold, while `test` is the held-out
                1/5 used for the cross-validated generalization estimate.
                """
            ),
            bayesianopt_results["metrics_display_table"],
        ]
    )
    return


@app.cell
def _(mo, selected_shared_fold):
    mo.vstack(
        [
            mo.md(
                """
                ### Bayesian optimisation fold-specific prediction example

                The cells below load the shared held-out fold and display the same 3
                held-out patients used in the hybrid and NODE sections.
                """
            ),
            mo.md(f"Using held-out fold `{int(selected_shared_fold.value)}`."),
        ]
    )
    return


@app.cell
def _(Path, bayesianopt_results, pd, selected_shared_fold, shared_patient_ids):
    bo_run_dir = Path(bayesianopt_results["resolved_run_dir"])
    bo_fold_prediction_frame = pd.read_parquet(
        bo_run_dir / f"fold_{int(selected_shared_fold.value)}" / "test_predictions.parquet"
    ).copy()
    sampled_bo_prediction_frame = (
        bo_fold_prediction_frame.loc[bo_fold_prediction_frame["patient_id"].isin(shared_patient_ids)]
        .sort_values(["patient_id", "measurement_index"])
        .reset_index(drop=True)
    )
    sampled_bo_patient_table = pd.DataFrame(
        [
            {
                "patient_id": int(row["patient_id"]),
                "age": float(row["age"]),
                "weight": float(row["weight"]),
                "height": float(row["height"]),
                "bsa": float(row["bsa"]),
                "dose_rate": float(row["dose_rate"]),
                "dose_duration": float(row["dose_duration"]),
            }
            for row in (
                sampled_bo_prediction_frame.loc[
                    :, ["patient_id", "age", "weight", "height", "bsa", "dose_rate", "dose_duration"]
                ]
                .drop_duplicates(subset="patient_id")
                .set_index("patient_id")
                .loc[shared_patient_ids]
                .reset_index()
                .to_dict(orient="records")
            )
        ]
    )
    return sampled_bo_patient_table, sampled_bo_prediction_frame


@app.cell
def _(
    mo,
    plt,
    sampled_bo_patient_table,
    sampled_bo_prediction_frame,
    selected_shared_fold,
):
    bo_patient_ids = [int(value) for value in sampled_bo_patient_table["patient_id"].tolist()]
    bo_figure, bo_axes = plt.subplots(
        len(bo_patient_ids),
        1,
        figsize=(8, 3 * len(bo_patient_ids)),
        squeeze=False,
    )
    bo_axes = bo_axes.ravel()

    for _axis, _patient_id in zip(bo_axes, bo_patient_ids, strict=True):
        _bo_patient_prediction_frame = sampled_bo_prediction_frame.loc[sampled_bo_prediction_frame["patient_id"] == _patient_id]
        _bo_patient_row = sampled_bo_patient_table.loc[sampled_bo_patient_table["patient_id"] == _patient_id].iloc[0]
        _axis.plot(
            _bo_patient_prediction_frame["time"],
            _bo_patient_prediction_frame["observed"],
            marker="o",
            linestyle="",
            label="Observed",
            alpha=0.8,
        )
        _axis.plot(
            _bo_patient_prediction_frame["time"],
            _bo_patient_prediction_frame["predicted"],
            linewidth=2.0,
            label="Predicted",
        )
        _axis.set_title(
            f"BO fold {int(selected_shared_fold.value)} test patient {_patient_id} | "
            f"age={float(_bo_patient_row['age']):.0f}, wt={float(_bo_patient_row['weight']):.0f}, "
            f"bsa={float(_bo_patient_row['bsa']):.2f}"
        )
        _axis.set_xlabel("Time [min]")
        _axis.set_ylabel("Concentration")
        _axis.legend(loc="best")

    bo_figure.tight_layout()
    mo.vstack(
        [
            mo.md("### Three held-out test patients from the saved Bayesian optimisation fold predictions"),
            sampled_bo_patient_table,
            bo_figure,
        ]
    )
    return


@app.cell
def _(hybrid_results):
    hybrid_hyperparameter_table = (
        hybrid_results["run_summary_table"]
        .loc[
            hybrid_results["run_summary_table"]["field"].isin(
                ["Width size", "Depth", "Learning rate", "Epochs", "Report every", "Covariates"]
            )
        ]
        .reset_index(drop=True)
    )
    return (hybrid_hyperparameter_table,)


@app.cell
def _(hybrid_hyperparameter_table, mo):
    mo.vstack(
        [
            mo.md(
                """
                ## Hybrid model

                The population-level BO model had trouble capturing the peak injections,
                which points to a structural limitation of using one shared kinetic
                parameter vector for all patients. Hybrid modelling offers a principled
                solution: retain the mechanistic PBPK simulator as the dynamical core,
                while learning the patient-specific kinetic parameters from data. This
                grey-box architecture preserves the physics-informed state evolution and
                mass-balance guarantees, yet gains the flexibility to capture
                inter-patient variability that a single population-level parameter
                vector cannot express.

                In `experiments/adapters/hybrid_fixed_hparams.py`, each fold first MinMax
                scales `age`, `weight`, `height`, `bsa`, `dose_rate`, and
                `dose_duration` into `[-1, 1]`. Those 6 covariates are passed into an
                **Artificial Neural Network** with `swish` activation, producing 8
                kinetic-parameter preactivations. Positive rate parameters are mapped
                through `softplus`, while the fractional parameters `Eff_kid` and
                `Eff_hep` are mapped through `sigmoid`, after which the resulting
                patient-specific kinetic vectors are evaluated through
                `pharmacokinetics.remifentanil`. The train and test parity plots are
                extremely good, showing that the hybrid model is very strong at learning
                robust patient-specific surrogates and supports a personalised-medicine
                framing in which mechanistic structure and subject-level flexibility are
                combined.
                """
            ),
            mo.md("### Fixed hyperparameters"),
            hybrid_hyperparameter_table,
        ]
    )
    return


@app.cell
def _(hybrid_results, mo, pd, plot_hybrid_double_cv_parity):
    if hybrid_results["parity_data_available"]:
        _hybrid_parity_figure = plot_hybrid_double_cv_parity(hybrid_results)
        _hybrid_output = _hybrid_parity_figure
    else:
        _hybrid_missing_table = pd.DataFrame({"missing_artifact": hybrid_results["parity_missing_artifacts"]})
        _hybrid_output = mo.vstack(
            [
                mo.md("### Double-CV parity plot"),
                mo.md(hybrid_results["parity_status_message"]),
                _hybrid_missing_table,
            ]
        )
    _hybrid_output
    return


@app.cell
def _(hybrid_results, mo):
    mo.vstack(
        [
            mo.md(
                """
                ### Fold-aggregated error summary

                As with the other benchmark sections, `train` corresponds to the pooled
                4/5 of patients used to fit each outer fold, while `test` is the held-out
                1/5 used for the cross-validated generalization estimate. The key point
                in this notebook is not only that the numerical errors are low, but that
                the hybrid parity plots remain very tight in both training and held-out
                testing. This indicates that the Artificial Neural Network is not merely
                memorising trajectories: it is learning a stable patient-to-kinetics map
                that makes the PBPK simulator substantially more robust than the
                population-only BO fit.
                """
            ),
            hybrid_results["metrics_display_table"],
        ]
    )
    return


@app.cell
def _(mo, selected_shared_fold):
    mo.vstack(
        [
            mo.md(
                """
                ### Hybrid fold-specific prediction example

                The cells below load the shared held-out fold and display the same 3
                held-out patients used in the BO and NODE sections.
                """
            ),
            mo.md(f"Using held-out fold `{int(selected_shared_fold.value)}`."),
        ]
    )
    return


@app.cell
def _(Path, hybrid_results, pd, selected_shared_fold, shared_patient_ids):
    hybrid_run_dir = Path(hybrid_results["resolved_run_dir"])
    hybrid_fold_prediction_frame = pd.read_parquet(
        hybrid_run_dir / f"fold_{int(selected_shared_fold.value)}" / "test_predictions.parquet"
    ).copy()
    sampled_hybrid_prediction_frame = (
        hybrid_fold_prediction_frame.loc[hybrid_fold_prediction_frame["patient_id"].isin(shared_patient_ids)]
        .sort_values(["patient_id", "measurement_index"])
        .reset_index(drop=True)
    )
    sampled_hybrid_patient_table = pd.DataFrame(
        [
            {
                "patient_id": int(row["patient_id"]),
                "age": float(row["age"]),
                "weight": float(row["weight"]),
                "height": float(row["height"]),
                "bsa": float(row["bsa"]),
                "dose_rate": float(row["dose_rate"]),
                "dose_duration": float(row["dose_duration"]),
            }
            for row in (
                sampled_hybrid_prediction_frame.loc[
                    :, ["patient_id", "age", "weight", "height", "bsa", "dose_rate", "dose_duration"]
                ]
                .drop_duplicates(subset="patient_id")
                .set_index("patient_id")
                .loc[shared_patient_ids]
                .reset_index()
                .to_dict(orient="records")
            )
        ]
    )
    return sampled_hybrid_patient_table, sampled_hybrid_prediction_frame


@app.cell
def _(
    mo,
    plt,
    sampled_hybrid_patient_table,
    sampled_hybrid_prediction_frame,
    selected_shared_fold,
):
    hybrid_patient_ids = [int(value) for value in sampled_hybrid_patient_table["patient_id"].tolist()]
    hybrid_figure, hybrid_axes = plt.subplots(
        len(hybrid_patient_ids),
        1,
        figsize=(8, 3 * len(hybrid_patient_ids)),
        squeeze=False,
    )
    hybrid_axes = hybrid_axes.ravel()

    for _axis, _patient_id in zip(hybrid_axes, hybrid_patient_ids, strict=True):
        _hybrid_patient_prediction_frame = sampled_hybrid_prediction_frame.loc[
            sampled_hybrid_prediction_frame["patient_id"] == _patient_id
        ]
        _hybrid_patient_row = sampled_hybrid_patient_table.loc[sampled_hybrid_patient_table["patient_id"] == _patient_id].iloc[
            0
        ]
        _axis.plot(
            _hybrid_patient_prediction_frame["time"],
            _hybrid_patient_prediction_frame["observed"],
            marker="o",
            linestyle="",
            label="Observed",
            alpha=0.8,
        )
        _axis.plot(
            _hybrid_patient_prediction_frame["time"],
            _hybrid_patient_prediction_frame["predicted"],
            linewidth=2.0,
            label="Predicted",
        )
        _axis.set_title(
            f"Hybrid fold {int(selected_shared_fold.value)} test patient {_patient_id} | "
            f"age={float(_hybrid_patient_row['age']):.0f}, wt={float(_hybrid_patient_row['weight']):.0f}, "
            f"bsa={float(_hybrid_patient_row['bsa']):.2f}"
        )
        _axis.set_xlabel("Time [min]")
        _axis.set_ylabel("Concentration")
        _axis.legend(loc="best")

    hybrid_figure.tight_layout()
    mo.vstack(
        [
            mo.md("### Three held-out test patients from the saved hybrid fold predictions"),
            sampled_hybrid_patient_table,
            hybrid_figure,
        ]
    )
    return


@app.cell
def _(baseline_node_results, pd):
    baseline_node_hyperparameter_table = (
        baseline_node_results["run_summary_table"]
        .loc[
            baseline_node_results["run_summary_table"]["field"].isin(
                [
                    "Width size",
                    "Depth",
                    "Augment dim",
                    "Learning-rate strategy",
                    "Steps strategy",
                    "Length strategy",
                    "Batch size",
                    "Covariates",
                ]
            )
        ]
        .reset_index(drop=True)
    )
    baseline_node_run_summary = pd.DataFrame(
        [
            {
                "field": "Pinned run",
                "value": baseline_node_results["manifest"]["run_id"],
            },
            {
                "field": "Resolved run dir",
                "value": str(baseline_node_results["resolved_run_dir"]),
            },
        ]
    )
    return baseline_node_hyperparameter_table, baseline_node_run_summary


@app.cell
def _(mo):
    mo.md("""
    ## NODE: Previous unconstrained run

    A Neural ODE replaces an explicit compartmental right-hand side with a learned
    continuous-time vector field that is integrated by an ODE solver. In this
    repository, the maintained implementation lives in
    `pharmacokinetics.remifentanil_node`, where the model learns concentration
    dynamics directly from patient trajectories and static covariates rather than
    from a predefined PBPK transport formulation. This is attractive from an ML
    perspective because it is fully data driven, naturally continuous in time, and
    well suited to irregular measurement grids.

    This subsection pins the previous successful NODE benchmark **without** the
    infusion-aware sign constraints. Architecturally, the baseline NODE uses static
    covariates (`age`, `weight`, `height`, `sex`, `dose_rate`, `dose_duration`) to
    condition the vector field, while learned augmentation dimensions increase the
    expressivity of the latent dynamics. The advantage is flexibility: no explicit
    PBPK model formulation is required. The trade-off is that the model forfeits
    mechanistic guarantees such as mass conservation and non-negativity, and the
    parity plot shows that some learned vector fields are numerically unstable,
    yielding non-physical negative concentrations in certain patient trajectories.
    """)
    return


@app.cell
def _(baseline_node_hyperparameter_table, baseline_node_run_summary, mo):
    mo.vstack(
        [
            mo.md(
                """
                The saved unconstrained run is therefore best interpreted as a pure
                data-driven baseline. It can reproduce broad concentration trends,
                but the parity plot and fold-specific examples show that instability
                remains an issue, especially around trajectories that would require a
                stronger structural prior to prevent non-physical negative values.
                """
            ),
            mo.md("### Pinned unconstrained run"),
            baseline_node_run_summary,
            mo.md("### Hyperparameters"),
            baseline_node_hyperparameter_table,
        ]
    )
    return


@app.cell
def _(baseline_node_results, mo, pd, plot_node_double_cv_parity):
    if baseline_node_results["parity_data_available"]:
        _baseline_node_parity_figure = plot_node_double_cv_parity(baseline_node_results)
        _baseline_node_output = _baseline_node_parity_figure
    else:
        _baseline_node_missing_table = pd.DataFrame({"missing_artifact": baseline_node_results["parity_missing_artifacts"]})
        _baseline_node_output = mo.vstack(
            [
                mo.md("### Double-CV parity plot"),
                mo.md(baseline_node_results["parity_status_message"]),
                _baseline_node_missing_table,
            ]
        )
    _baseline_node_output
    return


@app.cell
def _(mo, selected_shared_fold):
    mo.vstack(
        [
            mo.md(
                """
                ### Unconstrained NODE fold-specific prediction example

                The cells below load the shared held-out fold and display the same 3
                held-out patients used in the BO, hybrid, and constrained NODE sections.
                """
            ),
            mo.md(f"Using held-out fold `{int(selected_shared_fold.value)}`."),
        ]
    )
    return


@app.cell
def _(baseline_node_results, pd, selected_shared_fold, shared_patient_ids):
    _baseline_node_run_dir = baseline_node_results["resolved_run_dir"]
    baseline_node_fold_prediction_frame = pd.read_parquet(
        _baseline_node_run_dir / f"fold_{int(selected_shared_fold.value)}" / "test_predictions.parquet"
    ).copy()
    sampled_baseline_node_prediction_frame = (
        baseline_node_fold_prediction_frame.loc[baseline_node_fold_prediction_frame["patient_id"].isin(shared_patient_ids)]
        .sort_values(["patient_id", "measurement_index"])
        .reset_index(drop=True)
    )
    sampled_baseline_node_patient_table = pd.DataFrame(
        [
            {
                "patient_id": int(row["patient_id"]),
                "age": float(row["age"]),
                "weight": float(row["weight"]),
                "height": float(row["height"]),
                "sex": float(row["sex"]),
                "dose_rate": float(row["dose_rate"]),
                "dose_duration": float(row["dose_duration"]),
            }
            for row in (
                sampled_baseline_node_prediction_frame.loc[
                    :, ["patient_id", "age", "weight", "height", "sex", "dose_rate", "dose_duration"]
                ]
                .drop_duplicates(subset="patient_id")
                .set_index("patient_id")
                .loc[shared_patient_ids]
                .reset_index()
                .to_dict(orient="records")
            )
        ]
    )
    return (
        sampled_baseline_node_patient_table,
        sampled_baseline_node_prediction_frame,
    )


@app.cell
def _(
    mo,
    plt,
    sampled_baseline_node_patient_table,
    sampled_baseline_node_prediction_frame,
    selected_shared_fold,
):
    baseline_node_patient_ids = [int(value) for value in sampled_baseline_node_patient_table["patient_id"].tolist()]
    baseline_node_figure, baseline_node_axes = plt.subplots(
        len(baseline_node_patient_ids),
        1,
        figsize=(8, 3 * len(baseline_node_patient_ids)),
        squeeze=False,
    )
    baseline_node_axes = baseline_node_axes.ravel()

    for _axis, _patient_id in zip(baseline_node_axes, baseline_node_patient_ids, strict=True):
        _baseline_node_patient_prediction_frame = sampled_baseline_node_prediction_frame.loc[
            sampled_baseline_node_prediction_frame["patient_id"] == _patient_id
        ]
        _baseline_node_patient_row = sampled_baseline_node_patient_table.loc[
            sampled_baseline_node_patient_table["patient_id"] == _patient_id
        ].iloc[0]
        _axis.plot(
            _baseline_node_patient_prediction_frame["time"],
            _baseline_node_patient_prediction_frame["observed"],
            marker="o",
            linestyle="",
            label="Observed",
            alpha=0.8,
        )
        _axis.plot(
            _baseline_node_patient_prediction_frame["time"],
            _baseline_node_patient_prediction_frame["predicted"],
            linewidth=2.0,
            label="Predicted",
        )
        _axis.set_title(
            f"Unconstrained NODE fold {int(selected_shared_fold.value)} test patient {_patient_id} | "
            f"age={float(_baseline_node_patient_row['age']):.0f}, "
            f"wt={float(_baseline_node_patient_row['weight']):.0f}, "
            f"sex={float(_baseline_node_patient_row['sex']):.0f}"
        )
        _axis.set_xlabel("Time [min]")
        _axis.set_ylabel("Concentration")
        _axis.legend(loc="best")

    baseline_node_figure.tight_layout()
    mo.vstack(
        [
            mo.md("### Three held-out test patients from the saved unconstrained NODE fold predictions"),
            sampled_baseline_node_patient_table,
            baseline_node_figure,
        ]
    )
    return


@app.cell
def _(constrained_node_results, mo, pd):
    constrained_model_config = constrained_node_results["config"]["resolved_config"]["model"]
    constrained_node_run_summary = pd.DataFrame(
        [
            {
                "field": "Pinned run",
                "value": constrained_node_results["manifest"]["run_id"],
            },
            {
                "field": "Resolved run dir",
                "value": str(constrained_node_results["resolved_run_dir"]),
            },
            {
                "field": "constraint_mode",
                "value": constrained_model_config["constraint_mode"],
            },
            {
                "field": "constraint_buffer_fraction",
                "value": constrained_model_config["constraint_buffer_fraction"],
            },
            {
                "field": "constraint_pre_bias_scale",
                "value": constrained_model_config["constraint_pre_bias_scale"],
            },
        ]
    )
    constrained_node_hyperparameter_table = (
        constrained_node_results["run_summary_table"]
        .loc[
            constrained_node_results["run_summary_table"]["field"].isin(
                [
                    "Width size",
                    "Depth",
                    "Augment dim",
                    "Learning-rate strategy",
                    "Steps strategy",
                    "Length strategy",
                    "Batch size",
                    "Covariates",
                ]
            )
        ]
        .reset_index(drop=True)
    )
    mo.vstack(
        [
            mo.md(
                """
                ## NODE: Infusion-aware constrained run

                This subsection pins the latest successful NODE run with the
                infusion-aware sign constraint enabled. The architectural
                modification is implemented directly in
                `pharmacokinetics/remifentanil_node.py`: the concentration
                derivative is treated as a privileged state, the vector field is
                given explicit timing features relative to `dose_duration`, a soft
                positive bias is applied during infusion, the dynamics are blended
                smoothly near the infusion end, and a negative post-infusion drift
                is enforced after the chosen buffer window.

                Scientifically, this is a minimal structural prior rather than a
                full mechanistic correction. It encodes domain knowledge about the
                expected sign of concentration derivatives during and after infusion,
                encouraging physically plausible dynamics without fully prescribing the
                ODE structure. The parity plot shows improvement, but some predictions
                still exhibit instability. This highlights a general tension in grey-box
                modelling: weak structural priors are easy to impose but may not fully
                constrain the learned dynamics, while stronger priors recover
                interpretability at the cost of model flexibility.
                """
            ),
            mo.md("### Pinned constrained run"),
            constrained_node_run_summary,
            mo.md("### Hyperparameters"),
            constrained_node_hyperparameter_table,
        ]
    )
    return


@app.cell
def _(constrained_node_results, mo, pd, plot_node_double_cv_parity):
    if constrained_node_results["parity_data_available"]:
        _constrained_node_parity_figure = plot_node_double_cv_parity(constrained_node_results)
        _constrained_node_output = _constrained_node_parity_figure
    else:
        _constrained_node_missing_table = pd.DataFrame(
            {"missing_artifact": constrained_node_results["parity_missing_artifacts"]}
        )
        _constrained_node_output = mo.vstack(
            [
                mo.md("### Double-CV parity plot"),
                mo.md(constrained_node_results["parity_status_message"]),
                _constrained_node_missing_table,
            ]
        )
    _constrained_node_output
    return


@app.cell
def _(mo, selected_shared_fold):
    mo.vstack(
        [
            mo.md(
                """
                ### Constrained NODE fold-specific prediction example

                The cells below load the same shared held-out fold and the same 3
                held-out patients used in the baseline NODE section so the shape of
                the constrained trajectories can be compared directly.
                """
            ),
            mo.md(f"Using held-out fold `{int(selected_shared_fold.value)}`."),
        ]
    )
    return


@app.cell
def _(constrained_node_results, pd, selected_shared_fold, shared_patient_ids):
    _constrained_node_run_dir = constrained_node_results["resolved_run_dir"]
    constrained_node_fold_prediction_frame = pd.read_parquet(
        _constrained_node_run_dir / f"fold_{int(selected_shared_fold.value)}" / "test_predictions.parquet"
    ).copy()
    sampled_constrained_node_prediction_frame = (
        constrained_node_fold_prediction_frame.loc[
            constrained_node_fold_prediction_frame["patient_id"].isin(shared_patient_ids)
        ]
        .sort_values(["patient_id", "measurement_index"])
        .reset_index(drop=True)
    )
    sampled_constrained_node_patient_table = pd.DataFrame(
        [
            {
                "patient_id": int(row["patient_id"]),
                "age": float(row["age"]),
                "weight": float(row["weight"]),
                "height": float(row["height"]),
                "sex": float(row["sex"]),
                "dose_rate": float(row["dose_rate"]),
                "dose_duration": float(row["dose_duration"]),
            }
            for row in (
                sampled_constrained_node_prediction_frame.loc[
                    :, ["patient_id", "age", "weight", "height", "sex", "dose_rate", "dose_duration"]
                ]
                .drop_duplicates(subset="patient_id")
                .set_index("patient_id")
                .loc[shared_patient_ids]
                .reset_index()
                .to_dict(orient="records")
            )
        ]
    )
    return (
        sampled_constrained_node_patient_table,
        sampled_constrained_node_prediction_frame,
    )


@app.cell
def _(
    mo,
    plt,
    sampled_constrained_node_patient_table,
    sampled_constrained_node_prediction_frame,
    selected_shared_fold,
):
    constrained_node_patient_ids = [int(value) for value in sampled_constrained_node_patient_table["patient_id"].tolist()]
    constrained_node_figure, constrained_node_axes = plt.subplots(
        len(constrained_node_patient_ids),
        1,
        figsize=(8, 3 * len(constrained_node_patient_ids)),
        squeeze=False,
    )
    constrained_node_axes = constrained_node_axes.ravel()

    for _axis, _patient_id in zip(constrained_node_axes, constrained_node_patient_ids, strict=True):
        _constrained_node_patient_prediction_frame = sampled_constrained_node_prediction_frame.loc[
            sampled_constrained_node_prediction_frame["patient_id"] == _patient_id
        ]
        _constrained_node_patient_row = sampled_constrained_node_patient_table.loc[
            sampled_constrained_node_patient_table["patient_id"] == _patient_id
        ].iloc[0]
        _axis.plot(
            _constrained_node_patient_prediction_frame["time"],
            _constrained_node_patient_prediction_frame["observed"],
            marker="o",
            linestyle="",
            label="Observed",
            alpha=0.8,
        )
        _axis.plot(
            _constrained_node_patient_prediction_frame["time"],
            _constrained_node_patient_prediction_frame["predicted"],
            linewidth=2.0,
            label="Predicted",
        )
        _axis.set_title(
            f"Constrained NODE fold {int(selected_shared_fold.value)} test patient {_patient_id} | "
            f"age={float(_constrained_node_patient_row['age']):.0f}, "
            f"wt={float(_constrained_node_patient_row['weight']):.0f}, "
            f"sex={float(_constrained_node_patient_row['sex']):.0f}"
        )
        _axis.set_xlabel("Time [min]")
        _axis.set_ylabel("Concentration")
        _axis.legend(loc="best")

    constrained_node_figure.tight_layout()
    mo.vstack(
        [
            mo.md("### Three held-out test patients from the saved constrained NODE fold predictions"),
            sampled_constrained_node_patient_table,
            constrained_node_figure,
        ]
    )
    return


@app.cell
def _(gsa_results, mo):
    mo.vstack(
        [
            mo.md(
                """
                ## GSA as interpretation

                In this notebook, global sensitivity analysis is used as an
                **interpretation technique** rather than as another predictive
                benchmark. The hybrid model was selected for this stage because it
                gave the best overall predictive behaviour and therefore provided the
                most credible basis for downstream interpretation. The objective of
                this section is to understand which patient-level covariates and
                dosing variables drive the model response, and how strongly those
                effects depend on cross-parameter interactions.

                The current implementation uses the
                [`gsax`](https://github.com/danielepessina/gsax) package for
                Sobol-based sensitivity analysis on dense time-series outputs. A
                healthy synthetic cohort is sampled from Gaussian covariate
                distributions, with fixed sex and varying age, height, BMI-derived
                weight/BSA, dose rate, and dose duration. The saved hybrid
                checkpoints are then used to generate patient-specific kinetics,
                after which `pharmacokinetics.remifentanil` simulates dense
                concentration trajectories and GSAX computes first-order (`S1`) and
                total-order (`ST`) Sobol indices over time. First-order indices
                quantify direct main effects, whereas total-order indices include
                both direct effects and interactions with the rest of the covariate
                space.
                """
            ),
            mo.md("### Loaded GSAX bundle"),
            gsa_results["run_summary_table"],
            mo.md("### Fold overview"),
            gsa_results["fold_overview_table"],
        ]
    )
    return


@app.cell
def _(gsa_results, np):
    fold_ids = sorted(gsa_results["sobol_datasets"])
    reference_dataset = gsa_results["sobol_datasets"][fold_ids[0]]
    time_values = np.asarray(reference_dataset["time"].values, dtype=float)
    parameter_names = [str(value) for value in reference_dataset["param"].values.tolist()]

    s1_stack = np.stack(
        [
            np.asarray(
                gsa_results["sobol_datasets"][fold_id]["S1"].sel(output="concentration").values,
                dtype=float,
            )
            for fold_id in fold_ids
        ],
        axis=0,
    )
    st_stack = np.stack(
        [
            np.asarray(
                gsa_results["sobol_datasets"][fold_id]["ST"].sel(output="concentration").values,
                dtype=float,
            )
            for fold_id in fold_ids
        ],
        axis=0,
    )

    s1_mean = np.nanmean(s1_stack, axis=0)
    st_mean = np.nanmean(st_stack, axis=0)
    return fold_ids, parameter_names, s1_mean, st_mean, time_values


@app.cell
def _(fold_ids, mo, parameter_names, plt, s1_mean, st_mean, time_values):
    gsa_figure, gsa_axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=False)
    line_handles = []

    for parameter_index, parameter_name in enumerate(parameter_names):
        (line_handle,) = gsa_axes[0].plot(
            time_values,
            s1_mean[:, parameter_index],
            linewidth=2.0,
            label=parameter_name,
        )
        gsa_axes[1].plot(
            time_values,
            st_mean[:, parameter_index],
            linewidth=2.0,
            label=parameter_name,
        )
        line_handles.append(line_handle)

    gsa_axes[0].set_title("First-order Sobol (S1)")
    gsa_axes[1].set_title("Total-order Sobol (ST)")
    gsa_axes[0].set_xlabel("Time [min]")
    gsa_axes[1].set_xlabel("Time [min]")
    gsa_axes[0].set_ylabel("S1")
    gsa_axes[1].set_ylabel("ST")
    gsa_axes[1].set_ylim(0.0, 1.0)

    gsa_figure.legend(
        handles=line_handles,
        loc="lower center",
        ncol=max(1, len(parameter_names)),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    gsa_figure.tight_layout(rect=(0.0, 0.10, 1.0, 0.95))

    mo.vstack(
        [
            mo.md(f"### Fold-averaged Sobol trajectories across `{len(fold_ids)}` saved hybrid folds"),
            gsa_figure,
            mo.md(
                """
                The main scientific conclusion is that **dose duration dominates**
                the hybrid model response across time. Equally important, the gap
                between first-order and total-order indices is large for several
                covariates, which indicates substantial cross-parameter interactions
                rather than purely additive effects. In other words, the response is
                not controlled by isolated single variables alone: age, body-size
                descriptors, dose rate, and especially dose duration interact to
                shape the personalised concentration trajectories. This is precisely
                why GSA is useful here as an interpretation tool for a personalised
                modelling workflow. More broadly, the Sobol analysis demonstrates how
                sensitivity methods can extract mechanistic insight from a trained
                hybrid model: the learned patient-to-kinetics map is not a black box,
                but a parametric function whose input-output behaviour can be
                interrogated systematically.
                """
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
