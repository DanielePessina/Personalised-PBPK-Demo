"""Utilities for the Remifentanil marimo exploratory notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import seaborn as sns
from seaborn.axisgrid import PairGrid


def load_patient_covariate_frame(dataset_path: str | Path) -> pd.DataFrame:
    """Load one patient-level row per subject from the remifentanil workbook.

    Parameters
    ----------
    dataset_path : str | Path
        Path to the local ``nlme-remifentanil.xlsx`` workbook.

    Returns
    -------
    pd.DataFrame
        Patient-level table with columns ``patient_id``, ``n_samples``,
        ``final_measurement_time``, ``age``, ``weight``, ``height``, ``sex``,
        and ``bsa``.
    """
    workbook = Path(dataset_path).resolve()
    frame = pd.read_excel(workbook)
    frame = frame.loc[:, ~((frame.columns.str.contains("Unnamed")) & (frame.isnull().any()))].copy()

    patient_covariates = (
        frame.groupby("ID", as_index=False)
        .agg(
            age=("Age", "first"),
            weight=("Wt", "first"),
            height=("Ht", "first"),
            sex=("Sex", "first"),
            bsa=("BSA", "first"),
        )
        .rename(columns={"ID": "patient_id"})
    )

    observed_concentrations = (
        frame.dropna(subset=["conc"])
        .groupby("ID", as_index=False)
        .agg(
            n_samples=("conc", "size"),
            final_measurement_time=("Time", "max"),
        )
        .rename(columns={"ID": "patient_id"})
    )

    patient_frame = patient_covariates.merge(observed_concentrations, on="patient_id", how="inner", validate="one_to_one")
    patient_frame = patient_frame[
        ["patient_id", "n_samples", "final_measurement_time", "age", "weight", "height", "sex", "bsa"]
    ].sort_values("patient_id", ignore_index=True)

    patient_frame["patient_id"] = patient_frame["patient_id"].astype(int)
    patient_frame["n_samples"] = patient_frame["n_samples"].astype(int)
    patient_frame["sex"] = patient_frame["sex"].astype(str)

    numeric_columns = ["final_measurement_time", "age", "weight", "height", "bsa"]
    patient_frame[numeric_columns] = patient_frame[numeric_columns].astype(float)
    return patient_frame


def compute_patient_summary(patient_frame: pd.DataFrame) -> dict[str, Any]:
    """Compute a compact descriptive summary for the patient-level cohort.

    Parameters
    ----------
    patient_frame : pd.DataFrame
        Patient-level DataFrame returned by ``load_patient_covariate_frame``.

    Returns
    -------
    dict[str, Any]
        Summary dictionary containing cohort size, average sampling density,
        average final measurement time, average covariates, and sex counts.
    """
    return {
        "patient_count": int(len(patient_frame)),
        "average_sampled_points": float(patient_frame["n_samples"].mean()),
        "average_final_measurement_time": float(patient_frame["final_measurement_time"].mean()),
        "average_age": float(patient_frame["age"].mean()),
        "average_weight": float(patient_frame["weight"].mean()),
        "average_height": float(patient_frame["height"].mean()),
        "average_bsa": float(patient_frame["bsa"].mean()),
        "sex_counts": patient_frame["sex"].value_counts().sort_index().to_dict(),
    }


def plot_covariate_corner(patient_frame: pd.DataFrame) -> PairGrid:
    """Build a seaborn corner plot for the requested patient covariates.

    Parameters
    ----------
    patient_frame : pd.DataFrame
        Patient-level DataFrame returned by ``load_patient_covariate_frame``.

    Returns
    -------
    PairGrid
        Seaborn pair grid with ``age``, ``weight``, ``height``, and ``bsa`` on
        the axes and ``sex`` encoded through the ``Dark2`` palette.
    """
    plot_frame = patient_frame.copy()
    plot_frame["sex"] = pd.Categorical(plot_frame["sex"], categories=["Female", "Male"], ordered=True)

    grid = sns.pairplot(
        data=plot_frame,
        vars=["age", "weight", "height", "bsa"],
        hue="sex",
        hue_order=["Female", "Male"],
        palette="Dark2",
        corner=True,
        diag_kind="hist",
        height=2.4,
        plot_kws={"alpha": 0.85, "s": 55, "edgecolor": "none"},
        diag_kws={"alpha": 0.75, "bins": 12},
    )
    grid.figure.suptitle("Remifentanil patient covariates", y=1.02)
    if grid._legend is not None:
        grid._legend.set_title("Sex")
    return grid
