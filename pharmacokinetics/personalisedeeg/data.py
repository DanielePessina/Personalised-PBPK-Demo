"""
Data loading and preprocessing for the remifentanil Neural ODE project.

This module defines simple containers for per‑patient observations and
provides functions to parse the raw Excel dataset, extract the EEG
measurements, scale and pad variable‑length sequences into fixed‑size
arrays, and split the cohort into training, validation and test sets.
The logic here is inspired by ``BatchedExperiments`` in the context
code but adapted to the remifentanil dataset structure.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from typing import Optional

from .config import DataSplitConfig, PBPKConfig
from .. import remifentanil
from .. import nlme
# from .pbpk_remifentanil import Physiology


@dataclass
class PatientRecord:
    """Container for a single patient's EEG measurements and covariates.

    Attributes:
        id: Integer patient identifier.
        times: 1D numpy array of measurement times (raw units).
        values: 1D numpy array of EEG measurements (raw units).
        covariates: 1D numpy array of selected covariate values for this patient.
        dose_rate: Optional scalar dose rate for this patient (per-time units).
        dose_duration: Optional scalar dose duration for this patient (time units).
    """

    id: int
    times: np.ndarray
    values: np.ndarray
    covariates: np.ndarray
    dose_rate: float = 0.0
    dose_duration: float = 0.0

    def __post_init__(self):
        assert self.times.ndim == 1, "times must be 1D"
        assert self.values.ndim == 1, "values must be 1D"
        assert self.times.shape == self.values.shape, "times and values must align"
        assert self.covariates.ndim == 1, "covariates must be 1D"
        # dose fields are scalars (floats)


@dataclass
class PreparedData:
    """Padded and scaled dataset ready for batching.

    The arrays here mirror those in ``BatchedExperiments``.  ``t_pad`` and
    ``y_pad`` have shape [N, T_max], ``mask`` is a boolean mask
    indicating which entries correspond to real measurements, and
    ``covars`` contains the scaled covariates.  The ``scalers`` dict
    stores the fitted MinMaxScalers for values and covariates as well
    as the global maximum time used for time scaling.
    """

    t_pad: jnp.ndarray  # [N, T_max]
    y_pad: jnp.ndarray  # [N, T_max]
    mask: jnp.ndarray  # [N, T_max] boolean
    covars: jnp.ndarray  # [N, C]
    dose_rate: jnp.ndarray  # [N] scalar per patient
    dose_duration: jnp.ndarray  # [N] scalar per patient
    lengths: jnp.ndarray  # [N] number of valid timepoints per patient
    ids: jnp.ndarray  # [N] patient IDs
    scalers: Dict[str, Any]
    # Optional fields for dense concentration profiles
    dense_ts: jnp.ndarray | None = None  # [T_dense] time points for concentration
    dense_cs: jnp.ndarray | None = None  # [N, T_dense] concentration profiles
    dense_ts: jnp.ndarray | None = None
    dense_cs: jnp.ndarray | None = None

    def __post_init__(self):
        assert self.t_pad.ndim == 2, "t_pad must be 2D"
        assert self.y_pad.ndim == 2, "y_pad must be 2D"
        assert self.mask.shape == self.t_pad.shape, "mask must match t_pad shape"
        assert self.t_pad.shape == self.y_pad.shape, "t_pad and y_pad must align"
        assert self.covars.ndim == 2, "covars must be 2D"
        assert self.covars.shape[0] == self.t_pad.shape[0], "covars must match batch size"
        assert self.dose_rate.ndim == 1 and self.dose_rate.shape[0] == self.t_pad.shape[0], (
            "dose_rate must match batch size"
        )
        assert self.dose_duration.ndim == 1 and self.dose_duration.shape[0] == self.t_pad.shape[0], (
            "dose_duration must match batch size"
        )
        assert self.lengths.ndim == 1, "lengths must be 1D"
        assert self.lengths.shape[0] == self.t_pad.shape[0], "lengths must match batch size"
        assert self.ids.ndim == 1 and self.ids.shape[0] == self.t_pad.shape[0], "ids must match batch size"


@dataclass
class PreparedPhysiology:
    """Batched physiology data ready for vmap operations.

    This class contains padded/batched physiology parameters for all patients
    to enable efficient vmap operations in JAX/Equinox models.
    """

    # Patient-level data [N] arrays
    covariates: jnp.ndarray  # [N, C] - AGE, SEX, WT, HT, etc.
    dose_rate: jnp.ndarray  # [N] - infusion rates
    dose_duration: jnp.ndarray  # [N] - infusion durations

    # Derived physiology parameters [N] arrays
    flows: jnp.ndarray  # [N, n_flows] - q_PV, q_HA, q_HV, q_K
    volumes: jnp.ndarray  # [N, n_volumes] - v_L, v_GL, v_SIL, v_LIL, v_GICS, v_T, v_HP, v_P
    fixed_params: jnp.ndarray  # [N, n_fixed] - alpha, Rp, t_G, t_SI, t_LI, k_A_*

    # Metadata
    patient_ids: jnp.ndarray  # [N] - patient identifiers
    scalers: Dict[str, Any]  # scaling information

    def __post_init__(self):
        """Validate array dimensions and consistency."""
        N = self.covariates.shape[0]

        # Check shapes
        assert self.covariates.ndim == 2, "covariates must be 2D [N, C]"
        assert self.dose_rate.shape == (N,), "dose_rate must be 1D [N]"
        assert self.dose_duration.shape == (N,), "dose_duration must be 1D [N]"

        assert self.flows.ndim == 2 and self.flows.shape[0] == N, "flows must be 2D [N, n_flows]"
        assert self.volumes.ndim == 2 and self.volumes.shape[0] == N, "volumes must be 2D [N, n_volumes]"
        assert self.fixed_params.ndim == 2 and self.fixed_params.shape[0] == N, "fixed_params must be 2D [N, n_fixed]"

        assert self.patient_ids.shape == (N,), "patient_ids must be 1D [N]"
        assert isinstance(self.scalers, dict), "scalers must be a dict"


@dataclass
class UnscaledPreparedData:
    """Padded dataset ready for batching but keeping original units/scale.

    This version avoids any scaling transformations and works directly with
    the original data units, making it JAX-friendly for optimization.
    """

    t_pad: jnp.ndarray  # [N, T_max] - times in original units
    y_pad: jnp.ndarray  # [N, T_max] - values in original units
    mask: jnp.ndarray  # [N, T_max] boolean
    covars: jnp.ndarray  # [N, C] - covariates in original units
    dose_rate: jnp.ndarray  # [N] - dose rates in original units
    dose_duration: jnp.ndarray  # [N] - dose durations in original units
    lengths: jnp.ndarray  # [N] number of valid timepoints per patient
    ids: jnp.ndarray  # [N] patient IDs
    global_max_time: float  # maximum time across all patients

    def __post_init__(self):
        assert self.t_pad.ndim == 2, "t_pad must be 2D"
        assert self.y_pad.ndim == 2, "y_pad must be 2D"
        assert self.mask.shape == self.t_pad.shape, "mask must match t_pad shape"
        assert self.t_pad.shape == self.y_pad.shape, "t_pad and y_pad must align"
        assert self.covars.ndim == 2, "covars must be 2D"
        assert self.covars.shape[0] == self.t_pad.shape[0], "covars must match batch size"
        assert self.dose_rate.ndim == 1 and self.dose_rate.shape[0] == self.t_pad.shape[0], (
            "dose_rate must match batch size"
        )
        assert self.dose_duration.ndim == 1 and self.dose_duration.shape[0] == self.t_pad.shape[0], (
            "dose_duration must match batch size"
        )
        assert self.lengths.ndim == 1, "lengths must be 1D"
        assert self.lengths.shape[0] == self.t_pad.shape[0], "lengths must match batch size"
        assert self.ids.ndim == 1 and self.ids.shape[0] == self.t_pad.shape[0], "ids must match batch size"


def load_raw_dataset(
    filepath: str,
    covariate_cols: List[str],
    *,
    time_col: str = "TIME",
    value_col: str = "DV",
    id_col: str = "ID",
    ytype_col: str = "YTYPE",
    ytype_value: Any = 2,
    mdv_col: str = "MDV",
    mdv_observation: Any = 0,
    normalise_eeg: bool = False,
    downsampling_stride: int = 1,
) -> List[PatientRecord]:
    """Load and parse the raw remifentanil dataset.

    The function reads the Excel or CSV file, filters rows where
    ``ytype_col`` equals ``ytype_value`` (to select EEG measurements) and
    ``mdv_col`` equals ``mdv_observation`` (indicating observation rows).
    It drops rows with non‑numeric values in ``value_col`` and groups the
    remaining rows by patient ID.  For each patient the times,
    values and covariates are extracted and stored in a
    :class:`PatientRecord`.

    Args:
        filepath: Path to the Excel (.xls/.xlsx) or CSV file.
        covariate_cols: List of column names to use as covariates.
        time_col: Name of the time column in the file.
        value_col: Name of the measurement column.
        id_col: Name of the patient identifier column.
        ytype_col: Name of the column specifying measurement type.
        ytype_value: Value of ``ytype_col`` that corresponds to the EEG signal.
        mdv_col: Name of the column indicating dosing/observation rows.
        mdv_observation: Value of ``mdv_col`` that indicates an observation.
        normalise_eeg: If True, subtract the first EEG value from all values for each patient.
        downsampling_stride: Stride for downsampling data points. Default is 1 (no downsampling).
                           If > 1, keeps every Nth data point (e.g., stride=2 keeps every 2nd point).

    Returns:
        A list of :class:`PatientRecord` instances.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext in {".xls", ".xlsx"}:
        # csv_path = _convert_excel_to_csv(filepath)
        df = pd.read_excel(filepath, engine="openpyxl" if ext == ".xlsx" else None)
    else:
        df = pd.read_csv(filepath)
    # Ensure columns exist
    for col in [time_col, value_col, id_col, ytype_col, mdv_col] + covariate_cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in {filepath}")
    # Filter for observation rows of the desired type
    df_filtered = df[(df[ytype_col].astype(str) == str(ytype_value)) & (df[mdv_col] == mdv_observation)]
    # Remove rows with missing or non‑numeric values
    df_filtered = df_filtered[df_filtered[value_col].apply(lambda x: str(x).strip() not in {".", "", "NA", "NaN"})]
    # Convert times and values to float; coerce errors to NaN and drop
    df_filtered[time_col] = pd.to_numeric(df_filtered[time_col], errors="coerce")
    df_filtered[value_col] = pd.to_numeric(df_filtered[value_col], errors="coerce")
    df_filtered = df_filtered.dropna(subset=[time_col, value_col])
    # Group by patient ID
    records: List[PatientRecord] = []
    for pid, group in df_filtered.groupby(id_col):
        times = group[time_col].to_numpy(dtype=float)
        values = group[value_col].to_numpy(dtype=float)
        # Sort by time just in case
        idx_sort = np.argsort(times)
        times = times[idx_sort]
        values = values[idx_sort]

        # Apply downsampling if stride > 1
        if downsampling_stride > 1 and len(times) > 1:
            # Select every Nth element where N is the stride
            downsample_indices = np.arange(0, len(times), downsampling_stride)
            times = times[downsample_indices]
            values = values[downsample_indices]

        if normalise_eeg:
            values = values - values[0]
        # Extract covariates from the first row of the original DataFrame (not filtered by ytype/mdv) so that
        # missing covariates on observation rows do not lead to NaNs.
        covariate_values = []
        for col in covariate_cols:
            # Look up the first non‑NaN value for this patient and covariate
            full_group = df[df[id_col] == pid]
            val_series = full_group[col]
            # If column is numeric, convert; otherwise keep as float
            val = val_series.dropna().iloc[0] if not val_series.dropna().empty else np.nan
            # Convert booleans or strings to float where possible
            try:
                val_float = float(val)
            except (ValueError, TypeError):
                # For categorical values like sex, attempt to map string to integer code
                val_str = str(val).strip().lower()
                if val_str in {"male", "m", "1"}:
                    val_float = 1.0
                elif val_str in {"female", "f", "0"}:
                    val_float = 0.0
                else:
                    # Default to NaN if unrecognised
                    val_float = np.nan
            covariate_values.append(val_float)
        # Derive dose_rate and dose_duration from AMT and RATE columns.
        full_group = df[df[id_col] == pid]
        dose_amt = 0.0
        dose_rate = 0.0
        if "AMT" in full_group.columns:
            amt_nonzero = full_group.loc[(full_group["AMT"].fillna(0).astype(float) != 0), "AMT"].astype(float)
            if not amt_nonzero.empty:
                dose_amt = float(amt_nonzero.iloc[0])
        if "RATE" in full_group.columns:
            rate_nonzero = full_group.loc[(full_group["RATE"].fillna(0).astype(float) != 0), "RATE"].astype(float)
            if not rate_nonzero.empty:
                dose_rate = float(rate_nonzero.iloc[0])
        if dose_rate > 0.0 and dose_amt > 0.0:
            dose_duration = float(dose_amt / dose_rate)
        else:
            dose_duration = 0.0
        covariates = np.array(covariate_values, dtype=float)
        records.append(
            PatientRecord(
                id=int(pid),
                times=times,
                values=values,
                covariates=covariates,
                dose_rate=dose_rate,
                dose_duration=dose_duration,
            )
        )
    return records


def prepare_dataset(records: List[PatientRecord]) -> Tuple[PreparedData, MinMaxScaler, MinMaxScaler]:
    """Scale and pad a list of patient records into fixed‑size arrays.

    This function fits global MinMaxScalers to the EEG values and
    covariates, scales the data accordingly and pads each patient's
    time series to the same length using the 'safe_beyond' strategy from
    the context code.  Padding beyond the last observation uses time
    points strictly greater than the maximum observed time to avoid
    duplicate time stamps which can cause issues with diffrax.  The
    padded values for the EEG are filled with the last observed value.

    Args:
        records: List of :class:`PatientRecord` instances.

    Returns:
        A :class:`PreparedData` instance with padded arrays and a
        dictionary of fitted scalers.
    """
    if not records:
        raise ValueError("No records provided to prepare_dataset")
    # Determine maximum sequence length and global maximum time
    lengths = np.array([r.times.shape[0] for r in records], dtype=int)
    T_max = int(lengths.max(initial=0))
    global_max_time = float(max((float(r.times[-1]) for r in records if r.times.size > 0), default=0.0))
    if not np.isfinite(global_max_time) or global_max_time <= 0.0:
        global_max_time = 1.0
    # Fit value scaler on all values concatenated
    all_values = np.concatenate([r.values for r in records if r.values.size > 0], axis=0)
    if all_values.size == 0:
        # No measurements; create dummy scaler
        value_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        value_scaler.fit(np.array([0.0, 1.0]).reshape(-1, 1))
    else:
        value_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        value_scaler.fit(all_values.reshape(-1, 1))
    # Fit covariate+dose scaler per column (each feature independently)
    covars_matrix = np.stack([r.covariates for r in records], axis=0)
    dose_rate_vec = np.array([getattr(r, "dose_rate", 0.0) for r in records], dtype=float).reshape(-1, 1)
    dose_duration_vec = np.array([getattr(r, "dose_duration", 0.0) for r in records], dtype=float).reshape(-1, 1)
    features_matrix = np.concatenate([covars_matrix, dose_rate_vec, dose_duration_vec], axis=1)
    covar_scaler = MinMaxScaler()
    covar_scaler.fit(features_matrix)
    # Allocate padded arrays
    N = len(records)
    t_pad = np.zeros((N, T_max), dtype=np.float32)
    y_pad = np.zeros((N, T_max), dtype=np.float32)
    mask = np.zeros((N, T_max), dtype=bool)
    covars_scaled = np.zeros((N, records[0].covariates.shape[0]), dtype=np.float32)
    dose_rate_arr = np.zeros((N,), dtype=np.float32)
    dose_duration_arr = np.zeros((N,), dtype=np.float32)
    lengths_arr = np.zeros(N, dtype=int)
    ids = np.zeros(N, dtype=int)
    # Populate
    for i, rec in enumerate(records):
        L = rec.times.shape[0]
        lengths_arr[i] = L
        ids[i] = rec.id
        # Scale times globally to [0,1]
        t_scaled = (rec.times / global_max_time).astype(np.float32) if L > 0 else np.array([], dtype=np.float32)
        # Scale values
        v_scaled = (
            value_scaler.transform(rec.values.reshape(-1, 1)).flatten().astype(np.float32)
            if L > 0
            else np.array([], dtype=np.float32)
        )
        # Scale covariates and dose fields together, then split
        dr_val = float(getattr(rec, "dose_rate", 0.0))
        dd_val = float(getattr(rec, "dose_duration", 0.0))
        combo = np.concatenate([rec.covariates.astype(float), [dr_val, dd_val]], axis=0).reshape(1, -1)
        scaled_combo = covar_scaler.transform(combo).astype(np.float32).flatten()
        covars_scaled[i] = scaled_combo[:-2]
        dose_rate_arr[i] = scaled_combo[-2]
        dose_duration_arr[i] = scaled_combo[-1]
        # Fill arrays
        if L > 0:
            t_pad[i, :L] = t_scaled
            y_pad[i, :L] = v_scaled
            mask[i, :L] = True
            # Pad the remainder using 'safe_beyond' strategy
            if T_max > L:
                n_pad = T_max - L
                # Determine delta as difference between last two points or default 1.0
                if L > 1:
                    dt = float(t_scaled[-1] - t_scaled[-2])
                else:
                    dt = 1.0 / max(1.0, global_max_time)
                # Start padding strictly beyond 1.0 to avoid duplicate times
                pad_start = 1.0 + dt
                t_pad[i, L:] = pad_start + np.arange(n_pad, dtype=np.float32) * dt
                y_pad[i, L:] = v_scaled[-1]
                mask[i, L:] = False
        else:
            # If no observations, pad entire row with zeros for values and times
            if T_max > 0:
                t_pad[i, :] = np.linspace(0.0, 1.0, T_max, dtype=np.float32)
                y_pad[i, :] = 0.0
                mask[i, :] = False
    # Convert to JAX arrays
    t_pad_j = jnp.asarray(t_pad)
    y_pad_j = jnp.asarray(y_pad)
    mask_j = jnp.asarray(mask)
    covars_j = jnp.asarray(covars_scaled)
    dose_rate_j = jnp.asarray(dose_rate_arr)
    dose_duration_j = jnp.asarray(dose_duration_arr)
    lengths_j = jnp.asarray(lengths_arr)
    ids_j = jnp.asarray(ids)
    scalers = {
        "value_scaler": value_scaler,
        "covar_scaler": covar_scaler,
        "global_max_time": global_max_time,
    }
    return (
        PreparedData(
            t_pad=t_pad_j,
            y_pad=y_pad_j,
            mask=mask_j,
            covars=covars_j,
            dose_rate=dose_rate_j,
            dose_duration=dose_duration_j,
            lengths=lengths_j,
            ids=ids_j,
            scalers=scalers,
        ),
        value_scaler,
        covar_scaler,
    )


# def prepare_physiology_dataset(
#     records: List[PatientRecord], covariate_cols: List[str]
# ) -> Tuple[PreparedPhysiology, Dict[str, Any]]:
#     """
#     Prepare batched physiology data from patient records.

#     This function computes physiology parameters for all patients and packages
#     them into a PreparedPhysiology instance for efficient vmap operations.

#     Args:
#         records: List of PatientRecord instances
#         covariate_cols: List of covariate column names

#     Returns:
#         PreparedPhysiology instance and scalers dictionary
#     """
#     if not records:
#         raise ValueError("No records provided to prepare_physiology_dataset")

#     N = len(records)

#     # Extract covariates, dose_rate, dose_duration
#     covariates_matrix = np.stack([r.covariates for r in records], axis=0)
#     dose_rate_array = np.array([getattr(r, "dose_rate", 0.0) for r in records], dtype=np.float64)
#     dose_duration_array = np.array([getattr(r, "dose_duration", 0.0) for r in records], dtype=np.float64)
#     patient_ids_array = np.array([r.id for r in records], dtype=int)

#     # Prepare arrays for physiology parameters
#     # Flows: [q_PV, q_HA, q_HV, q_K]
#     flows_matrix = np.zeros((N, 4), dtype=np.float64)

#     # Volumes: [v_L, v_GL, v_SIL, v_LIL, v_GICS, v_T, v_HP, v_P]
#     volumes_matrix = np.zeros((N, 8), dtype=np.float64)

#     # Fixed params: [alpha, Rp, t_G, t_SI, t_LI, k_A_SIL, k_A_LIL, k_A_GL, k_A_L]
#     fixed_params_matrix = np.zeros((N, 9), dtype=np.float64)

#     # Compute physiology for each patient
#     for i, record in enumerate(records):
#         # Create covariate dict
#         covs = {col: record.covariates[j] for j, col in enumerate(covariate_cols)}

#         # Compute physiology using the pbpk_remifentanil module
#         phys = Physiology.from_covariates(covs, dose_rate=dose_rate_array[i], dose_duration=dose_duration_array[i])

#         # Extract flows
#         flows_matrix[i] = [phys.q_PV, phys.q_HA, phys.q_HV, phys.q_K]

#         # Extract volumes
#         volumes_matrix[i] = [phys.v_L, phys.v_GL, phys.v_SIL, phys.v_LIL, phys.v_GICS, phys.v_T, phys.v_HP, phys.v_P]

#         # Extract fixed parameters
#         fixed_params_matrix[i] = [
#             phys.alpha,
#             phys.Rp,
#             phys.t_G,
#             phys.t_SI,
#             phys.t_LI,
#             phys.k_A_SIL,
#             phys.k_A_LIL,
#             phys.k_A_GL,
#             phys.k_A_L,
#         ]

#     # Create scalers for physiology parameters
#     # We'll scale flows, volumes, and covariates but leave fixed params unscaled
#     flows_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
#     flows_scaled = flows_scaler.fit_transform(flows_matrix)

#     volumes_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
#     volumes_scaled = volumes_scaler.fit_transform(volumes_matrix)

#     covariates_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
#     covariates_scaled = covariates_scaler.fit_transform(covariates_matrix)

#     # Create scalers dict
#     scalers = {
#         "flows_scaler": flows_scaler,
#         "volumes_scaler": volumes_scaler,
#         "covariates_scaler": covariates_scaler,
#         "covariate_cols": covariate_cols,
#     }

#     # Convert to JAX arrays
#     prepared_physiology = PreparedPhysiology(
#         covariates=jnp.asarray(covariates_scaled, dtype=jnp.float64),
#         dose_rate=jnp.asarray(dose_rate_array, dtype=jnp.float64),
#         dose_duration=jnp.asarray(dose_duration_array, dtype=jnp.float64),
#         flows=jnp.asarray(flows_scaled, dtype=jnp.float64),
#         volumes=jnp.asarray(volumes_scaled, dtype=jnp.float64),
#         fixed_params=jnp.asarray(fixed_params_matrix, dtype=jnp.float64),  # Keep unscaled
#         patient_ids=jnp.asarray(patient_ids_array, dtype=int),
#         scalers=scalers,
#     )

#     return prepared_physiology, scalers


def prepare_unscaled_dataset(records: List[PatientRecord]) -> UnscaledPreparedData:
    """Prepare padded dataset without any scaling, keeping original units.

    This function creates a padded dataset similar to prepare_dataset but avoids
    all scaling transformations, making it suitable for JAX optimization where
    sklearn scalers cannot be used within traced functions.

    Args:
        records: List of PatientRecord instances.

    Returns:
        UnscaledPreparedData instance with padded arrays in original units.
    """
    if not records:
        raise ValueError("No records provided to prepare_unscaled_dataset")

    # Determine maximum sequence length and global maximum time
    lengths = np.array([r.times.shape[0] for r in records], dtype=int)
    T_max = int(lengths.max(initial=0))
    global_max_time = float(max((float(r.times[-1]) for r in records if r.times.size > 0), default=0.0))
    if not np.isfinite(global_max_time) or global_max_time <= 0.0:
        global_max_time = 1.0

    # Allocate padded arrays
    N = len(records)
    t_pad = np.zeros((N, T_max), dtype=np.float64)  # Keep original precision
    y_pad = np.zeros((N, T_max), dtype=np.float64)
    mask = np.zeros((N, T_max), dtype=bool)
    covars = np.zeros((N, records[0].covariates.shape[0]), dtype=np.float64)
    dose_rate_arr = np.zeros((N,), dtype=np.float64)
    dose_duration_arr = np.zeros((N,), dtype=np.float64)
    lengths_arr = np.zeros(N, dtype=int)
    ids = np.zeros(N, dtype=int)

    # Populate arrays without scaling
    for i, rec in enumerate(records):
        L = rec.times.shape[0]
        lengths_arr[i] = L
        ids[i] = rec.id

        # Keep times and values in original units
        covars[i] = rec.covariates.astype(np.float64)
        dose_rate_arr[i] = float(getattr(rec, "dose_rate", 0.0))
        dose_duration_arr[i] = float(getattr(rec, "dose_duration", 0.0))

        # Fill arrays
        if L > 0:
            t_pad[i, :L] = rec.times.astype(np.float64)
            y_pad[i, :L] = rec.values.astype(np.float64)
            mask[i, :L] = True

            # Pad the remainder using 'safe_beyond' strategy
            if T_max > L:
                n_pad = T_max - L
                # Determine delta as difference between last two points or default
                if L > 1:
                    dt = float(rec.times[-1] - rec.times[-2])
                else:
                    dt = 1.0  # Default time step
                # Start padding beyond the last observed time
                pad_start = float(rec.times[-1]) + dt
                t_pad[i, L:] = pad_start + np.arange(n_pad, dtype=np.float64) * dt
                y_pad[i, L:] = rec.values[-1]  # Keep last observed value
                mask[i, L:] = False
        else:
            # If no observations, pad entire row with default values
            if T_max > 0:
                t_pad[i, :] = np.linspace(0.0, global_max_time, T_max, dtype=np.float64)
                y_pad[i, :] = 0.0
                mask[i, :] = False

    # Convert to JAX arrays
    return UnscaledPreparedData(
        t_pad=jnp.asarray(t_pad),
        y_pad=jnp.asarray(y_pad),
        mask=jnp.asarray(mask),
        covars=jnp.asarray(covars),
        dose_rate=jnp.asarray(dose_rate_arr),
        dose_duration=jnp.asarray(dose_duration_arr),
        lengths=jnp.asarray(lengths_arr),
        ids=jnp.asarray(ids),
        global_max_time=global_max_time,
    )


def split_dataset(
    records: List[PatientRecord],
    split_cfg: DataSplitConfig,
) -> Tuple[List[PatientRecord], List[PatientRecord], List[PatientRecord]]:
    """Split the list of patient records into train/val/test subsets.

    Splits are performed deterministically based on a shuffled ordering of
    unique patient IDs.  The fractions in ``split_cfg`` determine the
    sizes of the subsets.  The validation and test splits are drawn
    sequentially from the remaining patients after allocating the
    training set.

    Args:
        records: List of all patient records.
        split_cfg: Configuration specifying the fractions and random seed.

    Returns:
        A tuple of three lists (train_records, val_records, test_records).
    """
    ids = [r.id for r in records]
    unique_ids = np.unique(ids)
    rng = np.random.default_rng(split_cfg.seed)
    rng.shuffle(unique_ids)
    n_total = len(unique_ids)
    n_train = int(np.floor(split_cfg.train_frac * n_total))
    n_val = int(np.floor(split_cfg.val_frac * n_total))
    train_ids = set(unique_ids[:n_train])
    val_ids = set(unique_ids[n_train : n_train + n_val])
    test_ids = set(unique_ids[n_train + n_val :])
    train_records = [r for r in records if r.id in train_ids]
    val_records = [r for r in records if r.id in val_ids]
    test_records = [r for r in records if r.id in test_ids]
    return train_records, val_records, test_records


def prepare_dataset_with_concentration(
    records: List[PatientRecord],
    pbpk_cfg: PBPKConfig,
    covariate_cols: List[str],
) -> PreparedData:
    """Prepare dataset with dense concentration profiles from NLME kinetic parameters.

    This function orchestrates the PK/PD data preparation by:
    1. Loading pre-computed NLME kinetic parameters from CSV data.
    2. Creating physiological models for each patient.
    3. Using individual kinetic parameters for each patient.
    4. Simulating dense concentration profiles.
    5. Merging the concentration data with the standard prepared EEG data.

    Args:
        records: List of patient records.
        pbpk_cfg: Configuration for the PBPK integration, including kinetic parameters dict.
        covariate_cols: List of covariate column names.

    Returns:
        A PreparedData instance with dense concentration profiles.
    """
    # Check if we have kinetic parameters
    if not pbpk_cfg.nlme_kinetics_dict:
        raise ValueError("No NLME kinetic parameters provided in pbpk_cfg.nlme_kinetics_dict")

    # 1. Prepare standard dataset first
    prepared_data, value_scaler, covar_scaler = prepare_dataset(records)

    # 2. Create Physiological Models for patients that have kinetic parameters
    patients_with_kinetics = []
    kinetics_arrays = []

    for record in records:
        patient_id = record.id
        if patient_id in pbpk_cfg.nlme_kinetics_dict:
            # Create physiological parameters from the record
            phys_params = remifentanil.PhysiologicalParameters.from_patient_record(record)
            patients_with_kinetics.append(phys_params)

            # Get the kinetic parameters for this patient
            kinetic_dict = pbpk_cfg.nlme_kinetics_dict[patient_id]
            param_names = ['k_TP', 'k_PT', 'k_PHP', 'k_HPP', 'k_EL_Pl', 'Eff_kid', 'Eff_hep', 'k_EL_Tis']
            kinetic_array = jnp.array([kinetic_dict[name] for name in param_names])
            kinetics_arrays.append(kinetic_array)

    if not patients_with_kinetics:
        raise ValueError("No patients found with matching NLME kinetic parameters")

    # 3. Stack kinetics for vectorized simulation
    individual_kinetics = jnp.stack(kinetics_arrays)  # Shape: (N_patients, 8)

    # 4. Dense Simulation
    global_max_time = float(max((float(r.times[-1]) for r in records if r.times.size > 0), default=60.0))
    t_dense = jnp.linspace(0.0, global_max_time, pbpk_cfg.dense_sim_points)

    # Vectorized dense simulation
    dense_cs = remifentanil.vectorized_simulate_dense(
        patients_with_kinetics,
        individual_kinetics,
        t_dense=t_dense,
    )

    # 5. Create dense arrays for all patients (padding with zeros for patients without kinetics)
    n_total_patients = len(records)
    # IMPORTANT: Scale dense times to [0,1] to match EEG time scaling
    t_dense_scaled = t_dense / global_max_time  # Scale to [0,1] like EEG times
    dense_ts_padded = jnp.tile(t_dense_scaled[None, :], (n_total_patients, 1))  # (N_total, T_dense)

    # CRITICAL: Fit a separate scaler for PBPK concentrations
    # This ensures PBPK concentrations are scaled to [0,1] independently from EEG data
    concentration_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    if dense_cs.size > 0:
        # Flatten concentrations for scaling
        dense_cs_flat = np.array(dense_cs).flatten()
        # Only fit scaler on non-zero values to avoid issues with zero-padded regions
        nonzero_mask = dense_cs_flat > 0
        if np.any(nonzero_mask):
            # Fit scaler on non-zero concentrations
            concentration_scaler.fit(dense_cs_flat[nonzero_mask].reshape(-1, 1))
            # Apply scaler to all values (zeros will remain zeros after scaling)
            dense_cs_flat_scaled = np.zeros_like(dense_cs_flat)
            dense_cs_flat_scaled[nonzero_mask] = concentration_scaler.transform(
                dense_cs_flat[nonzero_mask].reshape(-1, 1)
            ).flatten()
            # Reshape back to original shape
            dense_cs_scaled = dense_cs_flat_scaled.reshape(dense_cs.shape)
        else:
            # All concentrations are zero, create dummy scaler and keep zeros
            concentration_scaler.fit(np.array([0.0, 1.0]).reshape(-1, 1))
            dense_cs_scaled = np.zeros_like(dense_cs)
    else:
        # No concentrations, create dummy scaler
        concentration_scaler.fit(np.array([0.0, 1.0]).reshape(-1, 1))
        dense_cs_scaled = dense_cs

    dense_cs_padded = jnp.zeros((n_total_patients, pbpk_cfg.dense_sim_points))  # (N_total, T_dense)

    # Fill in the computed concentrations for patients that had kinetics
    kinetics_patient_idx = 0
    for i, record in enumerate(records):
        if record.id in pbpk_cfg.nlme_kinetics_dict:
            dense_cs_padded = dense_cs_padded.at[i].set(dense_cs_scaled[kinetics_patient_idx])
            kinetics_patient_idx += 1

    # 6. Add dense data to prepared dataset and save the concentration scaler
    prepared_data.dense_ts = dense_ts_padded
    prepared_data.dense_cs = dense_cs_padded
    # Store the concentration scaler for potential inverse transforms or reuse
    prepared_data.scalers["concentration_scaler"] = concentration_scaler

    return prepared_data


def plot_eeg_measurements(records: List[PatientRecord], max_patients: int = 10) -> None:
    """Plot EEG measurements for all patients in the dataset.

    This function creates a visualization of the raw EEG time series data
    for each patient to help verify data loading and inspect the patterns.

    Args:
        records: List of PatientRecord instances containing EEG data.
        max_patients: Maximum number of patients to plot (default: 10).
                     If None, plots all patients.
    """
    if not records:
        print("No patient records to plot.")
        return

    # Limit number of patients if specified
    plot_records = records if max_patients is None else records[:max_patients]

    n_patients = len(plot_records)

    # Create subplots - arrange in a reasonable grid
    if n_patients <= 4:
        rows, cols = n_patients, 1
        figsize = (10, 3 * n_patients)
    elif n_patients <= 9:
        rows, cols = int(np.ceil(np.sqrt(n_patients))), int(np.ceil(np.sqrt(n_patients)))
        figsize = (4 * cols, 3 * rows)
    else:
        rows, cols = int(np.ceil(n_patients / 3)), 3
        figsize = (12, 3 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle(f"EEG Measurements for {n_patients} Patients", fontsize=16)

    # Handle case of single subplot
    if n_patients == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    else:
        axes = axes.flatten()

    # Plot each patient's data
    for i, record in enumerate(plot_records):
        ax = axes[i] if i < len(axes) else axes[-1]

        # Plot the time series
        ax.plot(record.times, record.values, "b-", linewidth=1.5, alpha=0.7)
        ax.scatter(record.times, record.values, c="red", s=20, alpha=0.8, zorder=5)

        ax.set_title(f"Patient {record.id}", fontsize=12)
        ax.set_xlabel("Time")
        ax.set_ylabel("EEG Value")
        ax.grid(True, alpha=0.3)

        # Add basic statistics to the plot
        mean_val = np.mean(record.values)
        std_val = np.std(record.values)
        ax.text(
            0.02,
            0.98,
            f"Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nN: {len(record.values)}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Hide any unused subplots
    for i in range(n_patients, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Total patients: {len(records)}")
    print(f"Patients plotted: {n_patients}")

    all_times = np.concatenate([r.times for r in records])
    all_values = np.concatenate([r.values for r in records])

    print(f"Time range: {all_times.min():.2f} to {all_times.max():.2f}")
    print(f"EEG value range: {all_values.min():.2f} to {all_values.max():.2f}")
    print(f"Average measurements per patient: {np.mean([len(r.values) for r in records]):.1f}")
    print(f"Total measurements: {len(all_values)}")

    # Show covariate information for first few patients
    print("\nCovariate examples (first 3 patients):")
    for record in records[:3]:
        print(f"Patient {record.id}: covariates = {record.covariates}")


def plot_eeg_summary(records: List[PatientRecord]) -> None:
    """Create summary plots showing EEG data distribution and characteristics.

    This function creates overview plots to understand the dataset characteristics,
    including measurement count distribution, value distributions, and time ranges.

    Args:
        records: List of PatientRecord instances containing EEG data.
    """
    if not records:
        print("No patient records to analyze.")
        return

    # Extract summary statistics
    measurement_counts = [len(r.values) for r in records]
    all_values = np.concatenate([r.values for r in records])
    time_ranges = [(r.times.max() - r.times.min()) if len(r.times) > 0 else 0 for r in records]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("EEG Dataset Summary Statistics", fontsize=16)

    # Plot 1: Distribution of measurement counts per patient
    ax1.hist(measurement_counts, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
    ax1.set_xlabel("Number of Measurements per Patient")
    ax1.set_ylabel("Number of Patients")
    ax1.set_title("Distribution of Measurement Counts")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Distribution of EEG values
    ax2.hist(all_values, bins=50, alpha=0.7, color="lightgreen", edgecolor="black")
    ax2.set_xlabel("EEG Value")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of EEG Values")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Time range per patient
    ax3.hist(time_ranges, bins=20, alpha=0.7, color="orange", edgecolor="black")
    ax3.set_xlabel("Time Range per Patient")
    ax3.set_ylabel("Number of Patients")
    ax3.set_title("Distribution of Time Ranges")
    ax3.grid(True, alpha=0.3)

    # Plot 4: EEG values over time (all patients overlaid)
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(records), 10)))
    for i, record in enumerate(records[:10]):  # Plot first 10 patients
        ax4.plot(record.times, record.values, alpha=0.6, linewidth=1, color=colors[i], label=f"Patient {record.id}")

    ax4.set_xlabel("Time")
    ax4.set_ylabel("EEG Value")
    ax4.set_title("EEG Time Series (First 10 Patients)")
    ax4.grid(True, alpha=0.3)
    if len(records) <= 10:
        ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()


def plot_all_patients_overlaid(
    records: List[PatientRecord], max_patients: int = None, figsize: Tuple[float, float] = (12, 8), alpha: float = 0.7
) -> None:
    """Plot all patients' EEG measurements on the same figure with overlaid line plots.

    This function creates a single plot with all patient time series overlaid,
    allowing for easy comparison of EEG patterns across patients.

    Args:
        records: List of PatientRecord instances containing EEG data.
        max_patients: Maximum number of patients to plot. If None, plots all patients.
        figsize: Figure size as (width, height) tuple.
        alpha: Transparency level for the lines (0-1, where 1 is opaque).
    """
    if not records:
        print("No patient records to plot.")
        return

    # Limit number of patients if specified
    plot_records = records if max_patients is None else records[:max_patients]
    n_patients = len(plot_records)

    # Create the figure
    plt.figure(figsize=figsize)

    # Generate colors for each patient
    colors = plt.cm.tab20(np.linspace(0, 1, n_patients))

    # Plot each patient's time series
    for i, record in enumerate(plot_records):
        plt.plot(record.times, record.values, color=colors[i], alpha=alpha, linewidth=1.5, label=f"Patient {record.id}")

    plt.xlabel("Time", fontsize=12)
    plt.ylabel("EEG Value", fontsize=12)
    plt.title(f"EEG Measurements - All {n_patients} Patients Overlaid", fontsize=14)
    plt.grid(True, alpha=0.3)

    # Add legend (but be smart about it for many patients)
    if n_patients <= 15:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    else:
        # For many patients, just show a sample in the legend
        handles, labels = plt.gca().get_legend_handles_labels()
        sample_indices = np.linspace(0, len(handles) - 1, min(10, len(handles)), dtype=int)
        sample_handles = [handles[i] for i in sample_indices]
        sample_labels = [labels[i] for i in sample_indices]
        if len(handles) > 10:
            sample_labels.append(f"... and {len(handles) - 10} more patients")
        plt.legend(sample_handles, sample_labels, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.show()

    # Print summary information
    print("\nOverlaid Plot Summary:")
    print(f"Number of patients plotted: {n_patients}")
    print(f"Total patients available: {len(records)}")

    # Calculate and show some interesting statistics
    all_times = np.concatenate([r.times for r in plot_records])
    all_values = np.concatenate([r.values for r in plot_records])

    print(f"Overall time range: {all_times.min():.2f} to {all_times.max():.2f}")
    print(f"Overall EEG value range: {all_values.min():.2f} to {all_values.max():.2f}")
    print(f"Mean EEG value across all patients: {all_values.mean():.2f} ± {all_values.std():.2f}")

    # Show measurement count distribution
    measurement_counts = [len(r.values) for r in plot_records]
    print(
        f"Measurements per patient: {min(measurement_counts)} to {max(measurement_counts)} "
        f"(mean: {np.mean(measurement_counts):.1f})"
    )
