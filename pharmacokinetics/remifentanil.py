"""Remifentanil PBPK model.

This module implements the complete remifentanil PBPK model using JAX and Diffrax.
It provides data ingestion helpers, patient data classes and simulation routines
for both single patient and vectorized batched simulations. All commonly used
functions are exposed with standardized naming conventions.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from .personalisedeeg.data import PatientRecord

jax.config.update("jax_enable_x64", True)  # Use double precision for JAX

# Order of the kinetic parameters inside the optimisation vector.
KINETIC_PARAMETER_NAMES = (
    "k_TP",  # tissue → plasma rate [min⁻¹]
    "k_PT",  # plasma → tissue rate [min⁻¹]
    "k_PHP",  # plasma → highly perfused organs [min⁻¹]
    "k_HPP",  # highly perfused organs → plasma [min⁻¹]
    "k_EL_Pl",  # plasma elimination constant [min⁻¹]
    "Eff_kid",  # renal clearance efficiency [fraction]
    "Eff_hep",  # hepatic clearance efficiency [fraction]
    "k_EL_Tis",  # tissue elimination constant [min⁻¹]
)


# ============================================================================
# Patient data structures
# ============================================================================
class RawPatient(eqx.Module):
    """Raw measurements for a single remifentanil patient."""

    id: int
    #
    t_meas: jnp.ndarray  # shape (N,)
    c_meas: jnp.ndarray  # shape (N,)
    mask: jnp.ndarray  # shape (N,)

    dose_rate: float
    dose_duration: float

    age: float
    weight: float
    height: float
    sex: bool  # True for Male, False for Female
    bsa: float


class PhysiologicalParameters(eqx.Module):
    """Physiological parameters (non-kinetic) for a remifentanil patient."""

    id: int
    #
    t_meas: jnp.ndarray  # shape (N,)
    c_meas: jnp.ndarray  # shape (N,)
    mask: jnp.ndarray  # shape (N,)

    dose_rate: float
    dose_duration: float

    age: float
    weight: float
    height: float
    bsa: float
    sex: int  # 1 for Male, 0 for Female

    ## Physiological parameters (non-kinetic)
    alpha: float
    Rp: float
    t_G: float
    t_SI: float
    t_LI: float

    k_A_SIL: float
    k_A_LIL: float
    k_A_GL: float
    k_A_L: float

    q_PV: float  # Portal vein flow rate [cm3/min]
    q_HA: float  # Hepatic artery flow rate [cm3/min]
    q_HV: float  # Hepatic vein flow rate [cm3/min]
    q_K: float  # Kidney flow rate [cm3/min]

    v_L: float  # Liver volume [cm3]
    v_GL: float  # Gastric lumen volume [cm3]
    v_SIL: float  # Small intestine lumen volume [cm3]
    v_LIL: float  # Large intestine lumen volume [cm3]
    v_GICS: float  # GI compartment solid volume [cm3]
    v_T: float  # Tissue volume [cm3]
    v_HP: float  # Highly perfused organs volume [cm3]
    v_P: float  # Plasma volume [cm3]

    @classmethod
    def from_patient_record(cls, record: PatientRecord) -> "PhysiologicalParameters":
        """Create a PhysiologicalParameters object from a PatientRecord.

        This method performs the covariate mapping between the EEG dataset's
        PatientRecord and the PBPK model's PhysiologicalParameters.

        Assumptions:
            - AGE, SEX, LBM are present in the covariates.
            - Lean Body Mass (LBM) is used as a proxy for weight.
            - Height is not available, so BSA is calculated without it.
        """
        # Covariate mapping
        # This is a critical step and depends on the order of covariates in PatientRecord
        # We assume AGE, SEX, LBM
        age = record.covariates[0]
        sex = record.covariates[1]  # 1 for Male, 0 for Female
        weight = record.covariates[2] # Using LBM as weight
        height = 0.0 # Not available in PatientRecord

        # Calculate BSA (Mosteller formula) if not available
        bsa = jnp.sqrt(height * weight / 3600.0) if height > 0 and weight > 0 else 0.0

        # Create a temporary RawPatient-like structure to reuse create_physiological_parameters
        # This is a bit of a hack, but it avoids duplicating the complex physiological calculations.
        temp_raw_patient = RawPatient(
            id=record.id,
            t_meas=jnp.array(record.times),
            c_meas=jnp.array(record.values),
            mask=jnp.ones_like(jnp.array(record.times), dtype=bool),
            dose_rate=record.dose_rate,
            dose_duration=record.dose_duration,
            age=age,
            weight=weight,
            height=height,
            sex=bool(sex),
            bsa=bsa,
        )

        return create_physiological_parameters(temp_raw_patient)

    def to_nlme_covariates(self) -> jnp.ndarray:
        """Extract covariates for the NLME model.

        Returns:
            A JAX array containing the covariates expected by the NLME model:
            [age, weight, height, bsa, dose_rate, dose_duration]
        """
        return jnp.array([self.age, self.weight, self.height, self.bsa, self.dose_rate, self.dose_duration])


def create_physiological_parameters(patient: RawPatient) -> PhysiologicalParameters:
    """Convert a RawPatient to PhysiologicalParameters (without kinetic parameters).

    This is the preferred way to create patient parameters for new code.
    Uses the same calculations as the master implementation in Helperfunctions3.py
    """
    # Body weight in kg
    WT = patient.weight

    # Flow rates – align with MATLAB context (Brown et al. (1997))
    BSA = patient.bsa
    cardiac_output_bsa = 3.5 * BSA * 1000.0
    cardiac_output_wt = 0.084 * WT * 1000.0
    cardiac_output = jnp.where(jnp.isfinite(BSA), cardiac_output_bsa, cardiac_output_wt)

    SEX = 1.0 if patient.sex else 0.0  # 1=male, 0=female
    frac_PV = jnp.where(SEX == 1, 0.19, 0.21)
    frac_HV = jnp.where(SEX == 1, 0.25, 0.27)
    frac_K = jnp.where(SEX == 1, 0.19, 0.17)
    frac_HA = 0.06  # Same for both sexes

    q_PV = 0.54 * (frac_PV * cardiac_output)
    q_HA = 0.54 * (frac_HA * cardiac_output)
    q_HV = 0.54 * (frac_HV * cardiac_output)
    q_K = 0.54 * (frac_K * cardiac_output)

    # Liver volume
    V_L = 0.026 * patient.weight * 1000  # [cm3]

    # GI volumes – match MATLAB context values
    V_GL = 500.0
    V_SIL = jnp.pi * 0.25 * (2.0**2) * 400.0
    V_LIL = jnp.pi * 0.25 * (4.0**2) * 150.0
    V_GICS = 0.0001766 * WT * 1000.0

    # Tissue weights [kg] - matching Helperfunctions3.py
    weight_adipose = 0.214 * patient.weight
    weight_bones = 0.143 * patient.weight
    weight_heart = 0.005 * patient.weight
    weight_skin = 0.037 * patient.weight
    weight_brain = 0.02 * patient.weight
    weight_kidneys = 0.004 * patient.weight
    weight_mscl = 0.4 * patient.weight
    weight_spleen = 0.0003 * patient.weight

    # Specific weights [kg/cm3]
    spweight_adipose = 0.916e-3
    spweight_bones = 1.6e-3
    spweight_heart = 1.03e-3
    spweight_skin = 1.3e-3
    spweight_brain = 1.035e-3
    spweight_kidneys = 1.05e-3
    spweight_mscl = 1.041e-3
    spweight_spleen = 1.05e-3

    # Calculate tissue and highly perfused organ volumes - matching Helperfunctions3.py
    V_T = (
        weight_adipose / spweight_adipose
        + weight_bones / spweight_bones
        + weight_heart / spweight_heart
        + weight_skin / spweight_skin
        + weight_mscl / spweight_mscl
    )  # [cm3]

    V_HP = weight_brain / spweight_brain + weight_kidneys / spweight_kidneys + weight_spleen / spweight_spleen

    # Plasma volume - matching Helperfunctions3.py
    V_P = 0.079 * 0.54 * patient.weight * 1000 / 1.06  # [cm3]

    # Fixed parameters - matching Helperfunctions3.py and model.jl exactly
    alpha = 1.0
    Rp = 3.33
    t_G = 78.0
    t_SI = 238.0
    t_LI = 2034.0
    k_A_SIL = 0.0
    k_A_LIL = 0.0
    k_A_GL = 0.0
    k_A_L = 0.0

    # Convert sex: True (male) -> 1, False (female) -> 0
    sex_int = 1 if patient.sex else 0

    return PhysiologicalParameters(
        id=patient.id,
        t_meas=patient.t_meas,
        c_meas=patient.c_meas,
        mask=patient.mask,
        dose_rate=patient.dose_rate,
        dose_duration=patient.dose_duration,
        age=patient.age,
        weight=patient.weight,
        height=patient.height,
        bsa=patient.bsa,
        sex=sex_int,
        alpha=alpha,
        Rp=Rp,
        t_G=t_G,
        t_SI=t_SI,
        t_LI=t_LI,
        k_A_SIL=k_A_SIL,
        k_A_LIL=k_A_LIL,
        k_A_GL=k_A_GL,
        k_A_L=k_A_L,
        q_PV=q_PV,
        q_HA=q_HA,
        q_HV=q_HV,
        q_K=q_K,
        v_L=V_L,
        v_GL=V_GL,
        v_SIL=V_SIL,
        v_LIL=V_LIL,
        v_GICS=V_GICS,
        v_T=V_T,
        v_HP=V_HP,
        v_P=V_P,
    )


# ============================================================================
# I/O utilities - data reader function
# ============================================================================
def import_patients(filename: str = "nlme-remifentanil.xlsx") -> list[RawPatient]:
    """Load the clinical dataset and build RawPatient objects.

    Parameters
    ----------
    filename : str
        Path to the Excel file containing patient data.

    Returns
    -------
    list[RawPatient]
        List of RawPatient objects with padded measurement arrays.
    """
    df = pd.read_excel(filename)
    # Remove columns with 'Unnamed' in their name and any missing values
    df = df.loc[:, ~((df.columns.str.contains("Unnamed")) & (df.isnull().any()))]

    df.dropna(inplace=True)

    patient_data = []

    # Group by ID (assuming each ID represents a unique patient)
    for patient_id, group in df.groupby("ID"):
        # Sort by time to ensure proper ordering
        group = group.sort_values("Time")

        # Get measurement times and concentrations (where conc is not NaN)
        conc_data = group.dropna(subset=["conc"])
        t_meas = jnp.array(conc_data["Time"].values)
        c_meas = jnp.array(conc_data["conc"].values)

        # Calculate dose duration (max time where rate is not zero)
        rate_data = group[group["Rate"] > 0]
        amt_data = group[group["Amt"] > 0]

        dose_duration = float(rate_data["Time"].iloc[-1]) + (float(amt_data["Amt"].iloc[-1])/float(rate_data["Rate"].iloc[-1]) if not rate_data.empty else 0.0)

        # Get dose rate (assuming it's constant when > 0)
        dose_rate = float(group[group["Rate"] > 0]["Rate"].iloc[0]) if not rate_data.empty else 0.0

        # Get patient characteristics (taking first row since they should be constant per patient)
        patient_info = group.iloc[0]

        patient_data.append(
            {
                "id": int(patient_id),
                "t_meas": t_meas,
                "c_meas": c_meas,
                "dose_rate": dose_rate,
                "dose_duration": dose_duration,
                "age": float(patient_info["Age"]),
                "weight": float(patient_info["Wt"]),
                "height": float(patient_info["Ht"]),
                "sex": patient_info["Sex"] == "Male",
                "bsa": float(patient_info["BSA"]),
            }
        )

    max_len = max(len(p["c_meas"]) for p in patient_data)

    patients = []
    for p_data in patient_data:
        original_len = len(p_data["c_meas"])
        t_meas_padded = jnp.pad(p_data["t_meas"], (0, max_len - original_len), mode="edge")  # shape (max_len,)
        c_meas_padded = jnp.pad(p_data["c_meas"], (0, max_len - original_len))  # shape (max_len,)
        mask = jnp.arange(max_len) < original_len  # shape (max_len,)

        patient = RawPatient(
            id=p_data["id"],
            t_meas=t_meas_padded,
            c_meas=c_meas_padded,
            mask=mask,
            dose_rate=p_data["dose_rate"],
            dose_duration=p_data["dose_duration"],
            age=p_data["age"],
            weight=p_data["weight"],
            height=p_data["height"],
            sex=p_data["sex"],
            bsa=p_data["bsa"],
        )
        patients.append(patient)

    return patients


# ============================================================================
# NONMEM Excel import utility
# ============================================================================


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Return the first existing column name from candidates (case-insensitive)."""
    cols_l = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in cols_l:
            return cols_l[key]
    return None


def import_patients_from_nmexcel(
    filename: str = "datasets/reference/pkpdRemifentanil_with_HT_WT_BSA.xlsx",
) -> list[RawPatient]:
    """Load a NONMEM-style dataset and build RawPatient objects.

    Notes
    -----
    * Concentration measurements are taken from the **DV** column **where YTYPE == 1**.
    * Additional covariates present in the file are ignored for now.
    * Dose information is inferred if possible from columns like RATE or DUR; otherwise set to 0.
    """
    df = pd.read_excel(filename)

    # Resolve required/optional columns with case-insensitive matching
    col_id = _find_column(df, ["ID"]) or "ID"
    col_time = _find_column(df, ["TIME", "Time"]) or "TIME"
    col_dv = _find_column(df, ["DV"]) or "DV"
    col_ytype = _find_column(df, ["YTYPE", "YType"]) or "YTYPE"

    # Optional columns
    col_rate = _find_column(df, ["RATE", "Rate"])  # infusion rate if present
    col_amt = _find_column(df, ["AMT", "Dose", "AMOUNT"])  # total amount if present
    col_dur = _find_column(df, ["DUR", "Duration", "DOSE_DURATION"])  # infusion duration if present
    col_age = _find_column(df, ["AGE", "Age"])  # optional
    col_wt = _find_column(df, ["WT", "Weight", "Wt"])  # kg
    col_ht = _find_column(df, ["HT", "Height", "Ht"])  # cm
    col_sex = _find_column(df, ["SEX", "Sex"])  # could be string or 0/1
    col_bsa = _find_column(df, ["BSA"])  # may be present; else compute if possible

    # Filter to concentration rows only: YTYPE == 1
    if col_ytype in df.columns:
        df = df[df[col_ytype] == 1]

    # Drop rows with missing DV or TIME
    df = df.dropna(subset=[col_time, col_dv])

    # Build per-patient data
    patient_data: list[dict] = []
    for patient_id, group in df.groupby(col_id):
        group = group.sort_values(col_time)

        # Measurement times and concentrations from DV
        t_meas = jnp.array(group[col_time].astype(float).values)
        c_meas = jnp.array(group[col_dv].astype(float).values)

        # Dose duration & rate inference: exactly one non-zero RATE and one non-zero AMT per ID
        dose_duration = 0.0
        dose_rate = 0.0
        if (col_rate is not None and col_rate in group.columns) and (col_amt is not None and col_amt in group.columns):
            rate_nonzero = group[col_rate].fillna(0.0)
            amt_nonzero = group[col_amt].fillna(0.0)
            rate_nonzero = rate_nonzero[rate_nonzero != 0.0]
            amt_nonzero = amt_nonzero[amt_nonzero != 0.0]
            if (not rate_nonzero.empty) and (not amt_nonzero.empty):
                dose_rate = float(rate_nonzero.iloc[0])
                amt_val = float(amt_nonzero.iloc[0])
                if dose_rate != 0.0:
                    dose_duration = amt_val / dose_rate
        # If either column is missing or zeros only, leave defaults at 0.0

        # Covariates (optional; ignore extras). Provide fallbacks if missing.
        age = float(group[col_age].iloc[0]) if col_age and not group[col_age].isna().all() else 0.0
        weight = float(group[col_wt].iloc[0]) if col_wt and not group[col_wt].isna().all() else 0.0
        height = float(group[col_ht].iloc[0]) if col_ht and not group[col_ht].isna().all() else 0.0

        # Sex handling: numeric or string
        sex_val = False
        if col_sex and col_sex in group.columns:
            s0 = group[col_sex].iloc[0]
            if isinstance(s0, str):
                sex_val = str(s0).strip().lower().startswith("m")  # 'Male' -> True
            else:
                try:
                    sex_val = bool(int(s0) == 1)
                except Exception:
                    sex_val = False

        # BSA: use column if available; else compute if possible (Mosteller)
        if col_bsa and not group[col_bsa].isna().all():
            bsa = float(group[col_bsa].iloc[0])
        else:
            bsa = float(np.sqrt(max(height, 0.0) * max(weight, 0.0) / 3600.0)) if (height > 0 and weight > 0) else 0.0

        patient_data.append(
            {
                "id": int(patient_id),
                "t_meas": t_meas,
                "c_meas": c_meas,
                "dose_rate": dose_rate,
                "dose_duration": dose_duration,
                "age": age,
                "weight": weight,
                "height": height,
                "sex": sex_val,
                "bsa": bsa,
            }
        )

    if len(patient_data) == 0:
        return []

    # Pad to equal length like the original importer
    max_len = max(len(p["c_meas"]) for p in patient_data)
    patients: list[RawPatient] = []
    for p_data in patient_data:
        original_len = len(p_data["c_meas"])
        t_meas_padded = jnp.pad(p_data["t_meas"], (0, max_len - original_len), mode="edge")
        c_meas_padded = jnp.pad(p_data["c_meas"], (0, max_len - original_len))
        mask = jnp.arange(max_len) < original_len

        patients.append(
            RawPatient(
                id=p_data["id"],
                t_meas=t_meas_padded,
                c_meas=c_meas_padded,
                mask=mask,
                dose_rate=p_data["dose_rate"],
                dose_duration=p_data["dose_duration"],
                age=p_data["age"],
                weight=p_data["weight"],
                height=p_data["height"],
                sex=p_data["sex"],
                bsa=p_data["bsa"],
            )
        )

    return patients


# ============================================================================
# Model parameters
# ============================================================================
def get_default_parameters() -> tuple[list[str], jnp.ndarray]:
    """Return kinetic parameter names and a typical initial vector."""
    example = jnp.array(
        [
            # 0.03538742575982093,
            # 1.970013557855141,
            # 1.8928271523721207,
            # 0.17767543157739724,
            # 5.525652727926651,
            # 0.5548170936796774,
            # 0.4866315841958292,
            # 0.7695315126895349,
            ### DE results Best loss function value: 149.588602
            0.482537,
            0.234613,
            0.281595,
            0.347571,
            0.113072,
            0.459964,
            0.091068,
            1.544213,
        ]
    )
    return list(KINETIC_PARAMETER_NAMES), example


def get_abbiati_parameters() -> tuple[list[str], jnp.ndarray]:
    """Return kinetic parameter names and a typical initial vector."""
    ### --- Abbiati Parameters ---
    # These parameters are fixed and used in the Abbiati et al. model.
    ## Commented variables refer to the paper notations
    # abbiati_parameters = {
    #     "k_TP": 0.28, # j_PT-P
    #     "k_PT": 0.4791, # j_P-PT
    #     "k_PHP": 0.6626, # j_P_HO
    #     "k_HPP": 0.0465, # j_HO-P
    #     "k_EL_Pl": 1.7324, # k_E ^ P
    #     "Eff_kid": 0.394, # Eff^K
    #     "Eff_hep": 0.144, # Eff^H
    #     "k_EL_Tis": 0.063, # k_E ^ T
    # }

    example = jnp.array(
        [
            0.28,  # k_TP
            0.4791,  # k_PT
            0.6626,  # k_PHP
            0.0465,  # k_HPP
            1.7324,  # k_EL_Pl
            0.394,  # Eff_kid
            0.144,  # Eff_hep
            0.063,  # k_EL_Tis
        ]
    )
    return list(KINETIC_PARAMETER_NAMES), example


# ============================================================================
# Simulation utilities
# ============================================================================


class _RemifentanilODE(eqx.Module):
    """ODE system for remifentanil PBPK model."""

    params: PhysiologicalParameters
    kinetics: jnp.ndarray  # kinetic parameters array

    def __call__(self, t, y, args):
        """
        ODE system for remifentanil PBPK model.

        State variables:
          y[0] = C_GL (Gastric lumen)
          y[1] = C_SIL (Small intestine lumen)
          y[2] = C_LIL (Large intestine lumen)
          y[3] = C_Plasma
          y[4] = C_Tiss (Tissue)
          y[5] = C_GICS (GI compartment solid)
          y[6] = C_Liv (Liver)
          y[7] = C_HP (Highly perfused organs)
          y[8] = A_EL_LIL (eliminated from large intestine)
          y[9] = A_EL_Ren (renal elimination accumulator)
          y[10] = A_EL_Hep (hepatic + tissue elimination accumulator)
        """

        # Unpack kinetic parameters
        k_TP, k_PT, k_PHP, k_HPP, k_EL_Pl, Eff_kid, Eff_hep, k_EL_Tis = self.kinetics

        # Unpack state variables
        c_GL, c_SIL, c_LIL, C_Plasma, C_Tiss, c_GICS, C_Liv, c_HP = y[:8]

        # Calculate injection rate
        IR = jnp.where(
            t < self.params.dose_duration,
            self.params.dose_rate * 1000,  # Convert to appropriate units
            0.0,
        )

        # d(C_GL)/dt
        dC_GL_dt = -c_GL / self.params.t_G

        # d(C_SIL)/dt
        dC_SIL_dt = (
            c_GL / self.params.t_G
            - self.params.k_A_SIL * c_SIL
            - c_SIL / self.params.t_SI
            + (c_GICS * self.params.k_A_SIL * self.params.v_GICS) / (self.params.v_SIL * self.params.Rp)
        )

        # d(C_LIL)/dt
        dC_LIL_dt = (
            c_SIL / self.params.t_SI
            - self.params.k_A_LIL * c_LIL
            - c_LIL / self.params.t_LI
            + (c_GICS * self.params.k_A_LIL * self.params.v_GICS) / (self.params.v_LIL * self.params.Rp)
        )

        # d(C_Plasma)/dt
        dC_Plasma_dt = (
            -C_Plasma
            / self.params.v_P
            * (
                k_PT * self.params.v_P / self.params.Rp
                + k_PHP * self.params.v_P / self.params.Rp
                + self.params.q_HA
                + self.params.q_PV
            )
            + C_Tiss / self.params.v_P * (k_TP * self.params.v_T)
            + C_Liv / self.params.v_P * self.params.q_HV
            + c_HP / self.params.v_P * (k_HPP * self.params.v_HP)
            + IR / self.params.v_P
            - C_Plasma * k_EL_Pl / self.params.Rp
            - (C_Plasma**self.params.alpha) * Eff_kid * self.params.q_K / self.params.v_P
        )

        # d(C_Tiss)/dt
        dC_Tiss_dt = (
            C_Plasma / self.params.Rp * k_PT * self.params.v_P / self.params.v_T - C_Tiss * k_TP - C_Tiss * k_EL_Tis
        )

        # d(C_GICS)/dt
        dC_GICS_dt = (
            c_SIL * self.params.k_A_SIL * self.params.v_SIL / self.params.v_GICS
            + c_LIL * self.params.k_A_LIL * self.params.v_LIL / self.params.v_GICS
            + C_Plasma * self.params.q_PV / self.params.v_GICS
            - c_GICS
            * (self.params.q_PV / self.params.v_GICS + (self.params.k_A_SIL + self.params.k_A_LIL) / self.params.Rp)
        )

        # d(C_Liv)/dt
        dC_Liv_dt = (
            C_Plasma / self.params.v_L * self.params.q_HA
            + c_GICS / self.params.v_L * self.params.q_PV
            - C_Liv / self.params.v_L * (self.params.q_HV + Eff_hep * self.params.q_PV)
        )

        # d(C_HP)/dt
        dC_HP_dt = C_Plasma / self.params.Rp * k_PHP * self.params.v_P / self.params.v_HP - c_HP * k_HPP

        # Elimination accumulators
        dA_EL_LIL_dt = c_LIL / self.params.t_LI * self.params.v_LIL
        dA_EL_Ren_dt = (C_Plasma**self.params.alpha) * Eff_kid * self.params.q_K + (
            C_Plasma / self.params.Rp
        ) * k_EL_Pl * self.params.v_P
        dA_EL_Hep_dt = C_Liv * Eff_hep * self.params.q_PV + C_Tiss * k_EL_Tis * self.params.v_T

        return jnp.array(
            [
                dC_GL_dt,
                dC_SIL_dt,
                dC_LIL_dt,
                dC_Plasma_dt,
                dC_Tiss_dt,
                dC_GICS_dt,
                dC_Liv_dt,
                dC_HP_dt,
                dA_EL_LIL_dt,
                dA_EL_Ren_dt,
                dA_EL_Hep_dt,
            ]
        )


def _simulate_single_patient_separated(
    physio_params: PhysiologicalParameters,
    kinetic_vec: jnp.ndarray,
    *,
    save_every=None,
):
    """Run the PBPK ODE system with separated kinetic parameters.

    This is the preferred way to simulate patients for new code.
    """
    # Extract kinetic parameters
    k_TP, k_PT, k_PHP, k_HPP, k_EL_Pl, Eff_kid, Eff_hep, k_EL_Tis = kinetic_vec

    # Create ODE system with updated parameters
    ode_system = _RemifentanilODE(params=physio_params, kinetics=kinetic_vec)

    # Initial conditions (all compartments start at zero)
    y0 = jnp.zeros(11)

    t_end = physio_params.t_meas[-1] + 1  # Extend simulation time beyond last measurement

    # Time points
    if isinstance(save_every, int):
        save_at = jnp.linspace(0.0, t_end, save_every)
    else:
        save_at = physio_params.t_meas

    # Setup solver with jump at end of injection
    term = diffrax.ODETerm(ode_system)
    # solver = diffrax.Kvaerno5()
    solver = diffrax.Tsit5()  # Use Tsit5 for better stability
    # solver = diffrax.Kvaerno3()  # Use Kvaerno3

    # Use PIDController with jump_ts to handle injection discontinuity
    stepsize_controller = diffrax.PIDController(
        rtol=1e-5,
        atol=1e-6,
        # dtmax=0.1,
        pcoeff=0.4,
        icoeff=0.3,
        dcoeff=0.0,
        jump_ts=jnp.array([physio_params.dose_duration]),  # Jump at end of injection
    )

    # Solve ODE
    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=t_end,
        dt0=None,
        y0=y0,
        saveat=diffrax.SaveAt(ts=save_at),
        stepsize_controller=stepsize_controller,
        max_steps=int(1e7),
    )

    # Extract results
    times = solution.ts
    concentrations = solution.ys

    # Create results dictionary
    results = {
        "times": times,
        "C_GL": concentrations[:, 0],
        "C_SIL": concentrations[:, 1],
        "C_LIL": concentrations[:, 2],
        "C_Plasma": concentrations[:, 3],
        "C_Tiss": concentrations[:, 4],
        "C_GICS": concentrations[:, 5],
        "C_Liv": concentrations[:, 6],
        "C_HP": concentrations[:, 7],
        "A_EL_LIL": concentrations[:, 8],
        "A_EL_Ren": concentrations[:, 9],
        "A_EL_Hep": concentrations[:, 10],
        "physio_params": physio_params,
        "kinetic_vec": kinetic_vec,
    }

    return results


@eqx.filter_jit
def _vectorized_simulate_impl_separated(physio_parameters, kinetic_vec):
    """Vectorised simulation of multiple patients with separated parameters."""

    def simulate_single(pp):
        results = _simulate_single_patient_separated(pp, kinetic_vec)
        return results["C_Plasma"]

    return jax.vmap(simulate_single)(physio_parameters)


# ============================================================================
# Batching utilities for training
# ============================================================================


def create_patient_batches(
    physio_params: Sequence[PhysiologicalParameters], covariates: jnp.ndarray, batch_size: int, key: jax.random.PRNGKey
) -> tuple[list[Sequence[PhysiologicalParameters]], list[jnp.ndarray]]:
    """Create random batches of patients for mini-batch training.

    Parameters
    ----------
    physio_params : Sequence[PhysiologicalParameters]
        The physiological parameters for all patients.
    covariates : jnp.ndarray
        The covariate matrix for all patients.
    batch_size : int
        Number of patients per batch.
    key : jax.random.PRNGKey
        Random key for shuffling.

    Returns
    -------
    tuple
        (batched_physio_params, batched_covariates) - lists of batches
    """
    n_patients = len(physio_params)

    # Shuffle indices
    shuffled_indices = jax.random.permutation(key, n_patients)

    # Create batches
    batched_physio = []
    batched_covariates = []

    for i in range(0, n_patients, batch_size):
        end_idx = min(i + batch_size, n_patients)
        batch_indices = shuffled_indices[i:end_idx]

        # Extract batch of physiological parameters
        batch_physio = [physio_params[int(idx)] for idx in batch_indices]

        # Extract batch of covariates
        batch_cov = covariates[batch_indices]

        batched_physio.append(batch_physio)
        batched_covariates.append(batch_cov)

    return batched_physio, batched_covariates


def get_random_patient_batch(
    physio_params: Sequence[PhysiologicalParameters], covariates: jnp.ndarray, batch_size: int, key: jax.random.PRNGKey
) -> tuple[Sequence[PhysiologicalParameters], jnp.ndarray]:
    """Get a single random batch of patients.

    Parameters
    ----------
    physio_params : Sequence[PhysiologicalParameters]
        The physiological parameters for all patients.
    covariates : jnp.ndarray
        The covariate matrix for all patients.
    batch_size : int
        Number of patients in the batch.
    key : jax.random.PRNGKey
        Random key for sampling.

    Returns
    -------
    tuple
        (batch_physio_params, batch_covariates)
    """
    n_patients = len(physio_params)
    batch_size = min(batch_size, n_patients)  # Don't exceed available patients

    # Sample random indices without replacement
    indices = jax.random.choice(key, n_patients, (batch_size,), replace=False)

    # Extract batch
    batch_physio = [physio_params[int(idx)] for idx in indices]
    batch_covariates = covariates[indices]

    return batch_physio, batch_covariates


# ============================================================================
# Public API functions
# ============================================================================
def simulate_patient_separated(physio_params: PhysiologicalParameters, kinetic_vec: jnp.ndarray):
    """Simulate a single remifentanil patient with separated parameters.

    This is the preferred way to simulate patients for new code.

    Parameters
    ----------
    physio_params : PhysiologicalParameters
        The physiological parameters (without kinetics).
    kinetic_vec : jnp.ndarray
        Array of kinetic parameters in KINETIC_PARAMETER_NAMES order.

    Returns
    -------
    tuple
        (times, concentrations) arrays.
    """
    res = _simulate_single_patient_separated(physio_params, kinetic_vec)
    return res["times"], res["C_Plasma"]


def vectorized_simulate_separated(physio_patients: Sequence[PhysiologicalParameters], kinetic_vec: jnp.ndarray):
    """Simulate multiple remifentanil patients with separated parameters.

    This is the preferred way to simulate multiple patients for new code.

    Parameters
    ----------
    physio_patients : Sequence[PhysiologicalParameters]
        The physiological parameters (without kinetics) for each patient.
    kinetic_vec : jnp.ndarray
        Array of kinetic parameters in KINETIC_PARAMETER_NAMES order.

    Returns
    -------
    jnp.ndarray
        Predicted concentrations for all patients.
    """
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *physio_patients)
    return _vectorized_simulate_impl_separated(stacked, kinetic_vec)


def simulate_patient_dense(
    physio_params: PhysiologicalParameters,
    kinetic_vec: jnp.ndarray,
    t_dense: jnp.ndarray,
):
    """
    Simulate a single remifentanil patient and return concentrations at dense time points.
    """
    ode_system = _RemifentanilODE(params=physio_params, kinetics=kinetic_vec)
    y0 = jnp.zeros(11)
    t0 = t_dense[0]
    t1 = t_dense[-1]

    term = diffrax.ODETerm(ode_system)
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(
        rtol=1e-5,
        atol=1e-6,
        jump_ts=jnp.array([physio_params.dose_duration]),
    )

    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=None,
        y0=y0,
        saveat=diffrax.SaveAt(ts=t_dense),
        stepsize_controller=stepsize_controller,
        max_steps=int(1e7),
    )

    return solution.ts, solution.ys[:, 3] # return times and plasma concentration (index 3)

def vectorized_simulate_dense(
    physio_patients: Sequence[PhysiologicalParameters],
    kinetic_params_batch: jnp.ndarray, # Shape (N, num_kin_params)
    t_dense: jnp.ndarray, # Shape (T_dense,)
):
    """
    Vectorized simulation for a batch of patients with dense output.

    Note: We can't use @eqx.filter_jit here because patients have different-shaped
    measurement arrays (t_meas, c_meas, mask). Instead, we iterate over patients.
    """
    concentrations = []

    for i, physio in enumerate(physio_patients):
        kinetic = kinetic_params_batch[i]
        _, concs = simulate_patient_dense(physio, kinetic, t_dense)
        concentrations.append(concs)

    return jnp.stack(concentrations)


__all__ = [
    "RawPatient",
    "PhysiologicalParameters",
    "import_patients",
    "import_patients_from_nmexcel",
    "create_physiological_parameters",
    "simulate_patient_separated",
    "vectorized_simulate_separated",
    "simulate_patient_dense",
    "vectorized_simulate_dense",
    "get_default_parameters",
    "get_abbiati_parameters",
    "create_patient_batches",
    "get_random_patient_batch",
    "KINETIC_PARAMETER_NAMES",
]
