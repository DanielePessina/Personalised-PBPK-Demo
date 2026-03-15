"""
Remi Node Project Package

This package provides modules for building and training Neural Ordinary
Differential Equation (NODE) models tailored to the remifentanil case study.
It includes utilities for parsing raw measurement data from Excel,
preprocessing and padding irregular time series, performing various
dimensionality reduction techniques on patient covariates, defining the
NODE architecture itself, and orchestrating the training loop.  The
code is intentionally modular so that users can switch between
different dimensionality reduction methods (PCA, t‑SNE, or a learnable
MLP) simply by changing the configuration.

At a high level the workflow is:

1.  Use :mod:`remi_node_project.data` to load the dataset from an Excel
    file, extract per‑patient time series for the EEG signal (identified
    by YTYPE == 2), and assemble a list of patient records.
2.  Call :func:`remi_node_project.data.prepare_dataset` to scale and
    pad the variable‑length sequences into fixed‑size arrays suitable
    for batching.  This function also normalises covariates using
    MinMax scaling and returns the fitted scalers.
3.  Choose a dimensionality reduction method in
    :mod:`remi_node_project.dim_reduction` by configuring
    :class:`remi_node_project.config.DimReductionConfig`.  Non‑learnable
    methods (PCA, t‑SNE) are fitted on the training covariates and
    provide a constant latent vector per patient; the learnable MLP
    reducer is trained jointly with the NODE.
4.  Instantiate the model architecture from
    :mod:`remi_node_project.model` using a
    :class:`remi_node_project.config.ModelConfig`, and train it via
    :func:`remi_node_project.trainer.train_model` using a
    :class:`remi_node_project.config.TrainingConfig`.  The training
    function handles curriculum learning with different sequence
    lengths and learning rates, computes masked losses for padded
    sequences, and maintains the proper gradient flow through the
    learnable reducer if selected.

The package makes heavy use of JAX, Equinox and Diffrax for defining
and solving the neural ODE, and scikit‑learn for the classical
dimensionality reduction techniques.  Assertions are sprinkled
throughout the code to ensure that unexpected shapes are caught
early.
"""

from .config import ModelConfig, TrainingConfig, DimReductionConfig, DataSplitConfig
from .data import PatientRecord, PreparedData, load_raw_dataset, prepare_dataset
from .kalman import KalmanSmoother1D, smooth_patients_with_kalman
from .dim_reduction import PCAReducer, TSNEReducer, MLPReducer
from .model import NeuralODEModel, FullModel
from .trainer import train_model, evaluate_model
# from .pbpk_remifentanil import (
#     Physiology,
#     KINETIC_PARAMETER_NAMES as PBPK_KINETIC_PARAMETER_NAMES,
#     simulate_concentration as simulate_remifentanil_concentration,
#     read_fixef_csv,
#     read_ranef_csv,
#     read_individual_kinetics_csv,
#     kinetics_for_id_from_files,
#     covariate_dict_from_arrays,
# )
from .plotting import plot_parity

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "DimReductionConfig",
    "DataSplitConfig",
    "PatientRecord",
    "PreparedData",
    "load_raw_dataset",
    "prepare_dataset",
    "KalmanSmoother1D",
    "smooth_patients_with_kalman",
    "PCAReducer",
    "TSNEReducer",
    "MLPReducer",
    "NeuralODEModel",
    "FullModel",
    "train_model",
    "evaluate_model",
    "plot_parity",
    # PBPK ODE utilities
    # "Physiology",
    # "PBPK_KINETIC_PARAMETER_NAMES",
    # "simulate_remifentanil_concentration",
    # "read_fixef_csv",
    # "read_ranef_csv",
    # "read_individual_kinetics_csv",
    # "kinetics_for_id_from_files",
    # "covariate_dict_from_arrays",
]
