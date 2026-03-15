#!/usr/bin/env python3
"""Smoke-test remifentanil batching and a simple batched forward pass."""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pharmacokinetics.remifentanil as pk
from sklearn.preprocessing import MinMaxScaler

jax.config.update("jax_enable_x64", True)

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    """Run a lightweight batching smoke test."""
    print("Testing batched training implementation...")

    patients_list = pk.import_patients(str(REPO_ROOT / "nlme-remifentanil.xlsx"))[:8]
    physio_parameters = [pk.create_physiological_parameters(p) for p in patients_list]

    def get_patient_covariates(patients):
        covariates = []
        for p in patients:
            covariates.append(jnp.array([p.age, p.weight, p.height, p.sex, p.dose_rate, p.dose_duration]))
        return jnp.stack(covariates)

    covariates = get_patient_covariates(physio_parameters)

    # Scale covariates
    scaler = MinMaxScaler(feature_range=(-1, 1))
    covariates_scaled = jnp.array(scaler.fit_transform(covariates))

    print(f"Total patients: {len(physio_parameters)}")
    print(f"Covariates shape: {covariates.shape}")

    # Test batching
    key = jax.random.PRNGKey(42)
    batch_size = 4
    batch_physio, batch_cov = pk.get_random_patient_batch(
        physio_parameters, covariates_scaled, batch_size, key
    )

    print(f"Batch size: {len(batch_physio)}")
    print(f"Batch covariates shape: {batch_cov.shape}")

    # Test that we can create a model and do a forward pass
    init_kin = jnp.array([
        0.03538742575982093,  # k_TP
        1.970013557855141,    # k_PT
        1.8928271523721207,   # k_PHP (will be fixed)
        0.17767543157739724,  # k_HPP (will be fixed)
        5.525652727926651,    # k_EL_Pl
        0.5548170936796774,   # Eff_kid (will be fixed)
        0.4866315841958292,   # Eff_hep (will be fixed)
        0.7695315126895349,   # k_EL_Tis (will be fixed)
    ])

    fixed_params = {
        "k_PHP": 0.6626,
        "k_HPP": 0.0465,
        "Eff_kid": 0.394,
        "Eff_hep": 0.144,
        "k_EL_Tis": 0.063,
    }

    # Create a small model
    mlp_key = jax.random.PRNGKey(333)
    model = eqx.nn.MLP(
        in_size=int(covariates_scaled.shape[1]),
        out_size=len(init_kin) - len(fixed_params),  # 3 parameters to predict
        width_size=32,
        depth=2,
        key=mlp_key,
        final_activation=jax.nn.swish,
    )

    # Test forward pass
    predicted_params = jax.vmap(model)(batch_cov)
    print(f"Predicted parameters shape: {predicted_params.shape}")

    print("Batched training smoke test passed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
