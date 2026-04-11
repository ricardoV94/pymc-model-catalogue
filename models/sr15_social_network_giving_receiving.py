"""
Model: Social Network Giving-Receiving Model with Dyadic Reciprocity
Source: pymc-examples/examples/statistical_rethinking_lectures/15-Social_Networks.ipynb, Section: "Giving-Receiving Model"
Authors: Dustin Stansbury
Description: Poisson model of food-sharing between households in the Koster & Leckie dataset.
    Models correlated social ties (reciprocity) within dyads via LKJ prior, and correlated
    giving/receiving tendencies at the household level via a second LKJ prior.

Changes from original:
- Loaded data from npz instead of using utils.load_data
- Removed sampling, log-likelihood computation, and plotting code

Benchmark results:
- Original:  logp = -6516.0907, grad norm = 2433.1136, 15.6 us/call (100000 evals)
- Frozen:    logp = -6516.0907, grad norm = 2433.1136, 15.2 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data = np.load(
        Path(__file__).parent / "data" / "sr_koster_leckie.npz", allow_pickle=True
    )
    dyad_id = data["did"].astype(int)
    household_A_id = data["hidA"].astype(int)
    household_B_id = data["hidB"].astype(int)
    gifts_AB = data["giftsAB"].astype(int)
    gifts_BA = data["giftsBA"].astype(int)

    n_dyads = len(dyad_id)
    n_correlated_features = 2

    # Data are 1-indexed
    if np.min(dyad_id) == 1:
        dyad_id = dyad_id - 1
        household_A_id = household_A_id - 1
        household_B_id = household_B_id - 1

    n_households = np.max([household_A_id, household_B_id]) + 1

    ETA = 2
    with pm.Model() as model:
        # Single, global alpha
        alpha = pm.Normal("alpha", 0, 1)

        # Social ties interaction; shared sigma
        sigma_T = pm.Exponential.dist(1)
        chol_T, corr_T, std_T = pm.LKJCholeskyCov(
            "rho_T", eta=ETA, n=n_correlated_features, sd_dist=sigma_T
        )
        z_T = pm.Normal("z_T", 0, 1, shape=(n_dyads, n_correlated_features))
        T = pm.Deterministic("T", chol_T.dot(z_T.T).T)

        # Giving-receiving interaction; full covariance
        sigma_GR = pm.Exponential.dist(1, shape=n_correlated_features)
        chol_GR, corr_GR, std_GR = pm.LKJCholeskyCov(
            "rho_GR", eta=ETA, n=n_correlated_features, sd_dist=sigma_GR
        )
        z_GR = pm.Normal(
            "z_GR", 0, 1, shape=(n_households, n_correlated_features)
        )
        GR = pm.Deterministic("GR", chol_GR.dot(z_GR.T).T)

        lambda_AB = pm.Deterministic(
            "lambda_AB",
            pm.math.exp(
                alpha
                + T[dyad_id, 0]
                + GR[household_A_id, 0]
                + GR[household_B_id, 1]
            ),
        )
        lambda_BA = pm.Deterministic(
            "lambda_BA",
            pm.math.exp(
                alpha
                + T[dyad_id, 1]
                + GR[household_B_id, 0]
                + GR[household_A_id, 1]
            ),
        )

        # Record quantities for reporting
        pm.Deterministic("corrcoef_T", corr_T[0, 1])
        pm.Deterministic("std_T", std_T)
        pm.Deterministic("corrcoef_GR", corr_GR[0, 1])
        pm.Deterministic("std_GR", std_GR)

        pm.Poisson("G_AB", lambda_AB, observed=gifts_AB)
        pm.Poisson("G_BA", lambda_BA, observed=gifts_BA)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
