"""
Model: Full Missing Data Imputation for Primate Brain Size with Phylogenetic Covariance
Source: pymc-examples/examples/statistical_rethinking_lectures/18-Missing_Data.ipynb, Section: "Full Model"
Authors: Dustin Stansbury
Description: Full Bayesian imputation model for primate brain size, body mass, and group size
    with missing data handled via MvNormal imputation. Each variable (M, G, B) has its own
    phylogenetic covariance kernel (L1/exponential), allowing imputation of missing body mass
    and group size values while controlling for evolutionary history.

Changes from original:
- Loaded data inline from npz instead of using utils.load_data
- Standardized data inline
- Removed nutpie sampler specification
- Removed sampling and plotting code

Benchmark results:
- Original:  logp = -12588.4357, grad norm = 12526.2000, 7660.9 us/call (2193 evals)
- Frozen:    logp = -12588.4357, grad norm = 12526.2000, 7195.5 us/call (1809 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data = np.load(
        Path(__file__).parent / "data" / "sr_primates301.npz", allow_pickle=True
    )
    name = data["name"]
    brain = data["brain"]
    body = data["body"]
    group_size = data["group_size"]
    distance_matrix = data["distance_matrix"]

    # Filter to cases with brain data (non-NaN brain)
    mask = ~np.isnan(brain)
    idx = np.where(mask)[0]

    name_cc = name[idx]
    brain_cc = brain[idx]
    body_cc = body[idx]
    group_size_cc = group_size[idx]

    # Filter distance matrix and rescale
    D_mat = distance_matrix[np.ix_(idx, idx)]
    D_mat = D_mat / D_mat.max()

    # Standardize log-transformed values (NaN-safe)
    def standardize(x):
        c = x - np.nanmean(x)
        return c / np.nanstd(c)

    G_obs = standardize(np.log(group_size_cc))
    M_obs = standardize(np.log(body_cc))
    B_obs = standardize(np.log(brain_cc))

    PRIMATE = name_cc.tolist()
    coords = {"primate": PRIMATE}

    with pm.Model(coords=coords) as model:
        # Priors
        alpha_B = pm.Normal("alpha_B", 0, 1)
        beta_GB = pm.Normal("beta_GB", 0, 0.5)
        beta_MB = pm.Normal("beta_MB", 0, 0.5)

        alpha_G = pm.Normal("alpha_G", 0, 1)
        beta_MG = pm.Normal("beta_MG", 0, 0.5)

        # M model (imputation)
        eta_squared_M = pm.TruncatedNormal("eta_squared_M", 1, 0.25, lower=0.001)
        rho_M = pm.TruncatedNormal("rho_M", 3, 0.25, lower=0.001)

        K_M = pm.Deterministic("K_M", eta_squared_M * pm.math.exp(-rho_M * D_mat))
        mu_M = pm.math.zeros_like(M_obs)
        M = pm.MvNormal("M", mu=mu_M, cov=K_M, observed=M_obs)

        # G Model (imputation)
        eta_squared_G = pm.TruncatedNormal("eta_squared_G", 1, 0.25, lower=0.001)
        rho_G = pm.TruncatedNormal("rho_G", 3, 0.25, lower=0.001)

        K_G = pm.Deterministic("K_G", eta_squared_G * pm.math.exp(-rho_G * D_mat))
        mu_G = alpha_G + beta_MG * M
        G = pm.MvNormal("G", mu=mu_G, cov=K_G, observed=G_obs)

        # B Model
        eta_squared_B = pm.TruncatedNormal("eta_squared_B", 1, 0.25, lower=0.001)
        rho_B = pm.TruncatedNormal("rho_B", 3, 0.25, lower=0.001)
        K_B = pm.Deterministic("K_B", eta_squared_B * pm.math.exp(-rho_B * D_mat))

        # Likelihood for B
        mu_B = alpha_B + beta_GB * G + beta_MB * M
        pm.MvNormal("B", mu=mu_B, cov=K_B, observed=B_obs)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
