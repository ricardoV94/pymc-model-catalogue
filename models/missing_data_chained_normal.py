"""
Model: Bayesian Imputation by Chained Equations (Gaussian Sampling Distribution)
Source: pymc-examples/examples/howto/Missing_Data_Imputation.ipynb, Section: "Bayesian Imputation by Chained Equations"
Authors: Nathaniel Forde
Description: Chained regression model where climate and lmx predictor distributions are
    assumed Gaussian and used as observed likelihoods (auto-imputing NaNs), feeding into a
    focal regression for empower on {male, climate, lmx}. Jointly imputes climate, lmx, and
    empower via PyMC's automatic missing-value handling.

Changes from original:
- Inlined data loaded from data/employee_missing.npz (float64) instead of pm.get_data csv.
- Removed sampling / plotting / prior predictive code.

Benchmark results:
- Original:  logp = -18882.1219, grad norm = 13738.5336, 14.7 us/call (100000 evals)
- Frozen:    logp = -18882.1219, grad norm = 13738.5336, 14.7 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    d = np.load(Path(__file__).parent / "data" / "employee_missing.npz")
    lmx = d["lmx"].astype(np.float64)
    empower = d["empower"].astype(np.float64)
    climate = d["climate"].astype(np.float64)
    male = d["male"].astype(np.float64)

    lmx_mean = np.nanmean(lmx)
    lmx_sd = np.nanstd(lmx, ddof=1)
    cli_mean = np.nanmean(climate)
    cli_sd = np.nanstd(climate, ddof=1)

    priors = {
        "climate": {"normal": [lmx_mean, lmx_sd, lmx_sd]},
        "lmx": {"normal": [cli_mean, cli_sd, cli_sd]},
    }

    coords = {
        "alpha_dim": ["lmx_imputed", "climate_imputed", "empower_imputed"],
        "beta_dim": [
            "lmxB_male",
            "lmxB_climate",
            "climateB_male",
            "empB_male",
            "empB_climate",
            "empB_lmx",
        ],
    }
    with pm.Model(coords=coords) as model:
        # Priors
        beta = pm.Normal("beta", 0, 1, size=6, dims="beta_dim")
        alpha = pm.Normal("alphas", 10, 5, size=3, dims="alpha_dim")
        sigma = pm.HalfNormal("sigmas", 5, size=3, dims="alpha_dim")

        mu_climate = pm.Normal(
            "mu_climate",
            priors["climate"]["normal"][0],
            priors["climate"]["normal"][1],
        )
        sigma_climate = pm.HalfNormal(
            "sigma_climate", priors["climate"]["normal"][2]
        )
        climate_pred = pm.Normal(
            "climate_pred", mu_climate, sigma_climate, observed=climate
        )

        mu_lmx = pm.Normal(
            "mu_lmx", priors["lmx"]["normal"][0], priors["lmx"]["normal"][1]
        )
        sigma_lmx = pm.HalfNormal("sigma_lmx", priors["lmx"]["normal"][2])
        lmx_pred = pm.Normal("lmx_pred", mu_lmx, sigma_lmx, observed=lmx)

        # Likelihood(s)
        pm.Normal(
            "lmx_imputed",
            alpha[0] + beta[0] * male + beta[1] * climate_pred,
            sigma[0],
            observed=lmx,
        )
        pm.Normal(
            "climate_imputed",
            alpha[1] + beta[2] * male,
            sigma[1],
            observed=climate,
        )
        pm.Normal(
            "emp_imputed",
            alpha[2]
            + beta[3] * male
            + beta[4] * climate_pred
            + beta[5] * lmx_pred,
            sigma[2],
            observed=empower,
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
