"""
Model: Bayesian Imputation by Chained Equations (Uniform Sampling Distribution)
Source: pymc-examples/examples/howto/Missing_Data_Imputation.ipynb, Section: "Bayesian Imputation by Chained Equations"
Authors: Nathaniel Forde
Description: Same chained-equation architecture as the Gaussian variant, but with the climate
    and lmx predictor distributions replaced by Uniform(0, 40) observed likelihoods. Jointly
    imputes climate, lmx, and empower via PyMC's automatic missing-value handling.

Changes from original:
- Inlined data loaded from data/employee_missing.npz (float64) instead of pm.get_data csv.
- Removed sampling / plotting / prior predictive code.

Benchmark results:
- Original:  logp = -14439.1536, grad norm = 11266.9401, 12.5 us/call (100000 evals)
- Frozen:    logp = -14439.1536, grad norm = 11266.9401, 12.9 us/call (100000 evals)
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

        climate_pred = pm.Uniform("climate_pred", 0, 40, observed=climate)
        lmx_pred = pm.Uniform("lmx_pred", 0, 40, observed=lmx)

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
