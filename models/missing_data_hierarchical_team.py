"""
Model: Hierarchical Team-Level Imputation Model for Empowerment
Source: pymc-examples/examples/howto/Missing_Data_Imputation.ipynb, Section: "Hierarchical Structures and Data Imputation"
Authors: Nathaniel Forde
Description: Hierarchical regression of empower on {lmx, male} with team-level partial pooling
    on the intercept and the lmx slope, employee-level sigma, and an auto-imputed Normal
    sampling distribution for the lmx predictor. PyMC handles imputation of lmx and empower
    missing values automatically from the NaN observed arrays.

Changes from original:
- Inlined data loaded from data/employee_missing.npz (float64) instead of pm.get_data csv.
- Removed sampling / plotting / prior predictive code.

Benchmark results:
- Original:  logp = -5259.3999, grad norm = 2725.3602, 13.8 us/call (100000 evals)
- Frozen:    logp = -5259.3999, grad norm = 2725.3602, 13.7 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm


def build_model():
    d = np.load(Path(__file__).parent / "data" / "employee_missing.npz")
    lmx = d["lmx"].astype(np.float64)
    empower = d["empower"].astype(np.float64)
    male = d["male"].astype(np.float64)
    team_raw = d["team"]
    n_rows = len(team_raw)

    team_idx, teams = pd.factorize(team_raw, sort=True)
    coords = {"team": teams, "employee": np.arange(n_rows)}

    with pm.Model(coords=coords) as model:
        # Priors
        company_beta_lmx = pm.Normal("company_beta_lmx", 0, 1)
        company_beta_male = pm.Normal("company_beta_male", 0, 1)
        company_alpha = pm.Normal("company_alpha", 20, 2)
        team_alpha = pm.Normal("team_alpha", 0, 1, dims="team")
        team_beta_lmx = pm.Normal("team_beta_lmx", 0, 1, dims="team")
        sigma = pm.HalfNormal("sigma", 4, dims="employee")

        # Imputed Predictors
        mu_lmx = pm.Normal("mu_lmx", 10, 5)
        sigma_lmx = pm.HalfNormal("sigma_lmx", 5)
        lmx_pred = pm.Normal("lmx_pred", mu_lmx, sigma_lmx, observed=lmx)

        # Combining Levels
        alpha_global = pm.Deterministic(
            "alpha_global", company_alpha + team_alpha[team_idx]
        )
        beta_global_lmx = pm.Deterministic(
            "beta_global_lmx", company_beta_lmx + team_beta_lmx[team_idx]
        )
        beta_global_male = pm.Deterministic(
            "beta_global_male", company_beta_male
        )

        # Likelihood
        mu = pm.Deterministic(
            "mu",
            alpha_global + beta_global_lmx * lmx_pred + beta_global_male * male,
        )

        pm.Normal(
            "emp_imputed",
            mu,
            sigma,
            observed=empower,
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
