"""
Model: Hierarchical Bayesian VAR across 8 countries (GDP, Consumption, Investment)
Source: pymc-examples/examples/time_series/bayesian_var_model.ipynb, Section: "Adding a Bayesian Twist: Hierarchical VARs"
Authors: Nathaniel Forde
Description: A hierarchical VAR(2) model with three equations (dl_gdp, dl_cons, dl_gfcf)
    fitted jointly across 8 countries from the World Bank World Development Indicators
    data. Each country has its own lag coefficients, intercepts, and innovation
    covariance, shrunk toward a global mean via a non-centred parameterisation with a
    shared global LKJ covariance mixed via an rho parameter.

Changes from original:
- Loaded the multi-country dl_gdp/dl_cons/dl_gfcf data from .npz with string country
  column (allow_pickle=True) instead of reading CSV inline
- Removed sampling and plotting code

Benchmark results:
- Original:  logp = -715.6973, grad norm = 363.3989, 4701.3 us/call (1202 evals)
- Frozen:    logp = -715.6973, grad norm = 363.3989, 1391.8 us/call (314 evals)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm


def build_model():
    data = np.load(
        Path(__file__).parent / "data" / "bayesian_var_gdp.npz", allow_pickle=True
    )
    df = pd.DataFrame(
        {
            "country": data["countries"],
            "dl_gdp": data["dl_gdp"],
            "dl_cons": data["dl_cons"],
            "dl_gfcf": data["dl_gfcf"],
        }
    )
    group_field = "country"
    n_lags = 2
    n_eqs = 3

    def calc_ar_step(lag_coefs, n_eqs, n_lags, df):
        ars = []
        for j in range(n_eqs):
            ar = pm.math.sum(
                [
                    pm.math.sum(
                        lag_coefs[j, i] * df.values[n_lags - (i + 1) : -(i + 1)], axis=-1
                    )
                    for i in range(n_lags)
                ],
                axis=0,
            )
            ars.append(ar)
        beta = pm.math.stack(ars, axis=-1)
        return beta

    cols = [col for col in df.columns if col != group_field]
    coords = {"lags": np.arange(n_lags) + 1, "equations": cols, "cross_vars": cols}
    groups = df[group_field].unique()

    with pm.Model(coords=coords) as model:
        rho = pm.Beta("rho", alpha=2, beta=2)
        alpha_hat_location = pm.Normal("alpha_hat_location", 0, 0.1)
        alpha_hat_scale = pm.InverseGamma("alpha_hat_scale", 3, 0.5)
        beta_hat_location = pm.Normal("beta_hat_location", 0, 0.1)
        beta_hat_scale = pm.InverseGamma("beta_hat_scale", 3, 0.5)
        omega_global, _, _ = pm.LKJCholeskyCov(
            "omega_global", n=n_eqs, eta=1.0, sd_dist=pm.Exponential.dist(1)
        )

        for grp in groups:
            df_grp = df[df[group_field] == grp][cols]
            z_scale_beta = pm.InverseGamma(f"z_scale_beta_{grp}", 3, 0.5)
            z_scale_alpha = pm.InverseGamma(f"z_scale_alpha_{grp}", 3, 0.5)
            lag_coefs = pm.Normal(
                f"lag_coefs_{grp}",
                mu=beta_hat_location,
                sigma=beta_hat_scale * z_scale_beta,
                dims=["equations", "lags", "cross_vars"],
            )
            alpha = pm.Normal(
                f"alpha_{grp}",
                mu=alpha_hat_location,
                sigma=alpha_hat_scale * z_scale_alpha,
                dims=("equations",),
            )

            betaX = calc_ar_step(lag_coefs, n_eqs, n_lags, df_grp)
            betaX = pm.Deterministic(f"betaX_{grp}", betaX)
            mean = alpha + betaX

            n = df_grp.shape[1]
            noise_chol, _, _ = pm.LKJCholeskyCov(
                f"noise_chol_{grp}", eta=10, n=n, sd_dist=pm.Exponential.dist(1)
            )
            omega = pm.Deterministic(
                f"omega_{grp}", rho * omega_global + (1 - rho) * noise_chol
            )
            pm.MvNormal(
                f"obs_{grp}", mu=mean, chol=omega, observed=df_grp.values[n_lags:]
            )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
