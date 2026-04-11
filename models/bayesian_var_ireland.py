"""
Model: Bayesian VAR(2) for Ireland GDP & Consumption (macroeconomic data)
Source: pymc-examples/examples/time_series/bayesian_var_model.ipynb, Section: "Ireland's Economic Situation"
Authors: Nathaniel Forde
Description: A VAR(2) model with two equations fitted to differenced-and-logged Irish GDP
    and consumption series from the World Bank World Development Indicators, using an
    LKJCholeskyCov prior on the innovation covariance and a multivariate normal likelihood.

Changes from original:
- Loaded Ireland [dl_gdp, dl_cons] slice from .npz instead of reading CSV inline
- Removed `mutable=True` kwarg from `pm.Data` (removed in current PyMC API)
- Removed sampling and plotting code

Benchmark results:
- Original:  logp = -95.4864, grad norm = 69.8885, 143.2 us/call (3308 evals)
- Frozen:    logp = -95.4864, grad norm = 69.8885, 40.2 us/call (2655 evals)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm


def build_model():
    data = np.load(
        Path(__file__).parent / "data" / "bayesian_var_gdp.npz", allow_pickle=True
    )
    ireland_arr = data["ireland_data"]
    df = pd.DataFrame(ireland_arr, columns=["dl_gdp", "dl_cons"])

    n_lags = 2
    n_eqs = 2
    priors = {
        "lag_coefs": {"mu": 0.3, "sigma": 1},
        "alpha": {"mu": 0, "sigma": 0.1},
        "noise_chol": {"eta": 1, "sigma": 1},
        "noise": {"sigma": 1},
    }

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

    coords = {
        "lags": np.arange(n_lags) + 1,
        "equations": df.columns.tolist(),
        "cross_vars": df.columns.tolist(),
        "time": [x for x in df.index[n_lags:]],
    }

    with pm.Model(coords=coords) as model:
        lag_coefs = pm.Normal(
            "lag_coefs",
            mu=priors["lag_coefs"]["mu"],
            sigma=priors["lag_coefs"]["sigma"],
            dims=["equations", "lags", "cross_vars"],
        )
        alpha = pm.Normal(
            "alpha",
            mu=priors["alpha"]["mu"],
            sigma=priors["alpha"]["sigma"],
            dims=("equations",),
        )
        data_obs = pm.Data(
            "data_obs", df.values[n_lags:], dims=["time", "equations"]
        )

        betaX = calc_ar_step(lag_coefs, n_eqs, n_lags, df)
        betaX = pm.Deterministic("betaX", betaX, dims=["time", "equations"])
        mean = alpha + betaX

        n = df.shape[1]
        noise_chol, _, _ = pm.LKJCholeskyCov(
            "noise_chol",
            eta=priors["noise_chol"]["eta"],
            n=n,
            sd_dist=pm.HalfNormal.dist(sigma=priors["noise_chol"]["sigma"]),
        )
        pm.MvNormal(
            "obs", mu=mean, chol=noise_chol, observed=data_obs, dims=["time", "equations"]
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
