"""
Model: Structural AR timeseries with linear trend and Fourier seasonality
Source: pymc-examples/examples/time_series/Forecasting_with_structural_timeseries.ipynb, Section: "Specifying the Trend + Seasonal Model"
Authors: Nathaniel Forde (Oct 2022)
Description: Bayesian structural timeseries combining a `pm.AR` component, an
    additive linear trend, and an additive Fourier seasonality component
    (10 orders of sine/cosine basis at a period of 7). Fitted to synthetic AR
    data plus linear drift plus a sinusoidal seasonal oscillation.

Changes from original:
- `pm.MutableData` -> `pm.Data` (current API)
- `Model.add_coord(..., mutable=True)` -> `add_coord(...)` (mutable kwarg removed)
- Inlined synthetic data generation with the original RANDOM_SEED
- Removed sampling, forecasting/prediction block, and plotting code

Benchmark results:
- Original:  logp = -1758.9233, grad norm = 696.9571, 8.1 us/call (100000 evals)
- Frozen:    logp = -1758.9233, grad norm = 696.9571, 6.2 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm


def build_model():
    RANDOM_SEED = 8929
    np.random.seed(RANDOM_SEED)

    def simulate_ar(intercept, coef1, coef2, noise=0.3, *, warmup=10, steps=200):
        draws = np.zeros(warmup + steps)
        draws[:2] = intercept
        for step in range(2, warmup + steps):
            draws[step] = (
                intercept
                + coef1 * draws[step - 1]
                + coef2 * draws[step - 2]
                + np.random.normal(0, noise)
            )
        return draws[warmup:]

    ar1_data = simulate_ar(10, -0.9, 0)
    y_t = -0.3 + np.arange(200) * -0.2 + np.random.normal(0, 10, 200)
    y_t = y_t + ar1_data

    t_data_full = list(range(200))
    n_order = 10
    periods = np.array(t_data_full) / 7
    fourier_features = pd.DataFrame(
        {
            f"{func}_order_{order}": getattr(np, func)(2 * np.pi * periods * order)
            for order in range(1, n_order + 1)
            for func in ("sin", "cos")
        }
    )
    y_t_s = y_t + 20 * fourier_features["sin_order_1"]
    y_t_s = y_t_s.to_numpy()

    priors = {
        "coefs": {"mu": [0.2, 0.2], "sigma": [0.5, 0.03], "size": 2},
        "alpha": {"mu": -4, "sigma": 0.1},
        "beta": {"mu": -0.1, "sigma": 0.2},
        "beta_fourier": {"mu": 0, "sigma": 2},
        "sigma": 8,
        "init": {"mu": -4, "sigma": 0.1, "size": 1},
    }

    ff = fourier_features.to_numpy().T
    t_data = list(range(len(y_t_s)))

    with pm.Model() as AR:
        pass

    AR.add_coord("obs_id", t_data)
    AR.add_coord("fourier_features", np.arange(len(ff)))

    with AR:
        t = pm.Data("t", t_data, dims="obs_id")
        y = pm.Data("y", y_t_s, dims="obs_id")
        coefs = pm.Normal("coefs", priors["coefs"]["mu"], priors["coefs"]["sigma"])
        sigma = pm.HalfNormal("sigma", priors["sigma"])
        init = pm.Normal.dist(
            priors["init"]["mu"], priors["init"]["sigma"], size=priors["init"]["size"]
        )
        ar1 = pm.AR(
            "ar",
            coefs,
            sigma=sigma,
            init_dist=init,
            constant=True,
            steps=t.shape[0] - (priors["coefs"]["size"] - 1),
            dims="obs_id",
        )

        alpha = pm.Normal("alpha", priors["alpha"]["mu"], priors["alpha"]["sigma"])
        beta = pm.Normal("beta", priors["beta"]["mu"], priors["beta"]["sigma"])
        trend = pm.Deterministic("trend", alpha + beta * t, dims="obs_id")

        beta_fourier = pm.Normal(
            "beta_fourier",
            mu=priors["beta_fourier"]["mu"],
            sigma=priors["beta_fourier"]["sigma"],
            dims="fourier_features",
        )
        fourier_terms = pm.Data("fourier_terms", ff)
        seasonality = pm.Deterministic(
            "seasonality", pm.math.dot(beta_fourier, fourier_terms), dims="obs_id"
        )

        mu = ar1 + trend + seasonality

        outcome = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=y, dims="obs_id")

    model = AR
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
