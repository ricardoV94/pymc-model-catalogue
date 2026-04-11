"""
Model: Structural AR timeseries with linear trend
Source: pymc-examples/examples/time_series/Forecasting_with_structural_timeseries.ipynb, Section: "Specifying a Trend Model"
Authors: Nathaniel Forde (Oct 2022)
Description: Bayesian structural timeseries combining a `pm.AR` component with an
    additive linear trend (alpha + beta * t). Fitted to synthetic AR data plus a
    negative linear drift.

Changes from original:
- `pm.MutableData` -> `pm.Data` (current API)
- `Model.add_coord(..., mutable=True)` -> `add_coord(...)` (mutable kwarg removed)
- Inlined synthetic data generation with the original RANDOM_SEED
- Removed sampling, forecasting/prediction block, and plotting code

Benchmark results:
- Original:  logp = -1414.4057, grad norm = 118.7652, 4.9 us/call (100000 evals)
- Frozen:    logp = -1414.4057, grad norm = 118.7652, 4.7 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
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
    # Trend + AR data
    y_t = -0.3 + np.arange(200) * -0.2 + np.random.normal(0, 10, 200)
    y_t = y_t + ar1_data

    priors = {
        "coefs": {"mu": [0.2, 0.2], "sigma": [0.5, 0.03], "size": 2},
        "alpha": {"mu": -4, "sigma": 0.1},
        "beta": {"mu": -0.1, "sigma": 0.2},
        "sigma": 8,
        "init": {"mu": -4, "sigma": 0.1, "size": 1},
    }

    with pm.Model() as AR:
        pass

    t_data = list(range(len(y_t)))
    AR.add_coord("obs_id", t_data)

    with AR:
        t = pm.Data("t", t_data, dims="obs_id")
        y = pm.Data("y", y_t, dims="obs_id")
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

        mu = ar1 + trend

        outcome = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=y, dims="obs_id")

    model = AR
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
