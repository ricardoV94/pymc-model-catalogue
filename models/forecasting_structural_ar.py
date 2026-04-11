"""
Model: Structural AR timeseries (plain AR, constant=True)
Source: pymc-examples/examples/time_series/Forecasting_with_structural_timeseries.ipynb, Section: "Specifying the Model"
Authors: Nathaniel Forde (Oct 2022)
Description: Bayesian structural timeseries using `pm.AR` with a constant term.
    Coefficients have Normal priors, innovation sigma is HalfNormal, and a tight
    Normal init distribution for the lag. Observations are Normal around the AR path.

Changes from original:
- `pm.MutableData` -> `pm.Data` (current API)
- `Model.add_coord(..., mutable=True)` -> `add_coord(...)` (mutable kwarg removed)
- Inlined synthetic data generation with the original RANDOM_SEED
- Removed sampling, forecasting/prediction block, and plotting code

Benchmark results:
- Original:  logp = -1227.7804, grad norm = 338.6025, 4.5 us/call (100000 evals)
- Frozen:    logp = -1227.7804, grad norm = 338.6025, 4.2 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    # Reproduce the notebook's synthetic AR(1) data
    RANDOM_SEED = 8929
    rng = np.random.default_rng(RANDOM_SEED)
    # The notebook calls np.random.normal inside simulate_ar (not the rng),
    # so we seed numpy's legacy global state too for faithfulness.
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

    priors = {
        "coefs": {"mu": [10, 0.2], "sigma": [0.1, 0.1], "size": 2},
        "sigma": 8,
        "init": {"mu": 9, "sigma": 0.1, "size": 1},
    }

    with pm.Model() as AR:
        pass

    t_data = list(range(len(ar1_data)))
    AR.add_coord("obs_id", t_data)

    with AR:
        t = pm.Data("t", t_data, dims="obs_id")
        y = pm.Data("y", ar1_data, dims="obs_id")

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

        outcome = pm.Normal("likelihood", mu=ar1, sigma=sigma, observed=y, dims="obs_id")

    model = AR
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
