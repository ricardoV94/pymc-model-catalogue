"""
Model: Rolling Regression with GaussianRandomWalk
Source: pymc-examples/examples/generalized_linear_models/GLM-rolling-regression.ipynb, Section: "Rolling regression"
Authors: Thomas Wiecki
Description: Time-varying linear regression on z-scored GFI/GLD stock prices using
    GaussianRandomWalk priors on intercept and slope, capturing changing
    relationships over time.

Changes from original:
- Loaded z-scored stock price data from .npz file.
- Removed sampling, plotting, and naive model code.

Benchmark results:
- Original:  logp = -48393.0445, grad norm = 112465.2063, 17.2 us/call (100000 evals)
- Frozen:    logp = -48393.0445, grad norm = 112465.2063, 15.8 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data = np.load(Path(__file__).parent / "data" / "stock_prices.npz")
    GFI_zscored = data["GFI_zscored"]
    GLD_zscored = data["GLD_zscored"]

    n_time = len(GFI_zscored)
    coords = {"time": np.arange(n_time)}

    with pm.Model(coords=coords) as model:
        # std of random walk
        sigma_alpha = pm.Exponential("sigma_alpha", 50.0)
        sigma_beta = pm.Exponential("sigma_beta", 50.0)

        alpha = pm.GaussianRandomWalk(
            "alpha", sigma=sigma_alpha, init_dist=pm.Normal.dist(0, 10), dims="time"
        )
        beta = pm.GaussianRandomWalk(
            "beta", sigma=sigma_beta, init_dist=pm.Normal.dist(0, 10), dims="time"
        )

        # Define regression
        regression = alpha + beta * GFI_zscored

        # Assume prices are Normally distributed
        sd = pm.HalfNormal("sd", sigma=0.1)
        pm.Normal("y", mu=regression, sigma=sd, observed=GLD_zscored)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
