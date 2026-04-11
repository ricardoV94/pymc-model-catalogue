"""
Model: Truncated Linear Regression
Source: pymc-examples/examples/generalized_linear_models/GLM-truncated-censored-regression.ipynb, Section: "Truncated regression model"
Authors: Benjamin T. Vincent
Description: Linear regression with Truncated Normal likelihood to correctly handle
    data that has been filtered to fall within bounds [-5, 5], correcting for
    selection bias in the truncation process.

Changes from original:
- Inlined synthetic data generation with fixed seed (12345).
- Removed sampling, plotting, and comparison code.

Benchmark results:
- Original:  logp = -551.0381, grad norm = 1177.6233, 15.3 us/call (100000 evals)
- Frozen:    logp = -551.0381, grad norm = 1177.6233, 15.8 us/call (100000 evals)
"""

import numpy as np
import pymc as pm

from numpy.random import default_rng


def build_model():
    rng = default_rng(12345)

    slope, intercept, sigma, N = 1, 0, 2, 200
    x = rng.uniform(-10, 10, N)
    y = rng.normal(loc=slope * x + intercept, scale=sigma)

    # Truncate: keep only points where y is within bounds
    bounds = [-5, 5]
    keep = (y >= bounds[0]) & (y <= bounds[1])
    xt = x[keep]
    yt = y[keep]

    with pm.Model() as model:
        slope_rv = pm.Normal("slope", mu=0, sigma=1)
        intercept_rv = pm.Normal("intercept", mu=0, sigma=1)
        sigma_rv = pm.HalfNormal("\u03c3", sigma=1)
        normal_dist = pm.Normal.dist(mu=slope_rv * xt + intercept_rv, sigma=sigma_rv)
        pm.Truncated("obs", normal_dist, lower=bounds[0], upper=bounds[1], observed=yt)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
