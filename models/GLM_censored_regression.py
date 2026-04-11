"""
Model: Censored Linear Regression
Source: pymc-examples/examples/generalized_linear_models/GLM-truncated-censored-regression.ipynb, Section: "Censored regression model"
Authors: Benjamin T. Vincent
Description: Linear regression with Censored Normal likelihood to correctly handle
    data where values exceeding bounds [-5, 5] are clipped to the boundary,
    correcting for censoring bias.

Changes from original:
- Inlined synthetic data generation with fixed seed (12345).
- Removed sampling, plotting, and comparison code.

Benchmark results:
- Original:  logp = -2117.7979, grad norm = 5815.7259, 22.5 us/call (100000 evals)
- Frozen:    logp = -2117.7979, grad norm = 5815.7259, 24.7 us/call (100000 evals)
"""

from copy import copy

import numpy as np
import pymc as pm

from numpy.random import default_rng


def build_model():
    rng = default_rng(12345)

    slope, intercept, sigma, N = 1, 0, 2, 200
    x = rng.uniform(-10, 10, N)
    y = rng.normal(loc=slope * x + intercept, scale=sigma)

    # Censor: clip y values to bounds
    bounds = [-5, 5]
    yc = copy(y)
    yc[yc <= bounds[0]] = bounds[0]
    yc[yc >= bounds[1]] = bounds[1]

    with pm.Model() as model:
        slope_rv = pm.Normal("slope", mu=0, sigma=1)
        intercept_rv = pm.Normal("intercept", mu=0, sigma=1)
        sigma_rv = pm.HalfNormal("\u03c3", sigma=1)
        y_latent = pm.Normal.dist(mu=slope_rv * x + intercept_rv, sigma=sigma_rv)
        pm.Censored("obs", y_latent, lower=bounds[0], upper=bounds[1], observed=yc)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
