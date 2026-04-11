"""
Model: GLM Robust Regression - Student-T Likelihood
Source: pymc-examples/examples/generalized_linear_models/GLM-robust.ipynb, Section: "Robust Regression"
Authors: Thomas Wiecki, Chris Fonnesbeck, Abhipsha Das, Conor Hassan, Igor Kuvychko, Reshama Shaikh, Oriol Abril Pla
Description: Robust linear regression using Student-T likelihood to handle outliers.
    The heavier tails of the Student-T distribution reduce the influence of outliers on the fit.

Changes from original:
- Inlined generated data as numpy arrays.
- Removed sampling and plotting code.

Benchmark results:
- Original:  logp = -347.6056, grad norm = 94.6107, 4.1 us/call (100000 evals)
- Frozen:    logp = -347.6056, grad norm = 94.6107, 4.0 us/call (100000 evals)
"""

import numpy as np
import pymc as pm

def build_model():
    RANDOM_SEED = 8927
    rng = np.random.default_rng(RANDOM_SEED)
    size = 100
    true_intercept = 1
    true_slope = 2
    x = np.linspace(0, 1, size)
    true_regression_line = true_intercept + true_slope * x
    y = true_regression_line + rng.normal(scale=0.5, size=size)
    # Add outliers
    x_out = np.append(x, [0.1, 0.15, 0.2])
    y_out = np.append(y, [8, 6, 9])

    with pm.Model() as robust_model:
        xdata = pm.Data("x", x_out, dims="obs_id")

        # define priors
        intercept = pm.Normal("intercept", mu=0, sigma=1)
        slope = pm.Normal("slope", mu=0, sigma=1)
        sigma = pm.HalfCauchy("sigma", beta=10)

        mu = pm.Deterministic("mu", intercept + slope * xdata, dims="obs_id")

        # define likelihood
        likelihood = pm.StudentT("y", mu=mu, sigma=sigma, nu=3, observed=y_out, dims="obs_id")

    ip = robust_model.initial_point()
    robust_model.rvs_to_initial_values = {rv: None for rv in robust_model.free_RVs}
    return robust_model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
