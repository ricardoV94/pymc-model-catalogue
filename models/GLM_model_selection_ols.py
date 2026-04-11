"""
Model: GLM Model Selection - Simple OLS Linear Regression
Source: pymc-examples/examples/generalized_linear_models/GLM-model-selection.ipynb, Section: "Demonstrate simple linear model"
Authors: Jon Sedar, Junpeng Lao, Abhipsha Das, Oriol Abril-Pla
Description: Simple linear regression with Normal priors (Ridge) and HalfCauchy noise
    prior on synthetic data, used as a baseline for polynomial model selection.

Changes from original:
- Inlined data generation logic with fixed seed (RANDOM_SEED=8927).
- Standardized x values inline.
- Removed sampling, plotting, bambi models, and model comparison code.

Benchmark results:
- Original:  logp = -8625.0966, grad norm = 17002.6635, 2.9 us/call (100000 evals)
- Frozen:    logp = -8625.0966, grad norm = 17002.6635, 2.6 us/call (100000 evals)
"""

import numpy as np
import pymc as pm


def build_model():
    rng = np.random.default_rng(8927)

    # Generate linear data: y ~ a + bx + e
    n = 30
    a, b, c = -30, 5, 0
    latent_sigma_y = 40
    x_raw = rng.choice(np.arange(100), n, replace=False).astype(float)
    y = a + b * x_raw + c * x_raw**2
    latent_error = rng.normal(0, latent_sigma_y, n)
    y = y + latent_error

    # Standardize x
    x = (x_raw - x_raw.mean()) / x_raw.std()

    with pm.Model() as model:
        ## define Normal priors to give Ridge regression
        b0 = pm.Normal("Intercept", mu=0, sigma=100)
        b1 = pm.Normal("x", mu=0, sigma=100)

        ## define Linear model
        yest = b0 + b1 * x

        ## define Normal likelihood with HalfCauchy noise (fat tails, equiv to HalfT 1DoF)
        y_sigma = pm.HalfCauchy("y_sigma", beta=10)
        pm.Normal("likelihood", mu=yest, sigma=y_sigma, observed=y)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
