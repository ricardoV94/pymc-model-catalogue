"""
Model: Linear regression with pm.Data containers
Source: pymc-examples/examples/fundamentals/data_container.ipynb, Section: "Using Data Containers for readability and reproducibility"
Authors: Juan Martin Loyola, Kavya Jaiswal, Oriol Abril, Jesse Grabowski
Description: Simple linear regression y ~ Normal(beta * x, sigma) demonstrating
    pm.Data containers wrapping both exogenous x and observed y on n=100 synthetic points.

Changes from original:
- Inlined small synthetic data (n=100, seed from "Data Containers in PyMC")
- Removed sampling and plotting code

Benchmark results:
- Original:  logp = -1876.6934, grad norm = 3477.7589, 2.2 us/call (100000 evals)
- Frozen:    logp = -1876.6934, grad norm = 3477.7589, 1.9 us/call (100000 evals)
"""

import numpy as np
import pymc as pm


def build_model():
    RANDOM_SEED = sum(map(ord, "Data Containers in PyMC"))
    rng = np.random.default_rng(RANDOM_SEED)

    true_beta = 3
    true_std = 5
    n_obs = 100
    x = rng.normal(size=n_obs)
    y = rng.normal(loc=true_beta * x, scale=true_std, size=n_obs)

    with pm.Model() as model:
        x_data = pm.Data("x_data", x)
        y_data = pm.Data("y_data", y)
        beta = pm.Normal("beta")
        mu = pm.Deterministic("mu", beta * x_data)
        sigma = pm.Exponential("sigma", 1)
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=y_data)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
