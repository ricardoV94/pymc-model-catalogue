"""
Model: Bayesian Moderation Analysis
Source: pymc-examples/examples/causal_inference/moderation_analysis.ipynb, Section: "Define the PyMC model and conduct inference"
Authors: Benjamin T. Vincent
Description: Moderation model examining whether age moderates the relationship between
    training hours and muscle percentage, using an interaction term.

Changes from original:
- pm.ConstantData -> pm.Data (API update)
- Data loaded from .npz instead of CSV
- Removed sampling, plotting, and interpretation code

Benchmark results:
- Original:  logp = -178185.8329, grad norm = 1426622.5471, 5.2 us/call (100000 evals)
- Frozen:    logp = -178185.8329, grad norm = 1426622.5471, 5.1 us/call (100000 evals)
"""

import numpy as np
import pymc as pm

from pathlib import Path

def build_model():
    # Load data
    data = np.load(Path(__file__).parent / "data" / "moderation_muscle.npz")
    x = data["x"]  # training hours
    m = data["m"]  # age (moderator)
    y = data["y"]  # muscle percentage

    with pm.Model() as model:
        x_data = pm.Data("x", x)
        m_data = pm.Data("m", m)
        # priors
        beta0 = pm.Normal("β0", mu=0, sigma=10)
        beta1 = pm.Normal("β1", mu=0, sigma=10)
        beta2 = pm.Normal("β2", mu=0, sigma=10)
        beta3 = pm.Normal("β3", mu=0, sigma=10)
        sigma = pm.HalfCauchy("σ", 1)
        # likelihood
        y_obs = pm.Normal(
            "y",
            mu=beta0 + (beta1 * x_data) + (beta2 * x_data * m_data) + (beta3 * m_data),
            sigma=sigma,
            observed=y,
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
