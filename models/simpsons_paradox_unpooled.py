"""
Model: Simpson's Paradox - Unpooled Regression (Model 2)
Source: pymc-examples/examples/causal_inference/GLM-simpsons-paradox.ipynb, Section: "Model 2: Unpooled regression with counfounder included"
Authors: Benjamin T. Vincent
Description: Unpooled regression with group-specific intercepts and slopes, correctly
    identifying the negative within-group x-y relationship by accounting for the
    group confounding variable.

Changes from original:
- pm.Data (was already pm.Data in original)
- Removed sampling, plotting, and prediction code

Benchmark results:
- Original:  logp = -219.6522, grad norm = 148.4233, 4.4 us/call (100000 evals)
- Frozen:    logp = -219.6522, grad norm = 148.4233, 4.1 us/call (100000 evals)
"""

import numpy as np
import pymc as pm

def build_model():
    # Reproduce synthetic data generation faithfully
    rng = np.random.default_rng(1234)

    group_list = ["one", "two", "three", "four", "five"]
    trials_per_group = 20
    group_intercepts = rng.normal(0, 1, len(group_list))
    group_slopes = np.ones(len(group_list)) * -0.5
    group_mx = group_intercepts * 2
    group = np.repeat(group_list, trials_per_group)
    subject = np.concatenate(
        [np.ones(trials_per_group) * i for i in np.arange(len(group_list))]
    ).astype(int)
    intercept = np.repeat(group_intercepts, trials_per_group)
    slope = np.repeat(group_slopes, trials_per_group)
    mx = np.repeat(group_mx, trials_per_group)
    x = rng.normal(mx, 1)
    y = rng.normal(intercept + (x - mx) * slope, 1)

    coords = {"group": group_list}

    with pm.Model(coords=coords) as model:
        # Define priors
        beta0 = pm.Normal("β0", 0, sigma=5, dims="group")
        beta1 = pm.Normal("β1", 0, sigma=5, dims="group")
        sigma = pm.Gamma("sigma", 2, 2)
        # Data
        x_data = pm.Data("x", x, dims="obs_id")
        g = pm.Data("g", subject, dims="obs_id")
        # Linear model
        mu = pm.Deterministic("μ", beta0[g] + beta1[g] * x_data, dims="obs_id")
        # Define likelihood
        pm.Normal("y", mu=mu, sigma=sigma, observed=y, dims="obs_id")

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
