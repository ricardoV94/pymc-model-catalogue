"""
Model: Sharp Regression Discontinuity Design
Source: pymc-examples/examples/causal_inference/regression_discontinuity.ipynb, Section: "Sharp regression discontinuity model"
Authors: Benjamin T. Vincent
Description: Bayesian regression discontinuity model estimating a treatment effect at a
    threshold, with a Cauchy prior on the discontinuity magnitude.

Changes from original:
- pm.MutableData -> pm.Data (API update)
- Removed sampling, plotting, and counterfactual inference code

Benchmark results:
- Original:  logp = -1082.6350, grad norm = 753.0397, 4.8 us/call (100000 evals)
- Frozen:    logp = -1082.6350, grad norm = 753.0397, 4.7 us/call (100000 evals)
"""

import numpy as np
import pymc as pm

def build_model():
    # Reproduce synthetic data generation faithfully
    RANDOM_SEED = 123
    rng = np.random.default_rng(RANDOM_SEED)

    # define true parameters
    threshold = 0.0
    treatment_effect = 0.7
    N = 1000
    sd = 0.3  # represents change between pre and post test with zero measurement error

    # No measurement error, but random change from pre to post
    x = rng.normal(size=N)
    treated = (x < threshold).astype(float)
    y = x + rng.normal(loc=0, scale=sd, size=N) + treatment_effect * treated

    with pm.Model() as model:
        x_data = pm.Data("x", x, dims="obs_id")
        treated_data = pm.Data("treated", treated, dims="obs_id")
        sigma = pm.HalfNormal("sigma", 1)
        delta = pm.Cauchy("effect", alpha=0, beta=1)
        mu = pm.Deterministic("mu", x_data + (delta * treated_data), dims="obs_id")
        pm.Normal("y", mu=mu, sigma=sigma, observed=y, dims="obs_id")

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
