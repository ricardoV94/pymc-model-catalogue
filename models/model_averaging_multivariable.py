"""
Model: Linear Regression on Wrist, Height, Weight (Body Fat)
Source: pymc-examples/examples/diagnostics_and_criticism/model_averaging.ipynb, Section: "Weighted posterior predictive samples"
Authors: Osvaldo Martin
Description: Multiple linear regression predicting body fat percentage (siri) from wrist
    circumference, height, and weight, used as one of two models for demonstrating
    Bayesian model averaging.

Changes from original:
- Loaded data from .npz file instead of CSV
- Used numpy array operations instead of pandas DataFrame indexing for the design matrix
- Removed sampling, model comparison, and plotting code

Benchmark results:
- Original:  logp = -2814.2576, grad norm = 38217.2185, 5.6 us/call (100000 evals)
- Frozen:    logp = -2814.2576, grad norm = 38217.2185, 5.8 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm

def build_model():
    data = np.load(Path(__file__).parent / "data" / "body_fat.npz")
    X = np.column_stack([data["wrist"], data["height"], data["weight"]])
    siri_obs = data["siri"]

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta = pm.Normal("beta", mu=0, sigma=1, shape=3)
        sigma = pm.HalfNormal("sigma", 5)

        mu = alpha + pm.math.dot(beta, X.T)

        siri = pm.Normal("siri", mu=mu, sigma=sigma, observed=siri_obs)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
