"""
Model: Linear Regression on Abdomen (Body Fat)
Source: pymc-examples/examples/diagnostics_and_criticism/model_averaging.ipynb, Section: "Weighted posterior predictive samples"
Authors: Osvaldo Martin
Description: Simple linear regression predicting body fat percentage (siri) from abdomen
    circumference, used as one of two models for demonstrating Bayesian model averaging.

Changes from original:
- Loaded data from .npz file instead of CSV
- Removed sampling, model comparison, and plotting code

Benchmark results:
- Original:  logp = -2812.4198, grad norm = 18847.6869, 3.7 us/call (100000 evals)
- Frozen:    logp = -2812.4198, grad norm = 18847.6869, 3.8 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm

def build_model():
    data = np.load(Path(__file__).parent / "data" / "body_fat.npz")
    abdomen = data["abdomen"]
    siri_obs = data["siri"]

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta = pm.Normal("beta", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", 5)

        mu = alpha + beta * abdomen

        siri = pm.Normal("siri", mu=mu, sigma=sigma, observed=siri_obs)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
