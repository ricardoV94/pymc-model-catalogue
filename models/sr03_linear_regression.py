"""
Model: Linear Regression of Weight on Height (Howell Adults)
Source: pymc-examples/examples/statistical_rethinking_lectures/03-Geocentric_Models.ipynb, Section: "(5) Analyse real data"
Authors: Dustin Stansbury
Description: Linear regression of adult weight on height using the Howell dataset.
    Priors encode knowledge that weight increases with height and variance is bounded.

Changes from original:
- Loaded and filtered data inline instead of using utils.load_data
- Removed sampling, plotting, and posterior predictive code

Benchmark results:
- Original:  logp = -8377.4057, grad norm = 19019.1227, 3.8 us/call (100000 evals)
- Frozen:    logp = -8377.4057, grad norm = 19019.1227, 3.8 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data = np.load(Path(__file__).parent / "data" / "sr_howell.npz")
    height = data["height"]
    weight = data["weight"]
    age = data["age"]

    # Filter to adults
    adults = age >= 18
    H = height[adults]
    W = weight[adults]

    with pm.Model() as model:
        H_ = pm.Data("H", H, dims="obs_ids")

        # Priors
        alpha = pm.Normal("alpha", 0, 10)
        beta = pm.Uniform("beta", 0, 1)
        sigma = pm.Uniform("sigma", 0, 10)

        # Likelihood
        mu = alpha + beta * H_
        pm.Normal("W_obs", mu, sigma, observed=W, dims="obs_ids")

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
