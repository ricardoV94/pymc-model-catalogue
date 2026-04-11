"""
Model: Cylinder Body Model (Weight from Height)
Source: pymc-examples/examples/statistical_rethinking_lectures/19-Generalized_Linear_Madness.ipynb, Section: "Revisiting Modeling Human Height"
Authors: Dustin Stansbury
Description: Mechanistic model treating the human body as a cylinder, where weight is
    proportional to height cubed via volume. Uses LogNormal likelihood with parameters
    for body proportionality (p) and density scaling (k).

Changes from original:
- Loaded data inline from npz instead of using utils.load_data
- Computed scaled height/weight inline
- Removed sampling and plotting code

Benchmark results:
- Original:  logp = -467.4999, grad norm = 545.6755, 7.9 us/call (100000 evals)
- Frozen:    logp = -467.4999, grad norm = 545.6755, 7.8 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data = np.load(Path(__file__).parent / "data" / "sr_howell.npz")
    height = data["height"]
    weight = data["weight"]

    # Scale to proportions of mean
    height_scaled = height / height.mean()
    weight_scaled = weight / weight.mean()

    PI = 3.141593

    with pm.Model() as model:
        H = pm.Data("H", height_scaled, dims="obs_id")
        W = weight_scaled

        sigma = pm.Exponential("sigma", 1)
        p = pm.Beta("p", 25, 50)
        k = pm.Exponential("k", 0.5)

        mu = pm.math.log(PI * k * p**2 * H**3)
        pm.LogNormal("W", mu, sigma, observed=W, dims="obs_id")

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
