"""
Model: AR(2) Model
Source: pymc-examples/examples/time_series/AR.ipynb, Section: "Extension to AR(p)"
Authors: Ed Herbst, Chris Fonnesbeck
Description: An AR(2) model with intercept (constant=True) using a single shape-3 Normal
    prior for rho and HalfNormal prior for sigma, fitted to synthetic AR(2) data.

Changes from original:
- Saved synthetic AR(2) data to .npz file instead of generating inline
- Removed sampling and plotting code

Benchmark results:
- Original:  logp = -12598.5776, grad norm = 4010.2439, 8.9 us/call (100000 evals)
- Frozen:    logp = -12598.5776, grad norm = 4010.2439, 9.1 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm

def build_model():
    data = np.load(Path(__file__).parent / "data" / "AR_data.npz")
    y = data["y"]

    with pm.Model() as model:
        rho = pm.Normal("rho", 0.0, 1.0, shape=3)
        sigma = pm.HalfNormal("sigma", 3)
        likelihood = pm.AR(
            "y", rho=rho, sigma=sigma, constant=True, init_dist=pm.Normal.dist(0, 10), observed=y
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip

if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
