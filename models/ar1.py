"""
Model: AR(1) Model (misspecified)
Source: pymc-examples/examples/time_series/AR.ipynb, Section: "Analysis of An AR(1) Model in PyMC"
Authors: Ed Herbst, Chris Fonnesbeck
Description: An AR(1) model with intercept (constant=True) fitted to data actually generated
    from an AR(2) process, demonstrating robustness of the misspecified model.

Changes from original:
- Saved synthetic AR(2) data to .npz file instead of generating inline
- Removed sampling and plotting code

Benchmark results:
- Original:  logp = -47994.2261, grad norm = 77133.8184, 7.4 us/call (100000 evals)
- Frozen:    logp = -47994.2261, grad norm = 77133.8184, 7.9 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm

def build_model():
    data = np.load(Path(__file__).parent / "data" / "AR_data.npz")
    y = data["y"]

    with pm.Model() as model:
        # assumes 95% of prob mass is between -2 and 2
        rho = pm.Normal("rho", mu=0.0, sigma=1.0, shape=2)
        # precision of the innovation term
        tau = pm.Exponential("tau", lam=0.5)

        likelihood = pm.AR(
            "y", rho=rho, tau=tau, constant=True, init_dist=pm.Normal.dist(0, 10), observed=y
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip

if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
