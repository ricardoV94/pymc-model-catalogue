"""
Model: AR(2) Model with Individual Priors
Source: pymc-examples/examples/time_series/AR.ipynb, Section: "Extension to AR(p)" (second variant)
Authors: Ed Herbst, Chris Fonnesbeck
Description: An AR(2) model with intercept where each AR coefficient has a distinct prior
    (Normal for intercept, Uniform for AR lags), using pt.stack to combine them.

Changes from original:
- Saved synthetic AR(2) data to .npz file instead of generating inline
- Removed sampling and plotting code

Benchmark results:
- Original:  logp = -12601.1218, grad norm = 2162.6238, 10.1 us/call (100000 evals)
- Frozen:    logp = -12601.1218, grad norm = 2162.6238, 9.2 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm
import pytensor.tensor as pt

def build_model():
    data = np.load(Path(__file__).parent / "data" / "AR_data.npz")
    y = data["y"]

    with pm.Model() as model:
        rho0 = pm.Normal("rho0", mu=0.0, sigma=5.0)
        rho1 = pm.Uniform("rho1", -1, 1)
        rho2 = pm.Uniform("rho2", -1, 1)
        sigma = pm.HalfNormal("sigma", 3)
        likelihood = pm.AR(
            "y",
            rho=pt.stack([rho0, rho1, rho2]),
            sigma=sigma,
            constant=True,
            init_dist=pm.Normal.dist(0, 10),
            observed=y,
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip

if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
