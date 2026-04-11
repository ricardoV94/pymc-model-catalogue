"""
Model: Dirichlet Process Mixture for Sunspot Density Estimation
Source: pymc-examples/examples/mixture_models/dp_mix.ipynb,
    Section: "Dirichlet process mixtures" (Sunspot)
Authors: Austin Rochford, Abhipsha Das
Description: A truncated Dirichlet process mixture of Poissons (K=50 components) for
    density estimation of yearly sunspot counts. Uses a stick-breaking construction
    with Beta-distributed breaking proportions, Gamma priors on Poisson rates,
    and a Mixture likelihood with Poisson components.

Changes from original:
- Saved sunspot year data to .npz file instead of loading via CSV
- Removed sampling, plotting, and density estimation code

Benchmark results:
- Original:  logp = -15252.7573, grad norm = 13071.8831, 169.5 us/call (78671 evals)
- Frozen:    logp = -15252.7573, grad norm = 13071.8831, 183.9 us/call (79705 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm
import pytensor.tensor as pt

def build_model():
    data = np.load(Path(__file__).parent / "data" / "sunspot.npz")
    sunspot_year = data["sunspot_year"]

    K = 50
    N = len(sunspot_year)

    def stick_breaking(beta):
        portion_remaining = pt.concatenate([[1], pt.extra_ops.cumprod(1 - beta)[:-1]])
        return beta * portion_remaining

    with pm.Model(coords={"component": np.arange(K), "obs_id": np.arange(N)}) as model:
        alpha = pm.Gamma("alpha", 1.0, 1.0)
        beta = pm.Beta("beta", 1, alpha, dims="component")
        w = pm.Deterministic("w", stick_breaking(beta), dims="component")
        # Gamma is conjugate prior to Poisson
        lambda_ = pm.Gamma("lambda_", 300.0, 2.0, dims="component")
        obs = pm.Mixture(
            "obs", w, pm.Poisson.dist(lambda_), observed=sunspot_year, dims="obs_id"
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
