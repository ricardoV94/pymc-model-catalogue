"""
Model: Dirichlet Process Mixture for Old Faithful Density Estimation
Source: pymc-examples/examples/mixture_models/dp_mix.ipynb,
    Section: "Dirichlet process mixtures" (Old Faithful)
Authors: Austin Rochford, Abhipsha Das
Description: A truncated Dirichlet process mixture of normals (K=30 components) for
    density estimation of standardized Old Faithful waiting times. Uses a
    stick-breaking construction with Beta-distributed breaking proportions,
    Gamma priors on precision components, and a NormalMixture likelihood.

Changes from original:
- Saved standardized Old Faithful waiting time data to .npz file instead of
  loading via CSV
- Removed sampling, plotting, and density estimation code

Benchmark results:
- Original:  logp = -1350.6972, grad norm = 994.5440, 103.5 us/call (100000 evals)
- Frozen:    logp = -1350.6972, grad norm = 994.5440, 100.2 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm
import pytensor.tensor as pt

def build_model():
    data = np.load(Path(__file__).parent / "data" / "old_faithful.npz")
    std_waiting = data["std_waiting"]

    N = len(std_waiting)
    K = 30

    def stick_breaking(beta):
        portion_remaining = pt.concatenate([[1], pt.extra_ops.cumprod(1 - beta)[:-1]])
        return beta * portion_remaining

    with pm.Model(coords={"component": np.arange(K), "obs_id": np.arange(N)}) as model:
        alpha = pm.Gamma("alpha", 1.0, 1.0)
        beta = pm.Beta("beta", 1.0, alpha, dims="component")
        w = pm.Deterministic("w", stick_breaking(beta), dims="component")

        tau = pm.Gamma("tau", 1.0, 1.0, dims="component")
        lambda_ = pm.Gamma("lambda_", 10.0, 1.0, dims="component")
        mu = pm.Normal("mu", 0, tau=lambda_ * tau, dims="component")
        obs = pm.NormalMixture(
            "obs", w, mu, tau=lambda_ * tau, observed=std_waiting, dims="obs_id"
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
