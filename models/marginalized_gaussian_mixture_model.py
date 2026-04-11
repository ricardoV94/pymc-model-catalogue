"""
Model: Marginalized Gaussian Mixture Model (3 components)
Source: pymc-examples/examples/mixture_models/marginalized_gaussian_mixture_model.ipynb
Authors: Austin Rochford, Marco Gorelli, Chris Fonnesbeck, Benjamin T. Vincent, Abhipsha Das
Description: A 3-component marginalized Gaussian mixture model using pm.NormalMixture
    with ordered means and Gamma-distributed precisions, fitted to synthetic data.

Changes from original:
- Updated from PyMC3 API to PyMC v5 API (pm.transforms.ordered -> pm.distributions.transforms.ordered,
  testval -> initval)
- Saved synthetic data to .npz file instead of generating inline
- Removed sampling, plotting, and posterior predictive code

Benchmark results:
- Original:  logp = -2399.7335, grad norm = 651.8244, 93.7 us/call (100000 evals)
- Frozen:    logp = -2399.7335, grad norm = 651.8244, 96.0 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm

def build_model():
    data = np.load(Path(__file__).parent / "data" / "marginalized_gmm_data.npz")
    x = data["x"]

    N = 1000
    W = np.array([0.35, 0.4, 0.25])

    with pm.Model(coords={"cluster": np.arange(len(W)), "obs_id": np.arange(N)}) as model:
        w = pm.Dirichlet("w", np.ones_like(W))

        mu = pm.Normal(
            "mu",
            np.zeros_like(W),
            1.0,
            dims="cluster",
            transform=pm.distributions.transforms.ordered,
            initval=[1, 2, 3],
        )
        tau = pm.Gamma("tau", 1.0, 1.0, dims="cluster")

        x_obs = pm.NormalMixture("x_obs", w, mu, tau=tau, observed=x, dims="obs_id")

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
