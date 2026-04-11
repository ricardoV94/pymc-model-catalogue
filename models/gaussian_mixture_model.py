"""
Model: Gaussian Mixture Model (3 components, NormalMixture)
Source: pymc-examples/examples/mixture_models/gaussian_mixture_model.ipynb
Authors: Abe Flaxman, Thomas Wiecki, Benjamin T. Vincent
Description: A 3-component Gaussian mixture model using pm.NormalMixture with ordered
    means, fitted to synthetic data drawn from known Gaussian components.

Changes from original:
- Saved synthetic data to .npz file instead of generating inline
- Removed sampling, plotting, and posterior analysis code

Benchmark results:
- Original:  logp = -1338.7665, grad norm = 680.8634, 47.0 us/call (100000 evals)
- Frozen:    logp = -1338.7665, grad norm = 680.8634, 48.8 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm

def build_model():
    data = np.load(Path(__file__).parent / "data" / "gaussian_mixture_data.npz")
    x = data["x"]

    k = 3

    with pm.Model(coords={"cluster": range(k)}) as model:
        μ = pm.Normal(
            "μ",
            mu=0,
            sigma=5,
            transform=pm.distributions.transforms.ordered,
            initval=[-4, 0, 4],
            dims="cluster",
        )
        σ = pm.HalfNormal("σ", sigma=1, dims="cluster")
        weights = pm.Dirichlet("w", np.ones(k), dims="cluster")
        pm.NormalMixture("x", w=weights, mu=μ, sigma=σ, observed=x)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
