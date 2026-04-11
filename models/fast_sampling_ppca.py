"""
Model: Probabilistic PCA with ordered loadings
Source: pymc-examples/examples/samplers/fast_sampling_with_jax_and_numba.ipynb, Section: "Faster Sampling with JAX and Numba"
Authors: Thomas Wiecki
Description: A simple probabilistic PCA model with an Ordered transform on the loading
    matrix W to resolve rotational non-identifiability. Latent factors z are drawn from a
    standard normal and combined with W to produce the observation mean.

Changes from original:
- Simulated data generated with the original seed (rng default_rng(42)) and saved to
  models/data/fast_sampling_ppca.npz
- Removed sampling and plotting code

Benchmark results:
- Original:  logp = -30651.8907, grad norm = 0.0000, 16.1 us/call (100000 evals)
- Frozen:    logp = -30651.8907, grad norm = 0.0000, 16.7 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data = np.load(Path(__file__).parent / "data" / "fast_sampling_ppca.npz")["data"]

    N = 5000  # number of data points
    D = 2  # data dimensionality
    K = 1  # latent dimensionality

    with pm.Model() as model:
        w = pm.Normal(
            "w",
            mu=0,
            sigma=2,
            shape=[D, K],
            transform=pm.distributions.transforms.Ordered(),
        )
        z = pm.Normal("z", mu=0, sigma=1, shape=[N, K])
        x = pm.Normal("x", mu=w.dot(z.T), sigma=1, shape=[D, N], observed=data)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
