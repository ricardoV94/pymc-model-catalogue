"""
Model: GP smoothing with GaussianRandomWalk
Source: pymc-examples/examples/gaussian_processes/GP-smoothing.ipynb, Section: "Gaussian Process (GP) smoothing"
Authors: Andrey Kuzmenko, Juan Orduz
Description: Smoothing model using a GaussianRandomWalk prior for the latent
    function values, with a shared smoothing parameter controlling the tradeoff
    between data fidelity and smoothness.

Changes from original:
- Removed sampling, plotting, smoothing parameter variation
- Inlined synthetic data generation with fixed seed
- Replaced pytensor.shared smoothing_param with a constant 0.9
- Added ip capture and initval clearing

Benchmark results:
- Original:  logp = -133160411.7587, grad norm = 133272440.0926, 5.6 us/call (100000 evals)
- Frozen:    logp = -133160411.7587, grad norm = 133272440.0926, 5.7 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    rng = np.random.default_rng(8927)

    x = np.linspace(0, 50, 100)
    y = np.exp(1.0 + np.power(x, 0.5) - np.exp(x / 15.0)) + rng.normal(
        scale=1.0, size=x.shape
    )

    LARGE_NUMBER = 1e5
    smoothing_param = 0.9

    with pm.Model() as model:
        mu = pm.Normal("mu", sigma=LARGE_NUMBER)
        tau = pm.Exponential("tau", 1.0 / LARGE_NUMBER)
        z = pm.GaussianRandomWalk(
            "z",
            mu=mu,
            sigma=pm.math.sqrt((1.0 - smoothing_param) / tau),
            shape=y.shape,
        )
        obs = pm.Normal("obs", mu=z, tau=tau / smoothing_param, observed=y)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
