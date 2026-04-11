"""
Model: Euler-Maruyama linear SDE
Source: pymc-examples/examples/time_series/Euler-Maruyama_and_SDEs.ipynb, Section: "Example Model"
Authors: @maedoc (Jul 2016), @fonnesbeck (updated to PyMC v5, Sep 2024)
Description: Infers parameters of a scalar linear SDE
    dX_t = lam * X_t + sigma^2 dW_t
    discretized with the Euler-Maruyama scheme. Latent path `xh` is modeled with
    `pm.EulerMaruyama` using a user-supplied drift/diffusion function, and noisy
    observations `zh` are Normal.

Changes from original:
- Inlined synthetic data generation with the original RANDOM_SEED
- Removed sampling and plotting code

Benchmark results:
- Original:  logp = 1356.8946, grad norm = 2885.8380, 3.9 us/call (100000 evals)
- Frozen:    logp = 1356.8946, grad norm = 2885.8380, 3.8 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    # Reproduce the notebook's synthetic data
    RANDOM_SEED = 8927
    rng = np.random.RandomState(RANDOM_SEED)

    # parameters
    lam = -0.78
    s2 = 5e-3
    N = 200
    dt = 1e-1

    # time series
    x = 0.1
    x_t = []
    for i in range(N):
        x += dt * lam * x + np.sqrt(dt) * s2 * rng.randn()
        x_t.append(x)
    x_t = np.array(x_t)

    # z_t noisy observation
    z_t = x_t + rng.randn(x_t.size) * 5e-3

    def lin_sde(x, lam, s2):
        return lam * x, s2

    with pm.Model() as model:
        # uniform prior, but we know it must be negative
        l = pm.HalfCauchy("l", beta=1)
        s = pm.Uniform("s", 0.005, 0.5)

        # "hidden states" following a linear SDE distribution
        # parametrized by time step (det. variable) and lam (random variable)
        xh = pm.EulerMaruyama(
            "xh",
            dt=dt,
            sde_fn=lin_sde,
            sde_pars=(-l, s**2),
            shape=N,
            initval=x_t,
        )

        # predicted observation
        zh = pm.Normal("zh", mu=xh, sigma=5e-3, observed=z_t)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
