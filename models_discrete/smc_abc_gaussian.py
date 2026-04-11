"""
Model: SMC-ABC Gaussian Fit via Simulator
Source: pymc-examples/examples/samplers/SMC-ABC_Lotka-Volterra_example.ipynb, Section: "Old good Gaussian fit"
Authors: Osvaldo Martin
Description: A simple ABC example estimating the mean and standard deviation of Gaussian data
    using a pm.Simulator likelihood with a pseudo random number generator as the synthetic
    data generator, sum_stat="sort", and a gaussian kernel distance.

Has discrete variables: No, but uses pm.Simulator which has no defined gradient and
    a *random* logp (synthetic data is redrawn on every call), so it is benchmarked
    via compile_logp (no dlogp) alongside the discrete models.

Changes from original:
- Saved the 1000-point synthetic observed dataset to .npz (generated with
  np.random.default_rng(42).normal(loc=0, scale=1, size=1000))
- Removed sampling and plotting code
- Inlined the normal_sim simulator inside build_model()

Benchmark results:
- Original:  logp = -6.1559, 72.1 us/call (100000 evals)
- Frozen:    logp = -7.9608, 70.9 us/call (100000 evals)
  (logp differs between runs — pm.Simulator is stochastic)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data = np.load(Path(__file__).parent / "data" / "smc_abc_gaussian.npz")["data"]

    def normal_sim(rng, a, b, size=1000):
        return rng.normal(a, b, size=size)

    with pm.Model() as model:
        a = pm.Normal("a", mu=0, sigma=5)
        b = pm.HalfNormal("b", sigma=1)
        s = pm.Simulator("s", normal_sim, params=(a, b), sum_stat="sort", epsilon=1, observed=data)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model, discrete=True)
