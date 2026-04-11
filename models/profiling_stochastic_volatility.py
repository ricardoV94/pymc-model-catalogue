"""
Model: Stochastic Volatility (profiling notebook parameterization)
Source: pymc-examples/examples/howto/profiling.ipynb, Section: "Stochastic volatility example"
Authors: PyMC developers
Description: Stochastic volatility model on S&P 500 daily returns as written in the
    profiling how-to. Uses Exponential priors on sigma and nu and a GaussianRandomWalk
    latent volatility with StudentT observation noise. Parameterization differs from
    the time_series/stochastic_volatility notebook (different priors, no init_dist,
    no dims).

Changes from original:
- Loaded S&P 500 returns from .npz instead of pm.get_data("SP500.csv")
- Added `initval=np.zeros(N)` on `s`. The default GRW init is an arithmetic
  sequence at step 100 (=sigma**-2 here), giving s[-1] ~ 3e5 and exp(-2*s)
  underflowing to 0 in the StudentT lam — producing -inf logp.
- Swapped `np.exp` for `pm.math.exp` so the latent `s` is symbolic at graph time.
- Removed profile() calls

Benchmark results:
- Original:  logp = -14525420.7782, grad norm = 58079996.0002, 62.2 us/call (100000 evals)
- Frozen:    logp = -14525420.7782, grad norm = 58079996.0002, 65.1 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data = np.load(Path(__file__).parent / "data" / "SP500.npz", allow_pickle=True)
    change = np.asarray(data["change"])

    with pm.Model() as model:
        sigma = pm.Exponential("sigma", 1.0 / 0.02, initval=0.1)
        nu = pm.Exponential("nu", 1.0 / 10)
        s = pm.GaussianRandomWalk(
            "s",
            sigma**-2,
            shape=change.shape[0],
            initval=np.zeros(change.shape[0]),
        )
        pm.StudentT("r", nu, lam=pm.math.exp(-2 * s), observed=change)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
