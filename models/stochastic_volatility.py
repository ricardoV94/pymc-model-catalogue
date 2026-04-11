"""
Model: Stochastic Volatility Model
Source: pymc-examples/examples/time_series/stochastic_volatility.ipynb, Section: "Build Model"
Authors: John Salvatier, Colin Carroll, Abhipsha Das
Description: A stochastic volatility model for S&P 500 daily log returns using a
    GaussianRandomWalk latent volatility process with StudentT observation noise.

Changes from original:
- Saved S&P 500 data to .npz file instead of loading via CSV/pm.get_data
- Removed sampling, prior/posterior predictive, and plotting code

Benchmark results:
- Original:  logp = 1268.3194, grad norm = 2905.3995, 104.7 us/call (100000 evals)
- Frozen:    logp = 1268.3194, grad norm = 2905.3995, 99.6 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm

def build_model():
    data = np.load(Path(__file__).parent / "data" / "SP500.npz", allow_pickle=True)
    change = data["change"]
    dates = data["dates"]

    with pm.Model(coords={"time": dates}) as model:
        step_size = pm.Exponential("step_size", 10)
        volatility = pm.GaussianRandomWalk(
            "volatility", sigma=step_size, dims="time", init_dist=pm.Normal.dist(0, 100)
        )
        nu = pm.Exponential("nu", 0.1)
        returns = pm.StudentT(
            "returns", nu=nu, lam=np.exp(-2 * volatility), observed=change, dims="time"
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip

if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
