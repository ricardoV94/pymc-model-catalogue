"""
Model: Dirichlet-Multinomial Hierarchical Model
Source: pymc-examples/examples/samplers/sampling_conjugate_step.ipynb, Section: "Comparing partial conjugate with full NUTS sampling"
Authors: Christopher Krapu
Description: Hierarchical Dirichlet-Multinomial model with N=500 observations of J=10
    outcome counts. A scalar Exponential concentration tau parameterises a Dirichlet
    prior over per-observation probability vectors p, which are then used in a
    Multinomial likelihood. Originally used to compare full NUTS sampling against a
    custom locally-conjugate Gibbs step for p.

Changes from original:
- Counts data (500 x 10) saved to data/sampling_conjugate_step_counts.npz
  (generated from the notebook's simulation with seed 8927)
- Removed sampling and plotting code
- Only the "Full NUTS" variant is extracted (the partial-conjugate variant omits
  the Multinomial likelihood, which would leave no observed data in the model)

Benchmark results:
- Original:  logp = -15144.5906, grad norm = 2645.7235, 181.3 us/call (77210 evals)
- Frozen:    logp = -15144.5906, grad norm = 2645.7235, 187.4 us/call (69890 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data = np.load(
        Path(__file__).parent / "data" / "sampling_conjugate_step_counts.npz"
    )
    counts = data["counts"]
    N, J = counts.shape
    ncounts = 20

    with pm.Model() as model:
        tau = pm.Exponential("tau", lam=1, initval=1.0)
        alpha = pm.Deterministic("alpha", tau * np.ones([N, J]))
        p = pm.Dirichlet("p", a=alpha)
        x = pm.Multinomial("x", n=ncounts, p=p, observed=counts)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
