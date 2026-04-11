"""
Model: Eight Schools (Centered Parameterization)
Source: pymc-examples/examples/diagnostics_and_criticism/Diagnosing_biased_Inference_with_Divergences.ipynb, Section: "The Eight Schools Model"
Authors: Agustina Arroyuelo
Description: The classic Eight Schools hierarchical model in centered parameterization,
    known to exhibit a pathological funnel geometry that causes divergences in HMC.

Changes from original:
- Inlined the small data arrays (8 values each)
- Removed sampling, diagnostics, and plotting code

Benchmark results:
- Original:  logp = -55.3556, grad norm = 8.0037, 3.0 us/call (100000 evals)
- Frozen:    logp = -55.3556, grad norm = 8.0037, 3.1 us/call (100000 evals)
"""

import numpy as np
import pymc as pm

def build_model():
    J = 8
    y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=5)
        tau = pm.HalfCauchy("tau", beta=5)
        theta = pm.Normal("theta", mu=mu, sigma=tau, shape=J)
        obs = pm.Normal("obs", mu=theta, sigma=sigma, observed=y)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip

if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
