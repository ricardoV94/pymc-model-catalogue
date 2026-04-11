"""
Model: Eight Schools (Non-Centered Parameterization)
Source: pymc-examples/examples/diagnostics_and_criticism/Diagnosing_biased_Inference_with_Divergences.ipynb, Section: "A Non-Centered Eight Schools Implementation"
Authors: Agustina Arroyuelo
Description: The classic Eight Schools hierarchical model in non-centered parameterization,
    which avoids the funnel pathology of the centered version by reparameterizing
    group-level parameters via a latent standard normal and a deterministic transform.

Changes from original:
- Inlined the small data arrays (8 values each)
- Removed sampling, diagnostics, and plotting code

Benchmark results:
- Original:  logp = -42.4801, grad norm = 1.3034, 2.8 us/call (100000 evals)
- Frozen:    logp = -42.4801, grad norm = 1.3034, 2.8 us/call (100000 evals)
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
        theta_tilde = pm.Normal("theta_t", mu=0, sigma=1, shape=J)
        theta = pm.Deterministic("theta", mu + tau * theta_tilde)
        obs = pm.Normal("obs", mu=theta, sigma=sigma, observed=y)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip

if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
