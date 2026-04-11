"""
Model: Unimputed Censored Normal Model
Source: pymc-examples/examples/survival_analysis/censored_data.ipynb, Section: "Model 2 - Unimputed Censored Model of Censored Data"
Authors: Luis Mario Domenzain, George Ho, Benjamin Vincent, Osvaldo Martin
Description: Censored normal model using pm.Censored to integrate out censored values.
    Uses simulated data with known mean=13 and sigma=5, censored at low=3 and high=16.
    More efficient than the imputed version for large amounts of censored data.

Changes from original:
- Inlined generated and censored data as numpy arrays (500 observations, seeded with rng=1234).
- Removed sampling and plotting code.

Benchmark results:
- Original:  logp = -2230.4525, grad norm = 1747.7002, 59.2 us/call (100000 evals)
- Frozen:    logp = -2230.4525, grad norm = 1747.7002, 58.7 us/call (100000 evals)
"""

import numpy as np
import pymc as pm

def build_model():
    # Reproduce the data generation
    from numpy.random import default_rng

    rng = default_rng(1234)
    size = 500
    true_mu = 13.0
    true_sigma = 5.0
    samples = rng.normal(true_mu, true_sigma, size)

    low = 3.0
    high = 16.0

    # Censor samples
    censored = samples.copy()
    censored[censored <= low] = low
    censored[censored >= high] = high

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=(high - low) / 2.0)
        sigma = pm.HalfNormal("sigma", sigma=(high - low) / 2.0)
        y_latent = pm.Normal.dist(mu=mu, sigma=sigma)
        obs = pm.Censored("obs", y_latent, lower=low, upper=high, observed=censored)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
