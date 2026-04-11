"""
Model: Imputed Censored Normal Model
Source: pymc-examples/examples/survival_analysis/censored_data.ipynb, Section: "Model 1 - Imputed Censored Model of Censored Data"
Authors: Luis Mario Domenzain, George Ho, Benjamin Vincent, Osvaldo Martin
Description: Censored normal model where left- and right-censored values are imputed as
    constrained parameters. Censored observations are modeled as Normal draws with
    interval transforms enforcing the censoring bounds. Uses simulated data with
    known mean=13 and sigma=5, censored at low=3 and high=16.

Changes from original:
- Inlined generated and censored data as numpy arrays (500 observations, seeded with rng=1234).
- Removed sampling and plotting code.

Benchmark results:
- Original:  logp = -1556.9045, grad norm = 189.2441, 4.8 us/call (100000 evals)
- Frozen:    logp = -1556.9045, grad norm = 189.2441, 4.8 us/call (100000 evals)
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

    n_right_censored = int(sum(censored >= high))
    n_left_censored = int(sum(censored <= low))
    n_observed = len(censored) - n_right_censored - n_left_censored
    uncensored = censored[(censored > low) & (censored < high)]

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=((high - low) / 2) + low, sigma=(high - low))
        sigma = pm.HalfNormal("sigma", sigma=(high - low) / 2.0)
        right_censored = pm.Normal(
            "right_censored",
            mu,
            sigma,
            transform=pm.distributions.transforms.Interval(high, None),
            shape=int(n_right_censored),
            initval=np.full(n_right_censored, high + 1),
        )
        left_censored = pm.Normal(
            "left_censored",
            mu,
            sigma,
            transform=pm.distributions.transforms.Interval(None, low),
            shape=int(n_left_censored),
            initval=np.full(n_left_censored, low - 1),
        )
        observed = pm.Normal(
            "observed", mu=mu, sigma=sigma, observed=uncensored, shape=int(n_observed)
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
