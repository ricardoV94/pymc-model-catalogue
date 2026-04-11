"""
Model: Hypothesis testing — Normal mean estimation
Source: pymc-examples/examples/howto/hypothesis_testing.ipynb, Section: "Sampling from the prior and posterior"
Authors: Benjamin T. Vincent
Description: Simple single-variable Normal model (mu, sigma) used to demonstrate Bayesian
    hypothesis testing methods (posterior probability statements, HDIs, ROPE, Bayes factors)
    on synthetic data drawn from N(2, 3).

Changes from original:
- Inlined the 12 synthetic observations generated with np.random.default_rng(42)
- Removed sampling and plotting code

Benchmark results:
- Original:  logp = -36.6635, grad norm = 18.7895, 2.0 us/call (100000 evals)
- Frozen:    logp = -36.6635, grad norm = 18.7895, 2.0 us/call (100000 evals)
"""

import numpy as np
import pymc as pm


def build_model():
    # Synthetic observations from np.random.default_rng(42).normal(loc=2.0, scale=3.0, size=12)
    x = np.array([
        2.91415124, -1.11995232, 4.25135359, 4.82169415, -3.85310557,
        -1.90653852, 2.38352121, 1.05127222, 1.94959653, -0.55913178,
        4.63819392, 4.33337581,
    ])

    with pm.Model() as model:
        # priors
        mu = pm.Normal("mu", mu=0, sigma=2)
        sigma = pm.Gamma("sigma", alpha=2, beta=1)
        # likelihood
        pm.Normal("y", mu=mu, sigma=sigma, observed=x)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
