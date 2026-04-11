"""
Model: LASSO regression with separate Laplace coefficients
Source: pymc-examples/examples/samplers/lasso_block_update.ipynb, Section: "Lasso regression with block updating"
Authors: Chris Fonnesbeck, Raul Maldonado, Michael Osthege, Thomas Wiecki, Lorenzo Toniazzi
Description: A Bayesian LASSO regression with two strongly-correlated covariates. The
    regression coefficients are given separate Laplace priors with a shared scale b = lam * tau,
    and tau is a Uniform(0, 1) hyperparameter. sigma has an Exponential(1) prior.

Changes from original:
- Inlined simulated data generation with the original seed (rng default_rng(8927))
- Removed sampling and plotting code

Benchmark results:
- Original:  logp = -143.8029, grad norm = 437.9676, 2.3 us/call (100000 evals)
- Frozen:    logp = -143.8029, grad norm = 437.9676, 2.6 us/call (100000 evals)
"""

import numpy as np
import pymc as pm


def build_model():
    RANDOM_SEED = 8927
    rng = np.random.default_rng(RANDOM_SEED)

    x = rng.standard_normal(size=(3, 30))
    x1 = x[0] + 4
    x2 = x[1] + 4
    noise = x[2]
    y_obs = x1 * 0.2 + x2 * 0.3 + noise

    lam = 3000

    with pm.Model() as model:
        sigma = pm.Exponential("sigma", 1)
        tau = pm.Uniform("tau", 0, 1)
        b = lam * tau
        beta1 = pm.Laplace("beta1", 0, b)
        beta2 = pm.Laplace("beta2", 0, b)

        mu = x1 * beta1 + x2 * beta2

        y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
