"""
Model: Updating Priors - Base Linear Regression
Source: pymc-examples/examples/howto/updating_priors.ipynb, Section: "Model specification"
Authors: David Brochart, Juan Orduz
Description: Linear regression with two predictors and Normal priors on alpha/beta0/beta1
    and HalfNormal on sigma. This is the initial (non-updated) model from the notebook;
    the subsequent iterations replace these priors with pm.Interpolated distributions
    built from the previous round's posterior, which is not extracted here because
    Interpolated priors do not numba-compile cleanly.

Changes from original:
- Inlined data generation (size=100) with the notebook's seed=42.
- Removed sampling and plotting code.
- Did not extract the Interpolated-prior variant used in subsequent update iterations.

Benchmark results:
- Original:  logp = -3231.3921, grad norm = 6192.8108, 2.4 us/call (100000 evals)
- Frozen:    logp = -3231.3921, grad norm = 6192.8108, 2.3 us/call (100000 evals)
"""

import numpy as np
import pymc as pm


def build_model():
    rng: np.random.Generator = np.random.default_rng(seed=42)

    # True parameter values
    alpha_true = 5
    beta0_true = 7
    beta1_true = 13
    sigma_true = 2

    # Size of dataset
    size = 100

    # Predictor variable
    X1 = rng.normal(size=size)
    X2 = rng.normal(size=size) * 0.2

    # Simulate outcome variable
    Y = alpha_true + beta0_true * X1 + beta1_true * X2 + rng.normal(size=size, scale=sigma_true)

    with pm.Model() as model:
        # Priors for unknown model parameters
        alpha = pm.Normal("alpha", mu=0, sigma=5)
        beta0 = pm.Normal("beta0", mu=0, sigma=5)
        beta1 = pm.Normal("beta1", mu=0, sigma=5)

        sigma = pm.HalfNormal("sigma", sigma=1)

        # Expected value of outcome
        mu = alpha + beta0 * X1 + beta1 * X2

        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
