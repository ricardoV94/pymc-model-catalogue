"""
Model: Binomial Regression
Source: pymc-examples/examples/generalized_linear_models/GLM-binomial-regression.ipynb, Section: "Binomial regression model"
Authors: Benjamin T. Vincent
Description: Binomial regression with logit link function on simulated data predicting
    number of successes out of n trials from a single predictor variable.

Changes from original:
- Inlined generated data as numpy arrays.
- Removed sampling and plotting code.

Benchmark results:
- Original:  logp = -291.8498, grad norm = 2357.7822, 3.3 us/call (100000 evals)
- Frozen:    logp = -291.8498, grad norm = 2357.7822, 3.2 us/call (100000 evals)
"""

import numpy as np
import pymc as pm
from scipy.special import expit
def build_model():
    rng = np.random.default_rng(1234)
    # true params
    beta0_true = 0.7
    beta1_true = 0.4
    # number of yes/no questions
    n = 20
    sample_size = 30
    x_data = np.linspace(-10, 20, sample_size)
    # Linear model
    mu_true = beta0_true + beta1_true * x_data
    # transformation (inverse logit function = expit)
    p_true = expit(mu_true)
    # Generate data
    y_data = rng.binomial(n, p_true)
    coords = {"observation": np.arange(sample_size)}

    with pm.Model(coords=coords) as binomial_regression_model:
        x = pm.Data("x", x_data, dims="observation")
        # priors
        beta0 = pm.Normal("beta0", mu=0, sigma=1)
        beta1 = pm.Normal("beta1", mu=0, sigma=1)
        # linear model
        mu = beta0 + beta1 * x
        p = pm.Deterministic("p", pm.math.invlogit(mu), dims="observation")
        # likelihood
        pm.Binomial("y", n=n, p=p, observed=y_data, dims="observation")

    ip = binomial_regression_model.initial_point()
    binomial_regression_model.rvs_to_initial_values = {rv: None for rv in binomial_regression_model.free_RVs}
    return binomial_regression_model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
