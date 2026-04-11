"""
Model: Out-of-Sample Logistic Regression
Source: pymc-examples/examples/generalized_linear_models/GLM-out-of-sample-predictions.ipynb, Section: "Define and Fit the Model"
Authors: Juan Orduz
Description: Logistic regression with interaction term on simulated binary outcome data,
    using MutableData containers for out-of-sample prediction capability.

Changes from original:
- Inlined data generation with fixed seed (RANDOM_SEED=8927).
- pm.MutableData -> pm.Data.
- Removed sampling, plotting, and evaluation code.
- Uses training split only for model definition.

Benchmark results:
- Original:  logp = -124.9765, grad norm = 185.1775, 6.5 us/call (100000 evals)
- Frozen:    logp = -124.9765, grad norm = 185.1775, 6.4 us/call (100000 evals)
"""

import numpy as np
import pymc as pm

from scipy.special import expit as inverse_logit


def build_model():
    RANDOM_SEED = 8927
    rng = np.random.default_rng(RANDOM_SEED)

    # Generate data
    n = 250
    x1 = rng.normal(loc=0.0, scale=2.0, size=n)
    x2 = rng.normal(loc=0.0, scale=2.0, size=n)
    intercept = -0.5
    beta_x1 = 1
    beta_x2 = -1
    beta_interaction = 2
    z = intercept + beta_x1 * x1 + beta_x2 * x2 + beta_interaction * x1 * x2
    p = inverse_logit(z)
    y = rng.binomial(n=1, p=p, size=n)

    # Prepare design matrix
    labels = ["Intercept", "x1", "x2", "x1:x2"]
    intercept_col = np.ones(n)
    interaction = x1 * x2
    x = np.column_stack([intercept_col, x1, x2, interaction])

    # Train-test split
    indices = rng.permutation(x.shape[0])
    train_prop = 0.7
    train_size = int(train_prop * x.shape[0])
    training_idx = indices[:train_size]
    x_train = x[training_idx, :]
    y_train = y[training_idx]

    coords = {"coeffs": labels}

    with pm.Model(coords=coords) as model:
        # data containers
        X = pm.Data("X", x_train)
        y_data = pm.Data("y", y_train)
        # priors
        b = pm.Normal("b", mu=0, sigma=1, dims="coeffs")
        # linear model
        mu = pm.math.dot(X, b)
        # link function
        p = pm.Deterministic("p", pm.math.invlogit(mu))
        # likelihood
        pm.Bernoulli("obs", p=p, observed=y_data)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
