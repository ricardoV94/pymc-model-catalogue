"""
Model: GLM Poisson Regression with Interaction Terms
Source: pymc-examples/examples/generalized_linear_models/GLM-poisson-regression.ipynb, Section: "Poisson Regression"
Authors: Jonathan Sedar, Benjamin Vincent
Description: Poisson regression predicting sneeze counts from alcohol consumption and
    antihistamine usage, with interaction term. Uses design matrix from formulae library.

Changes from original:
- Inlined generated data (design matrix columns and response) as numpy arrays instead of
  using formulae library to create design matrices from a DataFrame.
- Removed sampling and plotting code.

Benchmark results:
- Original:  logp = -109560.6517, grad norm = 77037.4949, 27.3 us/call (100000 evals)
- Frozen:    logp = -109560.6517, grad norm = 77037.4949, 27.9 us/call (100000 evals)
"""

import numpy as np
import pymc as pm

def build_model():
    RANDOM_SEED = 8927
    rng = np.random.default_rng(RANDOM_SEED)
    # Reproduce the data generation from the notebook
    q = 1000
    theta_noalcohol_meds = 1
    theta_alcohol_meds = 3
    theta_noalcohol_nomeds = 6
    theta_alcohol_nomeds = 36
    nsneeze = np.concatenate((
        rng.poisson(theta_noalcohol_meds, q),
        rng.poisson(theta_alcohol_meds, q),
        rng.poisson(theta_noalcohol_nomeds, q),
        rng.poisson(theta_alcohol_nomeds, q),
    ))
    alcohol = np.concatenate((
        np.repeat(False, q),
        np.repeat(True, q),
        np.repeat(False, q),
        np.repeat(True, q),
    )).astype(float)
    nomeds = np.concatenate((
        np.repeat(False, q),
        np.repeat(False, q),
        np.repeat(True, q),
        np.repeat(True, q),
    )).astype(float)
    alcohol_nomeds = alcohol * nomeds
    # Design matrix columns (from formulae: "nsneeze ~ alcohol * nomeds")
    # Intercept column is all ones, handled by b0 in the model
    mx_alcohol = alcohol
    mx_nomeds = nomeds
    mx_interaction = alcohol_nomeds
    mx_response = nsneeze
    with pm.Model() as model:
        # define priors, weakly informative Normal
        b0 = pm.Normal("Intercept", mu=0, sigma=10)
        b1 = pm.Normal("alcohol", mu=0, sigma=10)
        b2 = pm.Normal("nomeds", mu=0, sigma=10)
        b3 = pm.Normal("alcohol:nomeds", mu=0, sigma=10)

        # define linear model and exp link function
        theta = (
            b0
            + b1 * mx_alcohol
            + b2 * mx_nomeds
            + b3 * mx_interaction
        )

        ## Define Poisson likelihood
        y = pm.Poisson("y", mu=pm.math.exp(theta), observed=mx_response)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip

if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
