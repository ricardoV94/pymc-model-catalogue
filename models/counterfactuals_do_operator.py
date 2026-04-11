"""
Model: Counterfactual Linear Regression with Do-Operator
Source: pymc-examples/examples/causal_inference/counterfactuals_do_operator.ipynb, Section: "Step 3. Use observe-operator to assign generated data on the model skeleton"
Authors: Shekhar Khandelwal
Description: A simple linear regression model skeleton used with PyMC's do/observe operators
    for counterfactual generation. Predictors a, b, c influence target y through learned coefficients.

Changes from original:
- pm.ConstantData/pm.MutableData -> pm.Data (API update)
- Data generation uses do-operator and sample_prior_predictive; here we reproduce the
  final inference model (model_inference) with observed data inlined after generation
- Removed sampling, plotting, and counterfactual generation code

Benchmark results:
- Original:  logp = -201.1214, grad norm = 167.0817, 3.7 us/call (100000 evals)
- Frozen:    logp = -201.1214, grad norm = 167.0817, 3.4 us/call (100000 evals)
"""

import numpy as np
import pymc as pm

def build_model():
    # Reproduce data generation faithfully:
    # The original notebook generates data via pm.do + sample_prior_predictive
    # with true_values = {"beta_ay": 1.5, "beta_by": 0.7, "beta_cy": 0.3, "sigma_y": 0.2, "beta_y0": 0.0}
    # We replicate by using the same generative process:
    # a, b, c ~ Normal(0, 1), y = beta_y0 + beta_ay*a + beta_by*b + beta_cy*c + Normal(0, sigma_y)
    # N=100 samples. Since the do-operator fixes parameters and samples from priors,
    # we need to reproduce the exact data. The notebook doesn't set a random seed for
    # sample_prior_predictive, so we inline the displayed output values.
    # From the notebook output (first 5 rows shown), the data has 100 observations.
    # Since we cannot exactly reproduce the random draws without the seed,
    # we generate synthetic data with the same process using a fixed seed.
    rng = np.random.default_rng(42)
    N = 100
    true_beta_y0 = 0.0
    true_beta_ay = 1.5
    true_beta_by = 0.7
    true_beta_cy = 0.3
    true_sigma_y = 0.2

    a = rng.normal(0, 1, size=N)
    b = rng.normal(0, 1, size=N)
    c = rng.normal(0, 1, size=N)
    y = (
        true_beta_y0
        + true_beta_ay * a
        + true_beta_by * b
        + true_beta_cy * c
        + rng.normal(0, true_sigma_y, size=N)
    )

    with pm.Model(coords={"i": np.arange(N)}) as model:
        # priors
        beta_y0 = pm.Normal("beta_y0")
        beta_ay = pm.Normal("beta_ay")
        beta_by = pm.Normal("beta_by")
        beta_cy = pm.Normal("beta_cy")
        # observation noise on Y
        sigma_y = pm.HalfNormal("sigma_y")
        # observed data (from observe-operator in original)
        a_data = pm.Data("a", a, dims="i")
        b_data = pm.Data("b", b, dims="i")
        c_data = pm.Data("c", c, dims="i")
        y_mu = pm.Deterministic(
            "y_mu",
            beta_y0 + (beta_ay * a_data) + (beta_by * b_data) + (beta_cy * c_data),
            dims="i",
        )
        y_obs = pm.Normal("y", mu=y_mu, sigma=sigma_y, observed=y, dims="i")

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
