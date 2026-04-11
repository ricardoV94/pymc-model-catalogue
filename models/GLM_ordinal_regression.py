"""
Model: Ordinal Regression - Ordered Logistic (3 predictors, Normal cutpoints)
Source: pymc-examples/examples/generalized_linear_models/GLM-ordinal-regression.ipynb, Section: "Fit a variety of Model Specifications"
Authors: Nathaniel Forde
Description: Ordered logistic regression on synthetic employee rating data with salary,
    work satisfaction, and work-from-home predictors, using Normal priors on cutpoints
    with univariate ordered transform (model_spec=3).

Changes from original:
- univariate_ordered -> ordered (API update)
- Loaded data from .npz file instead of CSV.
- Extracted model_spec=3 (3 predictors, Normal cutpoints, logit) directly.
- Removed sampling, plotting, and model comparison code.

Benchmark results:
- Original:  logp = -2417.9870, grad norm = 19106.4474, 58.9 us/call (100000 evals)
- Frozen:    logp = -2417.9870, grad norm = 19106.4474, 59.5 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data = np.load(Path(__file__).parent / "data" / "fake_employee_rating.npz")
    salary = data["salary"]
    work_sat = data["work_sat"]
    work_from_home = data["work_from_home"]
    explicit_rating = data["explicit_rating"].astype(int)
    K = len(np.unique(explicit_rating))

    priors_sigma = 1
    priors_beta_mu = 0
    priors_beta_sigma = 1
    priors_mu = np.linspace(0, K, K - 1)

    with pm.Model() as model:
        sigma = pm.Exponential("sigma", priors_sigma)
        cutpoints = pm.Normal(
            "cutpoints",
            mu=priors_mu,
            sigma=sigma,
            transform=pm.distributions.transforms.ordered,
        )

        beta = pm.Normal("beta", priors_beta_mu, priors_beta_sigma, size=3)
        mu = pm.Deterministic(
            "mu", beta[0] * salary + beta[1] * work_sat + beta[2] * work_from_home
        )

        pm.OrderedLogistic("y", cutpoints=cutpoints, eta=mu, observed=explicit_rating)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
