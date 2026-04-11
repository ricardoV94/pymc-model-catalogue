"""
Model: Ordinal Regression - Ordered Logistic (Constrained Uniform Cutpoints)
Source: pymc-examples/examples/generalized_linear_models/GLM-ordinal-regression.ipynb, Section: "Fit a variety of Model Specifications"
Authors: Nathaniel Forde
Description: Ordered logistic regression on synthetic employee rating data with salary,
    work satisfaction, and work-from-home predictors, using Dirichlet-based constrained
    uniform cutpoints (model_spec=3, constrained_uniform=True).

Changes from original:
- Loaded data from .npz file instead of CSV.
- Extracted model_spec=3, constrained_uniform=True, logit directly.
- Removed sampling, plotting, and model comparison code.

Benchmark results:
- Original:  logp = -2416.4910, grad norm = 19100.9126, 58.1 us/call (100000 evals)
- Frozen:    logp = -2416.4910, grad norm = 19100.9126, 58.3 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    data = np.load(Path(__file__).parent / "data" / "fake_employee_rating.npz")
    salary = data["salary"]
    work_sat = data["work_sat"]
    work_from_home = data["work_from_home"]
    explicit_rating = data["explicit_rating"].astype(int)
    K = len(np.unique(explicit_rating))

    with pm.Model() as model:
        # Constrained Uniform cutpoints via Dirichlet
        cutpoints = pm.Deterministic(
            "cutpoints",
            pt.concatenate(
                [
                    np.ones(1) * 0,
                    pt.extra_ops.cumsum(pm.Dirichlet("cuts_unknown", a=np.ones(K - 2)))
                    * (K - 0)
                    + 0,
                ]
            ),
        )

        beta = pm.Normal("beta", 0, 1, size=3)
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
