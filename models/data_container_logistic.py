"""
Model: Binomial GLM with pm.Data shape-linking for out-of-sample prediction
Source: pymc-examples/examples/fundamentals/data_container.ipynb, Section: "Applied Example: Using Data containers as input to a binomial GLM"
Authors: Juan Martin Loyola, Kavya Jaiswal, Oriol Abril, Jesse Grabowski
Description: Logistic regression p = sigmoid(alpha + beta * x) with Bernoulli observations,
    wrapping both x and y in pm.Data and linking the observation shape to x_data.shape[0]
    to enable swapping x for prediction grids without reshaping y.

Changes from original:
- Inlined small synthetic data (n=100, seed from "Data Containers in PyMC")
- Removed sampling and posterior predictive code

Benchmark results:
- Original:  logp = -71.1526, grad norm = 23.6388, 3.9 us/call (100000 evals)
- Frozen:    logp = -71.1526, grad norm = 23.6388, 3.7 us/call (100000 evals)
"""

import numpy as np
import pymc as pm


def build_model():
    RANDOM_SEED = sum(map(ord, "Data Containers in PyMC"))
    rng = np.random.default_rng(RANDOM_SEED)

    n_obs = 100
    true_beta = 1.5
    true_alpha = 0.25

    x = rng.normal(size=n_obs)
    true_p = 1 / (1 + np.exp(-(true_alpha + true_beta * x)))
    y = rng.binomial(n=1, p=true_p)

    with pm.Model() as model:
        x_data = pm.Data("x", x)
        y_data = pm.Data("y", y)

        alpha = pm.Normal("alpha")
        beta = pm.Normal("beta")

        p = pm.Deterministic("p", pm.math.sigmoid(alpha + beta * x_data))

        # Link the shape of the observation to the input data shape
        obs = pm.Bernoulli("obs", p=p, observed=y_data, shape=x_data.shape[0])

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
