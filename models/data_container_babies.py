"""
Model: Toddler length as a function of age
Source: pymc-examples/examples/fundamentals/data_container.ipynb, Section: "Applied example: height of toddlers as a function of age"
Authors: Juan Martin Loyola, Kavya Jaiswal, Oriol Abril, Jesse Grabowski
Description: Heteroscedastic regression of infant length on age (months), with
    mean mu = a0 + a1 * sqrt(month) and scale sigma = b0 + b1 * month. Both mean
    and scale parameters are shared Normal(0, 10) random variables indexed by a
    "parameter" dim. Uses pm.Data for month and dims=["obs_idx"] to allow set_data
    out-of-sample prediction. Data: 800 WHO-style infant Length/Month observations.

Changes from original:
- Loaded babies.csv from models/data/babies.npz instead of pm.get_data
- Added `initval=[1.0, 0.0]` on sigma_params so the default initial point produces
  a positive observation scale. The notebook-default init of [0, 0] makes
  sigma = 0, which yields -inf logp on the Normal likelihood.
- Removed sampling and posterior predictive code

Benchmark results:
- Original:  logp = -2105841.1669, grad norm = 57339520.2130, 5.4 us/call (100000 evals)
- Frozen:    logp = -2105841.1669, grad norm = 57339520.2130, 5.3 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data = np.load(Path(__file__).parent / "data" / "babies.npz")
    month_values = data["month"].astype(float)
    length_values = data["length"].astype(float)
    n_obs = len(month_values)

    with pm.Model(
        coords={"obs_idx": np.arange(n_obs), "parameter": ["intercept", "slope"]}
    ) as model:
        mean_params = pm.Normal("mean_params", sigma=10, dims=["parameter"])
        sigma_params = pm.Normal(
            "sigma_params", sigma=10, dims=["parameter"], initval=[1.0, 0.0]
        )
        month = pm.Data("month", month_values, dims=["obs_idx"])

        mu = pm.Deterministic(
            "mu", mean_params[0] + mean_params[1] * month**0.5, dims=["obs_idx"]
        )
        sigma = pm.Deterministic(
            "sigma", sigma_params[0] + sigma_params[1] * month, dims=["obs_idx"]
        )

        length = pm.Normal(
            "length", mu=mu, sigma=sigma, observed=length_values, dims=["obs_idx"]
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
