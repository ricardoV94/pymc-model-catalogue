"""
Model: Hierarchical Radon GLM (ADVI Mini-batch notebook)
Source: pymc-examples/examples/variational_inference/GLM-hierarchical-advi-minibatch.ipynb, Section: "GLM: Mini-batch ADVI on hierarchical regression model"
Authors: PyMC Developers
Description: Hierarchical linear regression of log-radon on floor with county-level
    varying intercepts and slopes. Centered parameterization with wide Normal
    hyperpriors and Uniform(0, 100) priors on all group and observation standard
    deviations, as specified in the original ADVI mini-batch notebook.

Changes from original:
- Loaded preprocessed radon data from existing .npz file instead of reading CSV.
- Removed pm.Minibatch wrappers and total_size (VI-only apparatus); the underlying
  model is the same as the "Inference button" block at the end of the notebook
  which uses full-data likelihood.
- Removed sampling/fitting and plotting code.

Benchmark results:
- Original:  logp = -5285.7518, grad norm = 463.0005, 8.0 us/call (100000 evals)
- Frozen:    logp = -5285.7518, grad norm = 463.0005, 8.2 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data = np.load(Path(__file__).parent / "data" / "radon_mn.npz", allow_pickle=True)
    county_idx = data["county"]
    mn_counties = data["mn_counties"]
    floor_idx = data["floor_measure"]
    log_radon_idx = data["log_radon"]

    coords = {"counties": mn_counties}

    with pm.Model(coords=coords) as hierarchical_model:
        # Hyperpriors for group nodes
        mu_a = pm.Normal("mu_alpha", mu=0.0, sigma=100**2)
        sigma_a = pm.Uniform("sigma_alpha", lower=0, upper=100)
        mu_b = pm.Normal("mu_beta", mu=0.0, sigma=100**2)
        sigma_b = pm.Uniform("sigma_beta", lower=0, upper=100)

        a = pm.Normal("alpha", mu=mu_a, sigma=sigma_a, dims="counties")
        b = pm.Normal("beta", mu=mu_b, sigma=sigma_b, dims="counties")

        # Model error
        eps = pm.Uniform("eps", lower=0, upper=100)

        radon_est = a[county_idx] + b[county_idx] * floor_idx

        radon_like = pm.Normal(
            "radon_like", mu=radon_est, sigma=eps, observed=log_radon_idx
        )

    ip = hierarchical_model.initial_point()
    hierarchical_model.rvs_to_initial_values = {rv: None for rv in hierarchical_model.free_RVs}
    return hierarchical_model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
