"""
Model: Multilevel Modeling - Varying Intercept
Source: pymc-examples/examples/generalized_linear_models/multilevel_modeling.ipynb, Section: "Partial pooling model"
Authors: Chris Fonnesbeck, Colin Carroll, Alex Andorra, Oriol Abril, Farhan Reynaldo
Description: Varying intercept model with hierarchical Normal prior on county intercepts
    and a common floor effect slope. Extends partial pooling by adding floor predictor.

Changes from original:
- Loaded preprocessed data from .npz file instead of reading CSV files.
- Removed sampling and plotting code.

Benchmark results:
- Original:  logp = -1974.2503, grad norm = 1198.7085, 10.7 us/call (100000 evals)
- Frozen:    logp = -1974.2503, grad norm = 1198.7085, 8.7 us/call (100000 evals)
"""

import numpy as np
import pymc as pm
from pathlib import Path
def build_model():
    data = np.load(Path(__file__).parent / "data" / "radon_mn.npz", allow_pickle=True)
    county = data["county"]
    mn_counties = data["mn_counties"]
    floor_measure = data["floor_measure"]
    log_radon = data["log_radon"]
    coords = {"county": mn_counties}

    with pm.Model(coords=coords) as varying_intercept:
        floor_idx = pm.Data("floor_idx", floor_measure, dims="obs_id")
        county_idx = pm.Data("county_idx", county, dims="obs_id")

        # Priors
        mu_a = pm.Normal("mu_a", mu=0.0, sigma=10.0)
        sigma_a = pm.Exponential("sigma_a", 1)

        # Random intercepts
        alpha = pm.Normal("alpha", mu=mu_a, sigma=sigma_a, dims="county")
        # Common slope
        beta = pm.Normal("beta", mu=0.0, sigma=10.0)

        # Model error
        sd_y = pm.Exponential("sd_y", 1)

        # Expected value
        y_hat = alpha[county_idx] + beta * floor_idx

        # Data likelihood
        y_like = pm.Normal("y_like", mu=y_hat, sigma=sd_y, observed=log_radon, dims="obs_id")

    ip = varying_intercept.initial_point()
    varying_intercept.rvs_to_initial_values = {rv: None for rv in varying_intercept.free_RVs}
    return varying_intercept, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
