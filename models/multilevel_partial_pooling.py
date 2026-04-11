"""
Model: Multilevel Modeling - Partial Pooling
Source: pymc-examples/examples/generalized_linear_models/multilevel_modeling.ipynb, Section: "Partial pooling model"
Authors: Chris Fonnesbeck, Colin Carroll, Alex Andorra, Oriol Abril, Farhan Reynaldo
Description: Partial pooling model with hierarchical Normal prior on county intercepts.
    Ignores floor measurement predictor. Simplest hierarchical model in the series.

Changes from original:
- Loaded preprocessed data from .npz file instead of reading CSV files.
- Removed sampling and plotting code.

Benchmark results:
- Original:  logp = -1971.0288, grad norm = 1192.8144, 9.1 us/call (100000 evals)
- Frozen:    logp = -1971.0288, grad norm = 1192.8144, 7.9 us/call (100000 evals)
"""

import numpy as np
import pymc as pm
from pathlib import Path
def build_model():
    data = np.load(Path(__file__).parent / "data" / "radon_mn.npz", allow_pickle=True)
    county = data["county"]
    mn_counties = data["mn_counties"]
    log_radon = data["log_radon"]
    coords = {"county": mn_counties}

    with pm.Model(coords=coords) as partial_pooling:
        county_idx = pm.Data("county_idx", county, dims="obs_id")

        # Priors
        mu_a = pm.Normal("mu_a", mu=0.0, sigma=10)
        sigma_a = pm.Exponential("sigma_a", 1)

        # Random intercepts
        alpha = pm.Normal("alpha", mu=mu_a, sigma=sigma_a, dims="county")

        # Model error
        sigma_y = pm.Exponential("sigma_y", 1)

        # Expected value
        y_hat = alpha[county_idx]

        # Data likelihood
        y_like = pm.Normal("y_like", mu=y_hat, sigma=sigma_y, observed=log_radon, dims="obs_id")

    ip = partial_pooling.initial_point()
    partial_pooling.rvs_to_initial_values = {rv: None for rv in partial_pooling.free_RVs}
    return partial_pooling, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
