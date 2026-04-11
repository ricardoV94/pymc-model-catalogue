"""
Model: Multilevel Modeling - Unpooled (No Pooling)
Source: pymc-examples/examples/generalized_linear_models/multilevel_modeling.ipynb, Section: "Conventional approaches"
Authors: Chris Fonnesbeck, Colin Carroll, Alex Andorra, Oriol Abril, Farhan Reynaldo
Description: No-pooling linear regression of log-radon levels with separate intercept per
    county. Baseline conventional approach for comparison with hierarchical models.

Changes from original:
- Loaded preprocessed data from .npz file instead of reading CSV files.
- Removed sampling and plotting code.

Benchmark results:
- Original:  logp = -2165.7485, grad norm = 1195.6910, 10.1 us/call (100000 evals)
- Frozen:    logp = -2165.7485, grad norm = 1195.6910, 8.1 us/call (100000 evals)
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

    with pm.Model(coords=coords) as unpooled_model:
        floor_ind = pm.Data("floor_ind", floor_measure, dims="obs_id")

        alpha = pm.Normal("alpha", 0, sigma=10, dims="county")
        beta = pm.Normal("beta", 0, sigma=10)
        sigma = pm.Exponential("sigma", 1)

        theta = alpha[county] + beta * floor_ind

        y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")

    ip = unpooled_model.initial_point()
    unpooled_model.rvs_to_initial_values = {rv: None for rv in unpooled_model.free_RVs}
    return unpooled_model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
