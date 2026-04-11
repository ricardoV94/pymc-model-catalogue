"""
Model: Multilevel Modeling - Pooled (Complete Pooling)
Source: pymc-examples/examples/generalized_linear_models/multilevel_modeling.ipynb, Section: "Conventional approaches"
Authors: Chris Fonnesbeck, Colin Carroll, Alex Andorra, Oriol Abril, Farhan Reynaldo
Description: Complete pooling linear regression of log-radon levels on floor measurement,
    ignoring county-level variation. Baseline conventional approach for comparison with
    hierarchical models.

Changes from original:
- Loaded preprocessed data from .npz file instead of reading CSV files.
- Removed sampling and plotting code.

Benchmark results:
- Original:  logp = -25452.6984, grad norm = 58981.3439, 5.0 us/call (100000 evals)
- Frozen:    logp = -25452.6984, grad norm = 58981.3439, 4.5 us/call (100000 evals)
"""

import numpy as np
import pymc as pm
from pathlib import Path
def build_model():
    data = np.load(Path(__file__).parent / "data" / "radon_mn.npz", allow_pickle=True)
    floor_measure = data["floor_measure"]
    log_radon = data["log_radon"]
    with pm.Model() as pooled_model:
        floor_ind = pm.Data("floor_ind", floor_measure, dims="obs_id")

        alpha = pm.Normal("alpha", 0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma = pm.Exponential("sigma", 5)

        theta = alpha + beta * floor_ind

        y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")

    ip = pooled_model.initial_point()
    pooled_model.rvs_to_initial_values = {rv: None for rv in pooled_model.free_RVs}
    return pooled_model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
