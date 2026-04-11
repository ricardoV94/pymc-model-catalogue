"""
Model: Multilevel Modeling - Hierarchical Intercept with Group-level Predictor
Source: pymc-examples/examples/generalized_linear_models/multilevel_modeling.ipynb, Section: "Adding group-level predictors"
Authors: Chris Fonnesbeck, Colin Carroll, Alex Andorra, Oriol Abril, Farhan Reynaldo
Description: Hierarchical intercept model where the county-level intercept is modeled as
    a linear function of log-uranium concentration. Uses non-centered parameterization
    for county deviations.

Changes from original:
- Loaded preprocessed data from .npz file instead of reading CSV files.
- Removed sampling and plotting code.

Benchmark results:
- Original:  logp = -4530.3763, grad norm = 459.0832, 8.4 us/call (100000 evals)
- Frozen:    logp = -4530.3763, grad norm = 459.0832, 8.9 us/call (100000 evals)
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
    u = data["u"]
    coords = {"county": mn_counties}

    with pm.Model(coords=coords) as hierarchical_intercept:
        # Priors
        sigma_a = pm.HalfCauchy("sigma_a", 5)

        # County uranium model
        gamma_0 = pm.Normal("gamma_0", mu=0.0, sigma=10.0)
        gamma_1 = pm.Normal("gamma_1", mu=0.0, sigma=10.0)

        # Uranium model for intercept
        mu_a = pm.Deterministic("mu_a", gamma_0 + gamma_1 * u)
        # County variation not explained by uranium
        epsilon_a = pm.Normal("epsilon_a", mu=0, sigma=1, dims="county")
        alpha = pm.Deterministic("alpha", mu_a + sigma_a * epsilon_a, dims="county")

        # Common slope
        beta = pm.Normal("beta", mu=0.0, sigma=10.0)

        # Model error
        sigma_y = pm.Uniform("sigma_y", lower=0, upper=100)

        # Expected value
        y_hat = alpha[county] + beta * floor_measure

        # Data likelihood
        y_like = pm.Normal("y_like", mu=y_hat, sigma=sigma_y, observed=log_radon)

    ip = hierarchical_intercept.initial_point()
    hierarchical_intercept.rvs_to_initial_values = {rv: None for rv in hierarchical_intercept.free_RVs}
    return hierarchical_intercept, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
