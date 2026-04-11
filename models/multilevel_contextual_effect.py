"""
Model: Multilevel Modeling - Contextual Effect
Source: pymc-examples/examples/generalized_linear_models/multilevel_modeling.ipynb, Section: "Correlations among levels"
Authors: Chris Fonnesbeck, Colin Carroll, Alex Andorra, Oriol Abril, Farhan Reynaldo
Description: Contextual effect model where county intercepts depend on both log-uranium
    and average floor measurement (a county-level summary of the individual predictor).
    Non-centered parameterization for county deviations.

Changes from original:
- Loaded preprocessed data from .npz file instead of reading CSV files.
- Removed sampling, prediction, and plotting code.

Benchmark results:
- Original:  logp = -4533.5978, grad norm = 459.0832, 10.3 us/call (100000 evals)
- Frozen:    logp = -4533.5978, grad norm = 459.0832, 9.6 us/call (100000 evals)
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
    avg_floor_data = data["avg_floor_data"]
    coords = {"county": mn_counties}

    with pm.Model(coords=coords) as contextual_effect:
        floor_idx = pm.Data("floor_idx", floor_measure)
        county_idx = pm.Data("county_idx", county)
        y = pm.Data("y", log_radon)

        # Priors
        sigma_a = pm.HalfCauchy("sigma_a", 5)

        # County uranium model for slope
        gamma = pm.Normal("gamma", mu=0.0, sigma=10, shape=3)

        # Uranium model for intercept
        mu_a = pm.Deterministic("mu_a", gamma[0] + gamma[1] * u + gamma[2] * avg_floor_data)

        # County variation not explained by uranium
        epsilon_a = pm.Normal("epsilon_a", mu=0, sigma=1, dims="county")
        alpha = pm.Deterministic("alpha", mu_a + sigma_a * epsilon_a)

        # Common slope
        beta = pm.Normal("beta", mu=0.0, sigma=10)

        # Model error
        sigma_y = pm.Uniform("sigma_y", lower=0, upper=100)

        # Expected value
        y_hat = alpha[county_idx] + beta * floor_idx

        # Data likelihood
        y_like = pm.Normal("y_like", mu=y_hat, sigma=sigma_y, observed=y)

    ip = contextual_effect.initial_point()
    contextual_effect.rvs_to_initial_values = {rv: None for rv in contextual_effect.free_RVs}
    return contextual_effect, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
