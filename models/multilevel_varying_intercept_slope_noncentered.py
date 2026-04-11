"""
Model: Multilevel Modeling - Varying Intercept and Slope (Non-centered)
Source: pymc-examples/examples/generalized_linear_models/multilevel_modeling.ipynb, Section: "Non-centered Parameterization"
Authors: Chris Fonnesbeck, Colin Carroll, Alex Andorra, Oriol Abril, Farhan Reynaldo
Description: Non-centered parameterization of varying intercept and slope model for radon
    levels. Uses standard normal auxiliaries (z_a, z_b) multiplied by group-level standard
    deviations to avoid the funnel geometry that causes divergences in the centered version.

Changes from original:
- Loaded preprocessed data from .npz file instead of reading CSV files.
- Removed sampling and plotting code.

Benchmark results:
- Original:  logp = -25610.9180, grad norm = 58992.6213, 13.6 us/call (100000 evals)
- Frozen:    logp = -25610.9180, grad norm = 58992.6213, 12.9 us/call (100000 evals)
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

    with pm.Model(coords=coords) as varying_intercept_slope_noncentered:
        floor_idx = pm.Data("floor_idx", floor_measure, dims="obs_id")
        county_idx = pm.Data("county_idx", county, dims="obs_id")

        # Priors
        mu_a = pm.Normal("mu_a", mu=0.0, sigma=10.0)
        sigma_a = pm.Exponential("sigma_a", 5)

        # Non-centered random intercepts
        # Centered: a = pm.Normal('a', mu_a, sigma=sigma_a, shape=counties)
        z_a = pm.Normal("z_a", mu=0, sigma=1, dims="county")
        alpha = pm.Deterministic("alpha", mu_a + z_a * sigma_a, dims="county")

        mu_b = pm.Normal("mu_b", mu=0.0, sigma=10.0)
        sigma_b = pm.Exponential("sigma_b", 5)

        # Non-centered random slopes
        z_b = pm.Normal("z_b", mu=0, sigma=1, dims="county")
        beta = pm.Deterministic("beta", mu_b + z_b * sigma_b, dims="county")

        # Model error
        sigma_y = pm.Exponential("sigma_y", 5)

        # Expected value
        y_hat = alpha[county_idx] + beta[county_idx] * floor_idx

        # Data likelihood
        y_like = pm.Normal("y_like", mu=y_hat, sigma=sigma_y, observed=log_radon, dims="obs_id")

    ip = varying_intercept_slope_noncentered.initial_point()
    varying_intercept_slope_noncentered.rvs_to_initial_values = {rv: None for rv in varying_intercept_slope_noncentered.free_RVs}
    return varying_intercept_slope_noncentered, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
