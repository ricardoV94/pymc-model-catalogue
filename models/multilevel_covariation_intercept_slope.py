"""
Model: Multilevel Modeling - Covarying Intercept and Slope (Non-centered with LKJ)
Source: pymc-examples/examples/generalized_linear_models/multilevel_modeling.ipynb, Section: "Varying intercept and slope model"
Authors: Chris Fonnesbeck, Colin Carroll, Alex Andorra, Oriol Abril, Farhan Reynaldo
Description: Non-centered varying intercept and slope model with LKJ Cholesky covariance
    prior to capture correlation between county-level intercepts and slopes. Uses
    MvNormal structure via Cholesky factor applied to standard normal auxiliaries.

Changes from original:
- Loaded preprocessed data from .npz file instead of reading CSV files.
- Removed sampling and plotting code.

Benchmark results:
- Original:  logp = -2052.0724, grad norm = 1667.6691, 18.4 us/call (100000 evals)
- Frozen:    logp = -2052.0724, grad norm = 1667.6691, 16.8 us/call (100000 evals)
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pathlib import Path
def build_model():
    data = np.load(Path(__file__).parent / "data" / "radon_mn.npz", allow_pickle=True)
    county = data["county"]
    mn_counties = data["mn_counties"]
    floor_measure = data["floor_measure"]
    log_radon = data["log_radon"]
    coords = {"county": mn_counties, "param": ["alpha", "beta"], "param_bis": ["alpha", "beta"]}

    with pm.Model(coords=coords) as covariation_intercept_slope:
        floor_idx = pm.Data("floor_idx", floor_measure, dims="obs_id")
        county_idx = pm.Data("county_idx", county, dims="obs_id")

        # prior stddev in intercepts & slopes (variation across counties):
        sd_dist = pm.Exponential.dist(0.5, shape=(2,))

        # get back standard deviations and rho:
        chol, corr, stds = pm.LKJCholeskyCov("chol", n=2, eta=2.0, sd_dist=sd_dist)

        # priors for average intercept and slope:
        mu_alpha_beta = pm.Normal("mu_alpha_beta", mu=0.0, sigma=5.0, shape=2)

        # population of varying effects:
        z = pm.Normal("z", 0.0, 1.0, dims=("param", "county"))
        alpha_beta_county = pm.Deterministic(
            "alpha_beta_county", pt.dot(chol, z).T, dims=("county", "param")
        )

        # Expected value per county:
        theta = (
            mu_alpha_beta[0]
            + alpha_beta_county[county_idx, 0]
            + (mu_alpha_beta[1] + alpha_beta_county[county_idx, 1]) * floor_idx
        )

        # Model error:
        sigma = pm.Exponential("sigma", 1.0)

        y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")

    ip = covariation_intercept_slope.initial_point()
    covariation_intercept_slope.rvs_to_initial_values = {rv: None for rv in covariation_intercept_slope.free_RVs}
    return covariation_intercept_slope, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
