"""
Model: HSGP Basic 1D with Matern52
Source: pymc-examples/examples/gaussian_processes/HSGP-Basic.ipynb, Section: "HSGP Reference & First Steps"
Authors: Bill Engels, Alexandre Andorra
Description: 1D Hilbert Space GP approximation with Matern52 covariance on
    2000 simulated data points, using m=200 basis vectors and c=1.5.

Changes from original:
- Removed sampling, plotting, posterior predictive
- Inlined synthetic data generation with fixed seed
- Hardcoded LogNormal lengthscale prior, precomputed from
  pz.maxent(LogNormal, lower=0.5, upper=5.0, mass=0.9) so the build needs no preliz
- Added ip capture and initval clearing

Benchmark results:
- Original:  logp = -3688.7606, grad norm = 2113.5036, 78.3 us/call (100000 evals)
- Frozen:    logp = -3688.7606, grad norm = 2113.5036, 84.3 us/call (46969 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    seed = sum(map(ord, "hsgp"))
    rng = np.random.default_rng(seed)

    x = 100.0 * np.sort(rng.random(2000))

    # Simulate 1D GP
    n = len(x)
    ell_true = 1.0
    eta_true = 1.0
    sigma_true = 1.0

    cov_func_true = eta_true**2 * pm.gp.cov.Matern52(1, ell_true)
    cov_mat_stabilized = pm.gp.util.stabilize(cov_func_true(x[:, None]))
    gp_true = pm.MvNormal.dist(mu=np.zeros(n), cov=cov_mat_stabilized)
    f_true = pm.draw(gp_true, draws=1, random_seed=rng)

    noise_dist = pm.Normal.dist(mu=0.0, sigma=sigma_true)
    y_obs = f_true + pm.draw(noise_dist, draws=n, random_seed=rng)

    coords = {
        "basis_coeffs": np.arange(200),
        "obs_id": np.arange(len(y_obs)),
    }

    with pm.Model(coords=coords) as model:
        # precomputed from pz.maxent(LogNormal, lower=0.5, upper=5.0, mass=0.9)
        ell = pm.LogNormal("ell", mu=0.8307925109013862, sigma=0.5937605781861558)
        eta = pm.Exponential("eta", scale=1.0)
        cov_func = eta**2 * pm.gp.cov.Matern52(input_dim=1, ls=ell)

        m, c = 200, 1.5
        parametrization = "centered"
        gp = pm.gp.HSGP(
            m=[m], c=c, parametrization=parametrization, cov_func=cov_func
        )
        f = gp.prior(
            "f",
            X=x[:, None],
            hsgp_coeffs_dims="basis_coeffs",
            gp_dims="obs_id",
        )

        sigma = pm.Exponential("sigma", scale=1.0)
        pm.Normal("y_obs", mu=f, sigma=sigma, observed=y_obs, dims="obs_id")

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
