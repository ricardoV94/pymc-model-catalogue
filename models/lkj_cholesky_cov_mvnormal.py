"""
Model: LKJ Cholesky Covariance Prior for Multivariate Normal
Source: pymc-examples/examples/howto/LKJ.ipynb, Section: "LKJ Cholesky Covariance Priors for Multivariate Normal Models"
Authors: Unknown (no `:author:` in notebook `:::{post}` directive)
Description: 2-dimensional multivariate normal model with LKJ Cholesky covariance prior
    (eta=2) and Exponential(1) prior on component standard deviations. Observation
    data is generated from a known mean/covariance with seed 8927.

Changes from original:
- Inlined data generation (random multivariate normal draws with fixed seed).
- Removed sampling and plotting code.

Benchmark results:
- Original:  logp = -57274.3288, grad norm = 62565.2314, 781.0 us/call (47480 evals)
- Frozen:    logp = -57274.3288, grad norm = 62565.2314, 741.8 us/call (57544 evals)
"""

import numpy as np
import pymc as pm


def build_model():
    N = 10000

    mu_actual = np.array([1.0, -2.0])
    sigmas_actual = np.array([0.7, 1.5])
    Rho_actual = np.array([[1.0, -0.4], [-0.4, 1.0]])

    Sigma_actual = np.diag(sigmas_actual) @ Rho_actual @ np.diag(sigmas_actual)

    rng = np.random.default_rng(8927)
    x = rng.multivariate_normal(mu_actual, Sigma_actual, size=N)

    coords = {"axis": ["y", "z"], "axis_bis": ["y", "z"], "obs_id": np.arange(N)}
    with pm.Model(coords=coords) as model:
        chol, corr, stds = pm.LKJCholeskyCov(
            "chol", n=2, eta=2.0, sd_dist=pm.Exponential.dist(1.0, shape=2)
        )
        cov = pm.Deterministic("cov", chol.dot(chol.T), dims=("axis", "axis_bis"))

        mu = pm.Normal("mu", 0.0, sigma=1.5, dims="axis")
        obs = pm.MvNormal("obs", mu, chol=chol, observed=x, dims=("obs_id", "axis"))

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
