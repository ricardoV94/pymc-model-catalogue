"""
Model: GP Marginal with Matern52 covariance
Source: pymc-examples/examples/gaussian_processes/GP-Marginal.ipynb, Section: "The .marginal_likelihood method"
Authors: Bill Engels, Chris Fonnesbeck
Description: Marginal GP with Matern52 covariance and white noise on 100
    synthetically generated data points.

Changes from original:
- Removed sampling, plotting, conditional predictions
- Inlined synthetic data generation with fixed seed (np.random.seed(1))
- Added ip capture and initval clearing

Benchmark results:
- Original:  logp = -276.3962, grad norm = 74.6018, 610.6 us/call (22031 evals)
- Frozen:    logp = -276.3962, grad norm = 74.6018, 634.5 us/call (20158 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    np.random.seed(1)

    n = 100
    X = np.linspace(0, 10, n)[:, None]

    ell_true = 1.0
    eta_true = 3.0
    cov_func = eta_true**2 * pm.gp.cov.Matern52(1, ell_true)
    mean_func = pm.gp.mean.Zero()

    f_true = np.random.multivariate_normal(
        mean_func(X).eval(), cov_func(X).eval() + 1e-8 * np.eye(n), 1
    ).flatten()

    sigma_true = 2.0
    y = f_true + sigma_true * np.random.randn(n)

    with pm.Model() as model:
        ell = pm.Gamma("ell", alpha=2, beta=1)
        eta = pm.HalfCauchy("eta", beta=5)

        cov = eta**2 * pm.gp.cov.Matern52(1, ell)
        gp = pm.gp.Marginal(cov_func=cov)

        sigma = pm.HalfCauchy("sigma", beta=5)
        y_ = gp.marginal_likelihood("y", X=X, y=y, sigma=sigma)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
