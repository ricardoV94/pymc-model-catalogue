"""
Model: GP Latent with Student-t likelihood
Source: pymc-examples/examples/gaussian_processes/GP-Latent.ipynb, Section: "The .prior method"
Authors: Bill Engels, Chris Fonnesbeck
Description: Latent GP with ExpQuad covariance and Student-t likelihood,
    demonstrating robust GP regression on 50 data points.

Changes from original:
- Removed sampling, plotting, conditional predictions
- Inlined synthetic data generation with fixed seed
- Added ip capture and initval clearing

Benchmark results:
- Original:  logp = -193.5441, grad norm = 91.0191, 99.2 us/call (98099 evals)
- Frozen:    logp = -193.5441, grad norm = 91.0191, 100.9 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    RANDOM_SEED = 8998
    rng = np.random.default_rng(RANDOM_SEED)

    n = 50
    X = np.linspace(0, 10, n)[:, None]

    ell_true = 1.0
    eta_true = 4.0
    cov_func = eta_true**2 * pm.gp.cov.ExpQuad(1, ell_true)
    mean_func = pm.gp.mean.Zero()

    f_true = pm.draw(
        pm.MvNormal.dist(mu=mean_func(X), cov=cov_func(X), method="svd"),
        1,
        random_seed=rng,
    )

    sigma_true = 1.0
    nu_true = 5.0
    y = f_true + sigma_true * rng.standard_t(df=nu_true, size=n)

    with pm.Model() as model:
        ell = pm.Gamma("ell", alpha=2, beta=1)
        eta = pm.HalfNormal("eta", sigma=5)

        cov = eta**2 * pm.gp.cov.ExpQuad(1, ell)
        gp = pm.gp.Latent(cov_func=cov)

        f = gp.prior("f", X=X)

        sigma = pm.HalfNormal("sigma", sigma=2.0)
        nu = 1 + pm.Gamma("nu", alpha=2, beta=0.1)
        y_ = pm.StudentT("y", mu=f, lam=1.0 / sigma, nu=nu, observed=y)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
