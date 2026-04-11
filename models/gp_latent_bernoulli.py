"""
Model: GP Latent with Bernoulli likelihood (GP classification)
Source: pymc-examples/examples/gaussian_processes/GP-Latent.ipynb, Section: "Example 2: GP classification"
Authors: Bill Engels, Chris Fonnesbeck
Description: Latent GP with ExpQuad covariance and Bernoulli likelihood via
    logit link for binary classification on 300 data points.

Changes from original:
- Removed sampling, plotting, conditional predictions
- Inlined synthetic data generation with fixed seed
- Added ip capture and initval clearing

Benchmark results:
- Original:  logp = -484.7566, 1243.4 us/call (12448 evals)
- Frozen:    logp = -484.7566, 1264.3 us/call (12210 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    RANDOM_SEED = 8888
    rng = np.random.default_rng(RANDOM_SEED)

    n = 300
    x = np.linspace(0, 10, n)

    ell_true = 0.5
    eta_true = 1.0
    cov_func = eta_true**2 * pm.gp.cov.ExpQuad(1, ell_true)
    K = cov_func(x[:, None]).eval()

    mean = np.zeros(n)
    f_true = pm.draw(
        pm.MvNormal.dist(mu=mean, cov=K, method="svd"), 1, random_seed=rng
    )

    y = pm.Bernoulli.dist(p=pm.math.invlogit(f_true)).eval()

    with pm.Model() as model:
        ell = pm.InverseGamma("ell", mu=1.0, sigma=0.5)
        eta = pm.Exponential("eta", lam=1.0)
        cov = eta**2 * pm.gp.cov.ExpQuad(1, ell)

        gp = pm.gp.Latent(cov_func=cov)
        f = gp.prior("f", X=x[:, None])

        # logit link and Bernoulli likelihood
        p = pm.Deterministic("p", pm.math.invlogit(f))
        y_ = pm.Bernoulli("y", p=p, observed=y)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model, discrete=True)
