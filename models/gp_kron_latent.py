"""
Model: Kronecker Structured GP (Latent)
Source: pymc-examples/examples/gaussian_processes/GP-Kron.ipynb, Section: "LatentKron"
Authors: Bill Engels, Raul-ing Average, Christopher Krapu, Danh Phan, Alex Andorra
Description: GP model exploiting Kronecker structure with LatentKron, using
    separable Matern52 x Cosine covariance on a 50x30 grid with Normal likelihood.

Changes from original:
- Removed sampling, plotting
- Inlined synthetic data generation with fixed seed
- Added ip capture and initval clearing

Benchmark results:
- Original:  logp = -5463.2193, grad norm = 6093.9115, 180.1 us/call (51437 evals)
- Frozen:    logp = -5463.2193, grad norm = 6093.9115, 182.6 us/call (66382 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    seed = sum(map(ord, "gpkron"))
    rng = np.random.default_rng(seed)

    n1, n2 = (50, 30)
    x1 = np.linspace(0, 5, n1)
    x2 = np.linspace(0, 3, n2)

    X = pm.math.cartesian(x1[:, None], x2[:, None])

    l1_true = 0.8
    l2_true = 1.0
    eta_true = 1.0

    cov = (
        eta_true**2
        * pm.gp.cov.Matern52(2, l1_true, active_dims=[0])
        * pm.gp.cov.Cosine(2, ls=l2_true, active_dims=[1])
    )

    K = cov(X).eval()
    f_true = rng.multivariate_normal(np.zeros(X.shape[0]), K, 1).flatten()

    sigma_true = 0.5
    y = f_true + sigma_true * rng.standard_normal(X.shape[0])

    Xs = [x1[:, None], x2[:, None]]

    with pm.Model() as model:
        ls1 = pm.TruncatedNormal("ls1", lower=0.5, upper=1.5, mu=1, sigma=0.5)
        ls2 = pm.TruncatedNormal("ls2", lower=0.5, upper=1.5, mu=1, sigma=0.5)
        eta = pm.HalfNormal("eta", sigma=0.5)

        cov_x1 = pm.gp.cov.Matern52(1, ls=ls1)
        cov_x2 = eta**2 * pm.gp.cov.Cosine(1, ls=ls2)

        gp = pm.gp.LatentKron(cov_funcs=[cov_x1, cov_x2])

        f = gp.prior("f", Xs=Xs)

        sigma = pm.HalfNormal("sigma", sigma=0.5)

        y_ = pm.Normal("y_", mu=f, sigma=sigma, observed=y)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
