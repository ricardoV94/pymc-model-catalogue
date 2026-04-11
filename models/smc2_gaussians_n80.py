"""
Model: SMC Two-Gaussian Mixture via Potential (n=80)
Source: pymc-examples/examples/samplers/SMC2_gaussians.ipynb, Section: "Kill your darlings"
Authors: Osvaldo Martin
Description: High-dimensional (n=80) version of the bimodal two-Gaussian target defined as
    a weighted logsumexp of two multivariate Gaussians (weights 0.1 / 0.9, means +/- 0.5,
    stdev 0.1), with a Uniform prior on [-2, 2]^80 and a pm.Potential log-likelihood.

Changes from original:
- Removed sampling and plotting code
- Inlined the two_gaussians helper inside build_model()

Benchmark results:
- Original:  logp = -1023.3318, grad norm = 339.8823, 6.5 us/call (100000 evals)
- Frozen:    logp = -1023.3318, grad norm = 339.8823, 6.5 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    n = 80

    mu1 = np.ones(n) * (1.0 / 2)
    mu2 = -mu1

    stdev = 0.1
    sigma = np.power(stdev, 2) * np.eye(n)
    isigma = np.linalg.inv(sigma)
    dsigma = np.linalg.det(sigma)

    w1 = 0.1  # one mode with 0.1 of the mass
    w2 = 1 - w1  # the other mode with 0.9 of the mass

    def two_gaussians(x):
        log_like1 = (
            -0.5 * n * pt.log(2 * np.pi)
            - 0.5 * pt.log(dsigma)
            - 0.5 * (x - mu1).T.dot(isigma).dot(x - mu1)
        )
        log_like2 = (
            -0.5 * n * pt.log(2 * np.pi)
            - 0.5 * pt.log(dsigma)
            - 0.5 * (x - mu2).T.dot(isigma).dot(x - mu2)
        )
        return pm.math.logsumexp([pt.log(w1) + log_like1, pt.log(w2) + log_like2])

    with pm.Model() as model:
        X = pm.Uniform(
            "X",
            shape=n,
            lower=-2.0 * np.ones_like(mu1),
            upper=2.0 * np.ones_like(mu1),
            initval=-1.0 * np.ones_like(mu1),
        )
        llk = pm.Potential("llk", two_gaussians(X))

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
