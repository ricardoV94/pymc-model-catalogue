"""
Model: Copula estimation step 1 — marginal distribution parameters
Source: pymc-examples/examples/howto/copula-estimation.ipynb, Section: "PyMC models for copula and marginal estimation"
Authors: Eric Ma, Benjamin T. Vincent
Description: First of the two-step Bayesian copula estimation workflow. Estimates
    the parameters of the marginal distributions (Normal for `a`, Exponential for
    `b`) from bivariate samples drawn via a 2D Gaussian copula with correlation
    rho=0.9 and marginals N(0,1) / Exp(scale=0.5).

Changes from original:
- Inlined synthetic data generation inside build_model()
- Removed sampling/plotting code
- Added ip capture + initval clearing boilerplate

Benchmark results:
- Original:  logp = -13441.8748, grad norm = 5267.8455, 5.8 us/call (100000 evals)
- Frozen:    logp = -13441.8748, grad norm = 5267.8455, 5.5 us/call (100000 evals)
"""

import numpy as np
import pymc as pm

from scipy.stats import expon, multivariate_normal, norm


def build_model():
    SEED = 43
    rng = np.random.default_rng(SEED)

    # define properties of our copula
    b_scale = 2
    θ = {"a_dist": norm(), "b_dist": expon(scale=1 / b_scale), "rho": 0.9}

    n_samples = 5000

    # draw random samples in multivariate normal space
    mu = [0, 0]
    cov = [[1, θ["rho"]], [θ["rho"], 1]]
    x = multivariate_normal(mu, cov).rvs(n_samples, random_state=rng)
    a_norm = x[:, 0]
    b_norm = x[:, 1]

    # make marginals uniform
    a_unif = norm(loc=0, scale=1).cdf(a_norm)
    b_unif = norm(loc=0, scale=1).cdf(b_norm)

    # transform to observation space
    a = θ["a_dist"].ppf(a_unif)
    b = θ["b_dist"].ppf(b_unif)

    coords = {"obs_id": np.arange(len(a))}
    with pm.Model(coords=coords) as model:
        # marginal estimation
        a_mu = pm.Normal("a_mu", mu=0, sigma=1)
        a_sigma = pm.Exponential("a_sigma", lam=0.5)
        pm.Normal("a", mu=a_mu, sigma=a_sigma, observed=a, dims="obs_id")

        b_scale_rv = pm.Exponential("b_scale", lam=0.5)
        pm.Exponential("b", lam=1 / b_scale_rv, observed=b, dims="obs_id")

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
