"""
Model: Copula estimation step 2 — 2D Gaussian copula covariance
Source: pymc-examples/examples/howto/copula-estimation.ipynb, Section: "PyMC models for copula and marginal estimation"
Authors: Eric Ma, Benjamin T. Vincent
Description: Second of the two-step Bayesian copula estimation workflow. Given
    point estimates of the marginal distribution parameters from step 1, map the
    observations `(a, b)` back to multivariate-normal space via the marginal CDFs
    and probit, then infer the 2x2 Cholesky covariance (LKJ prior) of the Gaussian
    copula by fitting a zero-mean MvNormal likelihood on the transformed data.

Changes from original:
- Inlined synthetic data generation inside build_model()
- The notebook samples the step-1 marginal_model (copula_marginal.py) to obtain
  posterior-mean point estimates of `a_mu`, `a_sigma`, `b_scale`, then uses those
  to transform observation-space data into multivariate-normal space. We cannot
  run sampling inside build_model(), so we use the true marginal parameters
  (`a_mu=0`, `a_sigma=1`, `b_scale=2`) used to simulate the data — these are
  approximately what step 1 recovers — as the point estimates. This affects only
  the observed transformed `data` array, not the model structure.
- Removed sampling/plotting code
- Added ip capture + initval clearing boilerplate

Benchmark results:
- Original:  logp = -14348.7365, grad norm = 4680.1117, 177.4 us/call (13263 evals)
- Frozen:    logp = -14348.7365, grad norm = 4680.1117, 371.8 us/call (100000 evals)
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

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

    # Point estimates that would normally come from sampling the step-1
    # marginal_model (see `copula_marginal.py`). We substitute the true
    # data-generating parameters here.
    # The step-1 `b_scale` parameter is the Exponential scale (=1/lam). Data
    # were generated with `expon(scale=1/b_scale_const=0.5)`, so the step-1
    # posterior for b_scale should concentrate near 0.5.
    a_mu_pe = 0.0
    a_sigma_pe = 1.0
    b_scale_pe = 0.5

    def transform_data(a, b, a_mu, a_sigma, b_scale):
        # transformations from observation space -> uniform space
        __a = pt.exp(pm.logcdf(pm.Normal.dist(mu=a_mu, sigma=a_sigma), a))
        __b = pt.exp(pm.logcdf(pm.Exponential.dist(lam=1 / b_scale), b))
        # uniform space -> multivariate normal space
        _a = pm.math.probit(__a)
        _b = pm.math.probit(__b)
        # join into an Nx2 matrix
        data = pt.math.stack([_a, _b], axis=1).eval()
        return data

    data = transform_data(a, b, a_mu_pe, a_sigma_pe, b_scale_pe)

    coords = {
        "obs_id": np.arange(len(a)),
        "param": ["a", "b"],
        "param_bis": ["a", "b"],
    }
    with pm.Model(coords=coords) as model:
        # Prior on covariance of the multivariate normal
        chol, corr, stds = pm.LKJCholeskyCov(
            "chol",
            n=2,
            eta=2.0,
            sd_dist=pm.Exponential.dist(1.0),
            compute_corr=True,
        )
        cov = pm.Deterministic("cov", chol.dot(chol.T), dims=("param", "param_bis"))

        # Likelihood function
        pm.MvNormal("N", mu=0.0, cov=cov, observed=data, dims=("obs_id", "param"))

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
