"""
Model: Multivariate Gaussian Random Walk regression on correlated time series
Source: pymc-examples/examples/time_series/MvGaussianRandomWalk_demo.ipynb, Section: "Model"
Authors: Lorenzo Itoniazzi, Chris Fonnesbeck
Description: Bayesian regression of a 3-dimensional time series against two multivariate Gaussian random walks (intercept and slope) with LKJ Cholesky covariance priors.

Changes from original:
- Data is generated inline (small) instead of from an external dataset
- pytensor.shared wrappers replaced with plain numpy arrays passed directly to the model
- Helper Scaler class inlined inside build_model
- Removed sampling/plotting code

Benchmark results:
- Original:  logp = -1327.0209, grad norm = 370.9023, 36.8 us/call (100000 evals)
- Frozen:    logp = -1327.0209, grad norm = 370.9023, 45.7 us/call (100000 evals)
"""

import numpy as np
import pymc as pm

from scipy.linalg import cholesky


def build_model():
    rng = np.random.default_rng(8927)

    D = 3  # Dimension of random walks
    N = 300  # Number of steps
    sections = 5  # Number of sections
    period = N / sections  # Number of steps in each section

    Sigma_alpha = rng.standard_normal((D, D))
    Sigma_alpha = Sigma_alpha.T.dot(Sigma_alpha)
    L_alpha = cholesky(Sigma_alpha, lower=True)

    Sigma_beta = rng.standard_normal((D, D))
    Sigma_beta = Sigma_beta.T.dot(Sigma_beta)
    L_beta = cholesky(Sigma_beta, lower=True)

    # Gaussian random walks:
    alpha_np = np.cumsum(L_alpha.dot(rng.standard_normal((D, sections))), axis=1).T
    beta_np = np.cumsum(L_beta.dot(rng.standard_normal((D, sections))), axis=1).T
    t = np.arange(N)[:, None] / N
    alpha_np = np.repeat(alpha_np, period, axis=0)
    beta_np = np.repeat(beta_np, period, axis=0)
    sigma_noise = 0.1
    y = alpha_np + beta_np * t + sigma_noise * rng.standard_normal((N, 1))

    class Scaler:
        def __init__(self):
            self.mean_ = None
            self.std_ = None

        def transform(self, x):
            return (x - self.mean_) / self.std_

        def fit_transform(self, x):
            self.mean_ = x.mean(axis=0)
            self.std_ = x.std(axis=0)
            return self.transform(x)

        def inverse_transform(self, x):
            return x * self.std_ + self.mean_

    N_obs, D_obs = y.shape
    y_scaler = Scaler()
    t_scaler = Scaler()
    y_s = y_scaler.fit_transform(y)
    t_s = t_scaler.fit_transform(t)
    t_section = np.repeat(np.arange(sections), N_obs / sections)

    t_t = np.repeat(t_s, D_obs, axis=1)
    y_t = y_s
    t_section_t = t_section

    coords = {"y_": ["y_0", "y_1", "y_2"], "steps": np.arange(N_obs)}
    with pm.Model(coords=coords) as model:
        # Hyperpriors on Cholesky matrices
        chol_alpha, *_ = pm.LKJCholeskyCov(
            "chol_cov_alpha",
            n=D_obs,
            eta=2,
            sd_dist=pm.HalfCauchy.dist(2.5),
            compute_corr=True,
        )
        chol_beta, *_ = pm.LKJCholeskyCov(
            "chol_cov_beta",
            n=D_obs,
            eta=2,
            sd_dist=pm.HalfCauchy.dist(2.5),
            compute_corr=True,
        )

        # Priors on Gaussian random walks
        alpha = pm.MvGaussianRandomWalk(
            "alpha", mu=np.zeros(D_obs), chol=chol_alpha, shape=(sections, D_obs)
        )
        beta = pm.MvGaussianRandomWalk(
            "beta", mu=np.zeros(D_obs), chol=chol_beta, shape=(sections, D_obs)
        )

        # Deterministic construction of the correlated random walk
        alpha_r = alpha[t_section_t]
        beta_r = beta[t_section_t]
        regression = alpha_r + beta_r * t_t

        # Prior on noise
        sigma = pm.HalfNormal("sigma", 1.0)

        # Likelihood
        likelihood = pm.Normal(
            "y", mu=regression, sigma=sigma, observed=y_t, dims=("steps", "y_")
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
