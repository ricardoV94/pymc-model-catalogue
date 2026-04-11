"""
Model: Heteroskedastic GP with Coregionalization (LMC sparse latent)
Source: pymc-examples/examples/gaussian_processes/GP-Heteroskedastic.ipynb, Section: "Coregionalized Sparse Latent"
Authors: John Goertz
Description: Models heteroskedastic noise using a Linear Model of Coregionalization
    with sparse inducing points. Two output GPs (mean and log-variance) share a
    common latent process via a Coregion kernel with Kronecker product structure.

Changes from original:
- Removed sampling, plotting, conditional predictions
- Inlined synthetic data generation with fixed seed
- Added ip capture and initval clearing

Benchmark results:
- Original:  logp = -317.0129, grad norm = 400.6963, 313.2 us/call (34016 evals)
- Frozen:    logp = -317.0129, grad norm = 400.6963, 321.3 us/call (41604 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    class SparseLatent:
        def __init__(self, cov_func):
            self.cov = cov_func

        def prior(self, name, X, Xu):
            Kuu = self.cov(Xu)
            self.L = pt.linalg.cholesky(pm.gp.util.stabilize(Kuu))

            self.v = pm.Normal(f"u_rotated_{name}", mu=0.0, sigma=1.0, shape=len(Xu))
            self.u = pm.Deterministic(f"u_{name}", pt.dot(self.L, self.v))

            Kfu = self.cov(X, Xu)
            self.Kuiu = pt.linalg.solve_triangular(
                self.L.T,
                pt.linalg.solve_triangular(self.L, self.u, lower=True),
                lower=False,
            )
            self.mu = pm.Deterministic(f"mu_{name}", pt.dot(Kfu, self.Kuiu))
            return self.mu

    rng = np.random.default_rng(2020)

    def signal(x):
        return x / 2 + np.sin(2 * np.pi * x) / 5

    def noise(y):
        return np.exp(y) / 20

    X = np.linspace(0.1, 1, 20)[:, None]
    X = np.vstack([X, X + 2])
    X_ = X.flatten()
    y = signal(X_)
    sigma_fun = noise(y)

    y_err = rng.lognormal(np.log(sigma_fun), 0.1)
    y_obs = rng.normal(y, y_err, size=(5, len(y)))
    y_obs_ = y_obs.T.flatten()
    X_obs = np.tile(X.T, (5, 1)).T.reshape(-1, 1)

    from scipy.spatial.distance import pdist

    distances = pdist(X_[:, None])
    distinct = distances != 0
    ell_l = distances[distinct].min() if sum(distinct) > 0 else 0.1
    ell_u = distances[distinct].max() if sum(distinct) > 0 else 1
    ell_sigma = max(0.1, (ell_u - ell_l) / 6)
    ell_mu = ell_l + 3 * ell_sigma

    # Inducing points by downsampling
    Xu = X[1::2]

    def add_coreg_idx(x):
        return np.hstack(
            [np.tile(x, (2, 1)), np.vstack([np.zeros(x.shape), np.ones(x.shape)])]
        )

    Xu_c, X_obs_c = (add_coreg_idx(x) for x in [Xu, X_obs])

    with pm.Model() as model:
        ell = pm.InverseGamma("ell", mu=ell_mu, sigma=ell_sigma)
        eta = pm.Gamma("eta", alpha=2, beta=1)
        EQcov = eta**2 * pm.gp.cov.ExpQuad(input_dim=1, active_dims=[0], ls=ell)

        D_out = 2  # two output dimensions, mean and variance
        rank = 2  # two basis GPs
        W = pm.Normal(
            "W",
            mu=0,
            sigma=3,
            shape=(D_out, rank),
            initval=np.full([D_out, rank], 0.1),
        )
        kappa = pm.Gamma("kappa", alpha=1.5, beta=1, shape=D_out)
        coreg = pm.gp.cov.Coregion(input_dim=1, active_dims=[0], kappa=kappa, W=W)

        cov = pm.gp.cov.Kron([EQcov, coreg])

        gp_LMC = SparseLatent(cov)
        LMC_f = gp_LMC.prior("LMC", X_obs_c, Xu_c)

        mu_f = LMC_f[: len(y_obs_)]
        lg_sigma_f = LMC_f[len(y_obs_) :]
        sigma_f = pm.Deterministic("sigma_f", pm.math.exp(lg_sigma_f))

        lik_htsc = pm.Normal("lik_htsc", mu=mu_f, sigma=sigma_f, observed=y_obs_)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
