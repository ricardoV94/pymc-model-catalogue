"""
Model: Latent Mundlak Multilevel Model with Unobserved Group Confound
Source: pymc-examples/examples/statistical_rethinking_lectures/12-Multilevel_Models.ipynb, Section: "Full Luxury: Latent Mundlak Machine"
Authors: Dustin Stansbury
Description: Multilevel Bernoulli model with latent group-level confound. Jointly models
    individual outcome Y and individual covariate X, allowing recovery of the unobserved
    group-level confounder U. Uses non-centered parameterization for group intercepts.

Changes from original:
- Generated simulated data inline with fixed seed
- Removed sampling and plotting code

Benchmark results:
- Original:  logp = -647.5650, grad norm = 343.9091, 11.2 us/call (100000 evals)
- Frozen:    logp = -647.5650, grad norm = 343.9091, 10.1 us/call (100000 evals)
"""

import numpy as np
import pymc as pm
from scipy import stats


def build_model():
    # Generate simulated data (same as notebook)
    np.random.seed(12)

    N_GROUPS = 30
    N_IDS = 200
    ALPHA = -2
    BETA_ZY = -0.5
    BETA_XY = 1

    # Group-level data
    GROUP = np.random.choice(np.arange(N_GROUPS).astype(int), size=N_IDS, replace=True)
    Z = stats.norm.rvs(size=N_GROUPS)
    U = stats.norm(0, 1.5).rvs(size=N_GROUPS)

    # Individual-level data
    X = stats.norm.rvs(U[GROUP])

    def invlogit(x):
        return 1 / (1 + np.exp(-x))

    p = invlogit(ALPHA + BETA_XY * X + U[GROUP] + BETA_ZY * Z[GROUP])
    Y = stats.bernoulli.rvs(p=p)

    with pm.Model() as model:
        # Unobserved variable
        G = pm.Normal("u_X", 0, 1, shape=N_GROUPS)

        # X sub-model
        alpha_X = pm.Normal("alpha_X", 0, 1)
        beta_GX = pm.Exponential("beta_GX", 1)
        sigma_X = pm.Exponential("sigma_X", 1)

        mu_X = alpha_X + beta_GX * G[GROUP]
        X_ = pm.Normal("X", mu_X, sigma_X, observed=X)

        # Y sub-model
        tau = pm.Exponential("tau", 1)

        # Non-centered parameterization
        z = pm.Normal("z", 0, 1, size=N_GROUPS)
        alpha_bar = pm.Normal("alph_bar", 0, 1)
        alpha = alpha_bar + tau * z

        beta_XY = pm.Normal("beta_XY", 0, 1)
        beta_ZY = pm.Normal("beta_ZY", 0, 1)
        beta_GY = pm.Normal("beta_GY", 0, 1)

        # Y likelihood
        p_y = pm.math.invlogit(
            alpha[GROUP] + beta_XY * X_ + beta_ZY * Z[GROUP] + beta_GY * G[GROUP]
        )
        pm.Bernoulli("Y", p=p_y, observed=Y)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
