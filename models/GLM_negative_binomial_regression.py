"""
Model: GLM Negative Binomial Regression with Interaction Terms
Source: pymc-examples/examples/generalized_linear_models/GLM-negative-binomial-regression.ipynb, Section: "Negative Binomial Regression"
Authors: Ian Ozsvald, Abhipsha Das, Benjamin Vincent
Description: Negative binomial regression predicting overdispersed sneeze counts from
    alcohol consumption and antihistamine usage, with interaction term and estimated
    dispersion parameter.

Changes from original:
- Inlined generated data as numpy arrays.
- Removed sampling and plotting code.

Benchmark results:
- Original:  logp = -46206.6603, grad norm = 54331.2072, 271.3 us/call (49993 evals)
- Frozen:    logp = -46206.6603, grad norm = 54331.2072, 229.3 us/call (60987 evals)
"""

import numpy as np
import pymc as pm
from scipy import stats
def build_model():
    RANDOM_SEED = 8927
    rng = np.random.default_rng(RANDOM_SEED)
    # Mean Poisson values (same as Poisson regression example)
    theta_noalcohol_meds = 1
    theta_alcohol_meds = 3
    theta_noalcohol_nomeds = 6
    theta_alcohol_nomeds = 36
    # Gamma shape parameter
    alpha_true = 10
    def get_nb_vals(mu, alpha, size):
        """Generate negative binomially distributed samples."""
        g = stats.gamma.rvs(alpha, scale=mu / alpha, size=size, random_state=rng)
        return stats.poisson.rvs(g, random_state=rng)
    # Create samples
    n = 1000
    nsneeze = np.concatenate((
        get_nb_vals(theta_noalcohol_meds, alpha_true, n),
        get_nb_vals(theta_alcohol_meds, alpha_true, n),
        get_nb_vals(theta_noalcohol_nomeds, alpha_true, n),
        get_nb_vals(theta_alcohol_nomeds, alpha_true, n),
    ))
    nomeds_data = np.concatenate((
        np.repeat(False, n),
        np.repeat(False, n),
        np.repeat(True, n),
        np.repeat(True, n),
    )).astype(float)
    alcohol_data = np.concatenate((
        np.repeat(False, n),
        np.repeat(True, n),
        np.repeat(False, n),
        np.repeat(True, n),
    )).astype(float)
    obs_idx = np.arange(4 * n)
    COORDS = {"regressor": ["nomeds", "alcohol", "nomeds:alcohol"], "obs_idx": obs_idx}

    with pm.Model(coords=COORDS) as m_sneeze_inter:
        a = pm.Normal("intercept", mu=0, sigma=5)
        b = pm.Normal("slopes", mu=0, sigma=1, dims="regressor")
        alpha = pm.Exponential("alpha", 0.5)

        M = pm.Data("nomeds", nomeds_data, dims="obs_idx")
        A = pm.Data("alcohol", alcohol_data, dims="obs_idx")
        S = pm.Data("nsneeze", nsneeze, dims="obs_idx")

        lam = pm.math.exp(a + b[0] * M + b[1] * A + b[2] * M * A)

        y = pm.NegativeBinomial("y", mu=lam, alpha=alpha, observed=S, dims="obs_idx")

    ip = m_sneeze_inter.initial_point()
    m_sneeze_inter.rvs_to_initial_values = {rv: None for rv in m_sneeze_inter.free_RVs}
    return m_sneeze_inter, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
