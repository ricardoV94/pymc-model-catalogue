"""
Model: Bayesian Mediation Analysis
Source: pymc-examples/examples/causal_inference/mediation_analysis.ipynb, Section: "Define the PyMC model and conduct inference"
Authors: Benjamin T. Vincent
Description: Simple mediation model estimating direct effect (x->y), indirect effect (x->m->y),
    and total effect using two simultaneous regression equations.

Changes from original:
- pm.ConstantData -> pm.Data (API update)
- Removed sampling, plotting, and hypothesis testing code

Benchmark results:
- Original:  logp = -392.7452, grad norm = 374.4172, 5.2 us/call (100000 evals)
- Frozen:    logp = -392.7452, grad norm = 374.4172, 4.9 us/call (100000 evals)
"""

import numpy as np
import pymc as pm

def build_model():
    # Reproduce synthetic data generation faithfully
    seed = 42
    rng = np.random.default_rng(seed)

    N = 75
    a, b, cprime = 0.5, 0.6, 0.3
    im, iy, sigma_m, sigma_y = 2.0, 0.0, 0.5, 0.5
    x = rng.normal(loc=0, scale=1, size=N)
    m = im + rng.normal(loc=a * x, scale=sigma_m, size=N)
    y = iy + (cprime * x) + rng.normal(loc=b * m, scale=sigma_y, size=N)

    with pm.Model() as model:
        x_data = pm.Data("x", x, dims="obs_id")
        y_data = pm.Data("y", y, dims="obs_id")
        m_data = pm.Data("m", m, dims="obs_id")

        # intercept priors
        im_param = pm.Normal("im", mu=0, sigma=10)
        iy_param = pm.Normal("iy", mu=0, sigma=10)
        # slope priors
        a_param = pm.Normal("a", mu=0, sigma=10)
        b_param = pm.Normal("b", mu=0, sigma=10)
        cprime_param = pm.Normal("cprime", mu=0, sigma=10)
        # noise priors
        sigma_m_param = pm.HalfCauchy("σm", 1)
        sigma_y_param = pm.HalfCauchy("σy", 1)

        # likelihood
        pm.Normal(
            "m_likelihood",
            mu=im_param + a_param * x_data,
            sigma=sigma_m_param,
            observed=m,
            dims="obs_id",
        )
        pm.Normal(
            "y_likelihood",
            mu=iy_param + b_param * m_data + cprime_param * x_data,
            sigma=sigma_y_param,
            observed=y,
            dims="obs_id",
        )

        # calculate quantities of interest
        indirect_effect = pm.Deterministic("indirect effect", a_param * b_param)
        total_effect = pm.Deterministic("total effect", a_param * b_param + cprime_param)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
