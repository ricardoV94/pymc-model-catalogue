"""
Model: Bayesian Proportional Hazards with Time-Varying Coefficients
Source: pymc-examples/examples/survival_analysis/survival_analysis.ipynb, Section: "Time varying effects"
Authors: Austin Rochford, Fernando Irarrázaval
Description: Piecewise-constant proportional hazards model with time-varying regression
    coefficients modeled as a Gaussian random walk. Allows the effect of metastization
    on hazard to vary across time intervals. Uses Poisson likelihood approximation.

Changes from original:
- Inlined mastectomy data as numpy arrays (44 observations).
- Removed sampling and plotting code.

Benchmark results:
- Original:  logp = -4684.5612, grad norm = 691.7702, 62.9 us/call (100000 evals)
- Frozen:    logp = -4684.5612, grad norm = 691.7702, 61.3 us/call (100000 evals)
"""

import numpy as np
import pymc as pm
import pytensor.tensor as T

from pymc.distributions.timeseries import GaussianRandomWalk

def build_model():
    # Mastectomy data
    time = np.array(
        [23, 47, 69, 70, 100, 101, 148, 181, 198, 208, 212, 224,
         5, 8, 10, 13, 18, 24, 26, 26, 31, 35, 40, 41, 48, 50,
         59, 61, 68, 71, 76, 105, 107, 109, 113, 116, 118, 143,
         145, 162, 188, 212, 217, 225], dtype=float)
    event = np.array(
        [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0,
         1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    metastasized = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int64)

    n_patients = len(time)
    patients = np.arange(n_patients)

    interval_length = 3
    interval_bounds = np.arange(0, time.max() + interval_length + 1, interval_length)
    n_intervals = interval_bounds.size - 1
    intervals = np.arange(n_intervals)

    last_period = np.floor((time - 0.01) / interval_length).astype(int)

    death = np.zeros((n_patients, n_intervals))
    death[patients, last_period] = event

    exposure = np.greater_equal.outer(time, interval_bounds[:-1]) * interval_length
    exposure[patients, last_period] = time - interval_bounds[last_period]

    coords = {"intervals": intervals}

    with pm.Model(coords=coords) as model:
        lambda0 = pm.Gamma("lambda0", 0.01, 0.01, dims="intervals")
        beta = GaussianRandomWalk(
            "beta", init_dist=pm.Normal.dist(), sigma=1.0, dims="intervals"
        )

        lambda_ = pm.Deterministic(
            "h", lambda0 * T.exp(T.outer(T.constant(metastasized), beta))
        )
        mu = pm.Deterministic("mu", exposure * lambda_)

        obs = pm.Poisson("obs", mu, observed=death)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
