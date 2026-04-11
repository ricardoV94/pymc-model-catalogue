"""
Model: Bayesian Proportional Hazards (Cox PH)
Source: pymc-examples/examples/survival_analysis/survival_analysis.ipynb, Section: "Bayesian proportional hazards model"
Authors: Austin Rochford, Fernando Irarrázaval
Description: Piecewise-constant proportional hazards model for mastectomy survival data.
    Approximates the Cox model using Poisson regression on discretized time intervals.
    Includes metastization status as a single covariate with a vague normal prior.
    Baseline hazard has independent Gamma priors per interval.

Changes from original:
- Inlined mastectomy data as numpy arrays (44 observations).
- Removed sampling and plotting code.

Benchmark results:
- Original:  logp = -4622.5486, grad norm = 2728.7151, 36.2 us/call (100000 evals)
- Frozen:    logp = -4622.5486, grad norm = 2728.7151, 36.0 us/call (100000 evals)
"""

import numpy as np
import pymc as pm
import pytensor.tensor as T

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

        beta = pm.Normal("beta", 0, sigma=1000)

        lambda_ = pm.Deterministic("lambda_", T.outer(T.exp(beta * metastasized), lambda0))
        mu = pm.Deterministic("mu", exposure * lambda_)

        obs = pm.Poisson("obs", mu, observed=death)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
