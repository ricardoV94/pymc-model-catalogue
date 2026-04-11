"""
Model: Interrupted Time Series Analysis
Source: pymc-examples/examples/causal_inference/interrupted_time_series.ipynb, Section: "Modelling"
Authors: Benjamin T. Vincent
Description: Simple Bayesian linear model for interrupted time series analysis, estimating
    a linear trend from pre-intervention data to forecast counterfactual outcomes.

Changes from original:
- pm.Data (was already pm.Data in original)
- Removed sampling, plotting, and counterfactual inference code

Benchmark results:
- Original:  logp = -376.4252, grad norm = 4817.7145, 3.5 us/call (100000 evals)
- Frozen:    logp = -376.4252, grad norm = 4817.7145, 3.4 us/call (100000 evals)
"""

import numpy as np
import pymc as pm

def build_model():
    # Reproduce synthetic data generation faithfully
    import pandas as pd
    from scipy.stats import norm

    treatment_time = "2017-01-01"
    beta0_true = 0
    beta1_true = 0.1
    dates = pd.date_range(
        start=pd.to_datetime("2010-01-01"), end=pd.to_datetime("2020-01-01"), freq="ME"
    )
    N = len(dates)

    def causal_effect(date_index):
        return (date_index > pd.to_datetime(treatment_time)) * 2

    time_arr = np.arange(N)
    y_arr = (
        beta0_true
        + beta1_true * time_arr
        + causal_effect(dates)
        + norm(0, 0.5).rvs(N, random_state=8927)
    )

    # Split into pre-intervention
    pre_mask = dates < treatment_time
    pre_time = time_arr[pre_mask]
    pre_y = y_arr[pre_mask]

    with pm.Model() as model:
        # observed predictors and outcome
        time = pm.Data("time", pre_time, dims="obs_id")
        # priors
        beta0 = pm.Normal("beta0", 0, 1)
        beta1 = pm.Normal("beta1", 0, 0.2)
        # the actual linear model
        mu = pm.Deterministic("mu", beta0 + (beta1 * time), dims="obs_id")
        sigma = pm.HalfNormal("sigma", 2)
        # likelihood
        pm.Normal("obs", mu=mu, sigma=sigma, observed=pre_y, dims="obs_id")

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
