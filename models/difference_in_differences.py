"""
Model: Difference in Differences
Source: pymc-examples/examples/causal_inference/difference_in_differences.ipynb, Section: "PyMC model"
Authors: Benjamin T. Vincent
Description: Bayesian difference-in-differences model for causal inference in quasi-experimental
    settings with pre/post treatment measures and control/treatment groups.

Changes from original:
- pm.MutableData -> pm.Data (API update)
- Removed sampling, plotting, and counterfactual inference code

Benchmark results:
- Original:  logp = -115.5978, grad norm = 141.5481, 3.7 us/call (100000 evals)
- Frozen:    logp = -115.5978, grad norm = 141.5481, 3.4 us/call (100000 evals)
"""

import numpy as np
import pymc as pm

def build_model():
    # Reproduce synthetic data generation faithfully
    RANDOM_SEED = 8927
    rng = np.random.default_rng(RANDOM_SEED)

    # true parameters
    control_intercept = 1
    treat_intercept_delta = 0.25
    trend = 1
    delta = 0.5
    intervention_time = 0.5

    def outcome(t, control_intercept, treat_intercept_delta, trend, delta, group, treated):
        return (
            control_intercept
            + (treat_intercept_delta * group)
            + (t * trend)
            + (delta * treated * group)
        )

    def is_treated(t, intervention_time, group):
        return (t > intervention_time) * group

    # Generate panel data
    group = np.array([0, 0, 1, 1] * 10)
    t = np.array([0.0, 1.0, 0.0, 1.0] * 10)
    unit = np.concatenate([[i] * 2 for i in range(20)])
    treated = is_treated(t, intervention_time, group)

    y = outcome(t, control_intercept, treat_intercept_delta, trend, delta, group, treated)
    y = y + rng.normal(0, 0.1, y.shape[0])

    with pm.Model() as model:
        # data
        t_data = pm.Data("t", t, dims="obs_idx")
        treated_data = pm.Data("treated", treated, dims="obs_idx")
        group_data = pm.Data("group", group, dims="obs_idx")
        # priors
        _control_intercept = pm.Normal("control_intercept", 0, 5)
        _treat_intercept_delta = pm.Normal("treat_intercept_delta", 0, 1)
        _trend = pm.Normal("trend", 0, 5)
        _delta = pm.Normal("Δ", 0, 1)
        sigma = pm.HalfNormal("sigma", 1)
        # expectation
        mu = pm.Deterministic(
            "mu",
            outcome(
                t_data,
                _control_intercept,
                _treat_intercept_delta,
                _trend,
                _delta,
                group_data,
                treated_data,
            ),
            dims="obs_idx",
        )
        # likelihood
        pm.Normal("obs", mu, sigma, observed=y, dims="obs_idx")

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
