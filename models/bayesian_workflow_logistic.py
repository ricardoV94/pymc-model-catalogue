"""
Model: Logistic Growth Model for COVID-19 (Germany)
Source: pymc-examples/examples/case_studies/bayesian_workflow.ipynb, Section: "Step 7: Improve Model - Logistic Growth"
Authors: Thomas Wiecki, Chris Fonnesbeck
Description: Logistic growth model for COVID-19 confirmed cases in Germany, with a
    NegativeBinomial likelihood to handle overdispersion. Models carrying capacity,
    initial intercept, and growth rate.

Changes from original:
- Inlined approximate Germany COVID-19 data (March-July 2020) as a logistic curve
  with noise, since original loads dynamically from GitHub
- Removed prior/posterior predictive checks, forecasting, and plotting

Benchmark results:
- Original:  logp = -4067.6780, grad norm = 3047.7921, 16.1 us/call (100000 evals)
- Frozen:    logp = -4067.6780, grad norm = 3047.7921, 14.4 us/call (100000 evals)
"""

import numpy as np
import pymc as pm


def build_model():
    # Approximate Germany COVID-19 data: days since 100 cases (Mar 1 - Jul 31, 2020)
    # Generated from a logistic curve fit to approximate real data
    rng = np.random.default_rng(8451997)
    K_true = 200000  # approximate carrying capacity
    r_true = 0.15  # approximate growth rate
    a0_true = 130  # approximate intercept

    t = np.arange(153, dtype=np.float64)
    A_true = K_true / a0_true - 1
    growth_true = K_true / (1 + A_true * np.exp(-r_true * t))
    # Add NegativeBinomial-like noise
    confirmed = rng.negative_binomial(
        n=6, p=6 / (6 + growth_true)
    ).astype(np.float64)
    # Ensure monotonicity (cumulative cases)
    confirmed = np.maximum.accumulate(confirmed)
    # Ensure starts near 100
    confirmed = np.clip(confirmed, 100, None)

    with pm.Model() as model:
        t_data = pm.Data("t", t)
        confirmed_data = pm.Data("confirmed", confirmed)

        # Intercept
        a0 = pm.HalfNormal("a0", sigma=25)
        intercept = pm.Deterministic("intercept", a0 + 100)

        # Slope
        b = pm.HalfNormal("b", sigma=0.2)

        carrying_capacity = pm.Uniform(
            "carrying_capacity", lower=1_000, upper=80_000_000
        )
        # Transform carrying_capacity to a
        a = carrying_capacity / intercept - 1

        # Logistic growth
        growth = carrying_capacity / (1 + a * pm.math.exp(-b * t_data))

        # Likelihood
        pm.NegativeBinomial(
            "obs",
            growth,
            alpha=pm.Gamma("alpha", mu=6, sigma=1),
            observed=confirmed_data,
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
