"""
Model: Robust Regression with Student-T Likelihood
Source: pymc-examples/examples/statistical_rethinking_lectures/07-Fitting_Over_&_Under.ipynb, Section: "Robust Regression"
Authors: Dustin Stansbury
Description: Regression of divorce rate on median age at marriage using a Student-T likelihood
    (nu=2) instead of Gaussian, providing robustness to outliers like Idaho and Maine.

Changes from original:
- Loaded and standardized data inline instead of using utils
- Removed sampling, LOO-CV, and plotting code

Benchmark results:
- Original:  logp = -79.0503, grad norm = 25.1150, 4.7 us/call (100000 evals)
- Frozen:    logp = -79.0503, grad norm = 25.1150, 4.0 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data = np.load(
        Path(__file__).parent / "data" / "sr_waffle_divorce.npz", allow_pickle=True
    )
    divorce = data["Divorce"]
    age = data["MedianAgeMarriage"]

    # Standardize
    def standardize(x):
        return (x - np.nanmean(x)) / np.nanstd(x)

    std_divorce = standardize(divorce)
    std_age = standardize(age)

    with pm.Model() as model:
        # Priors
        sigma = pm.Exponential("sigma", 1)
        alpha = pm.Normal("alpha", 0, 0.5)
        beta_A = pm.Normal("beta_A", 0, 0.5)

        # Likelihood - Student-T for robustness
        mu = alpha + beta_A * std_age
        pm.StudentT("divorce", nu=2, mu=mu, sigma=sigma, observed=std_divorce)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
