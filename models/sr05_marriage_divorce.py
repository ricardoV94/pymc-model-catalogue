"""
Model: Marriage and Age Effects on Divorce Rate
Source: pymc-examples/examples/statistical_rethinking_lectures/05-Elemental_Confounds.ipynb, Section: "Multivariate Analysis"
Authors: Dustin Stansbury
Description: Multiple regression of divorce rate on marriage rate and median age at marriage,
    demonstrating confound analysis. Both marriage rate and age predict divorce, but once
    conditioned on age, the marriage rate effect shrinks.

Changes from original:
- Loaded data inline instead of using utils.load_data
- Standardized data inline instead of using utils.standardize
- Removed sampling and plotting code

Benchmark results:
- Original:  logp = -73.0943, grad norm = 35.2268, 3.9 us/call (100000 evals)
- Frozen:    logp = -73.0943, grad norm = 35.2268, 3.6 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data = np.load(
        Path(__file__).parent / "data" / "sr_waffle_divorce.npz", allow_pickle=True
    )
    divorce = data["Divorce"]
    marriage = data["Marriage"]
    age = data["MedianAgeMarriage"]

    # Standardize
    def standardize(x):
        return (x - np.nanmean(x)) / np.nanstd(x)

    std_divorce = standardize(divorce)
    std_marriage = standardize(marriage)
    std_age = standardize(age)

    with pm.Model() as model:
        age_ = pm.Data("age", std_age, dims="obs_ids")
        marriage_rate = pm.Data("marriage_rate", std_marriage, dims="obs_ids")

        sigma = pm.Exponential("sigma", 1)
        alpha = pm.Normal("alpha", 0, 0.2)
        beta_age = pm.Normal("beta_age", 0, 1)
        beta_marriage = pm.Normal("beta_marriage", 0, 1)

        # Divorce process
        mu = alpha + beta_marriage * marriage_rate + beta_age * age_
        pm.Normal("divorce_rate", mu, sigma, observed=std_divorce, dims="obs_ids")

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
