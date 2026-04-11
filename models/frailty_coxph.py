"""
Model: Cox Proportional Hazards (Employee Attrition)
Source: pymc-examples/examples/survival_analysis/frailty_models.ipynb, Section: "Fit Basic Cox Model with Fixed Effects"
Authors: Nathaniel Forde
Description: Piecewise-constant Cox proportional hazards model for employee attrition data.
    Uses Poisson regression on discretized monthly time intervals with multiple
    covariates (sentiment, intention, gender, level, field dummies). Baseline hazard
    has independent Gamma priors per interval.

Changes from original:
- Loaded pre-processed retention data from .npz file (3770 observations).
- Removed sampling, model comparison, and plotting code.
- pm.MutableData replaced with pm.Data.
- Extracted the version with the broader predictor set (preds2 including intention).

Benchmark results:
- Original:  logp = -37499.0558, grad norm = 314488.4993, 457.9 us/call (34587 evals)
- Frozen:    logp = -37499.0558, grad norm = 314488.4993, 424.5 us/call (26116 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm
import pytensor.tensor as pt

def build_model():
    data = np.load(
        Path(__file__).parent / "data" / "time_to_attrition.npz", allow_pickle=True
    )

    sentiment = data["sentiment"].astype(float)
    intention = data["intention"].astype(float)
    left = data["left"]
    month = data["month"].astype(float)
    Male = data["Male"]
    Low = data["Low"]
    Medium = data["Medium"]
    Finance = data["Finance"]
    Health = data["Health"]
    Law = data["Law"]
    Public_Government = data["Public_Government"]
    Sales_Marketing = data["Sales_Marketing"]

    intervals = np.arange(12)
    n_employees = len(month)
    n_intervals = len(intervals)
    last_period = np.floor((month - 0.01) / 1).astype(int)
    employees = np.arange(n_employees)

    quit_ = np.zeros((n_employees, n_intervals))
    quit_[employees, last_period] = left

    exposure = np.greater_equal.outer(month, intervals) * 1.0
    exposure[employees, last_period] = month - intervals[last_period]

    preds2 = [
        "sentiment",
        "intention",
        "Male",
        "Low",
        "Medium",
        "Finance",
        "Health",
        "Law",
        "Public/Government",
        "Sales/Marketing",
    ]

    X = np.column_stack([
        sentiment, intention, Male, Low, Medium,
        Finance, Health, Law, Public_Government, Sales_Marketing,
    ])

    coords = {
        "intervals": intervals,
        "preds": preds2,
        "individuals": range(n_employees),
    }

    with pm.Model(coords=coords) as model:
        X_data = pm.Data("X_data_obs", X, dims=("individuals", "preds"))
        lambda0 = pm.Gamma("lambda0", 0.01, 0.01, dims="intervals")

        beta = pm.Normal("beta", 0, sigma=1, dims="preds")
        lambda_ = pm.Deterministic(
            "lambda_",
            pt.outer(pt.exp(pm.math.dot(beta, X_data.T)), lambda0),
            dims=("individuals", "intervals"),
        )

        mu = pm.Deterministic(
            "mu", exposure * lambda_, dims=("individuals", "intervals")
        )

        obs = pm.Poisson("obs", mu, observed=quit_, dims=("individuals", "intervals"))

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
