"""
Model: Cox PH with Individual Frailty Terms (Employee Attrition)
Source: pymc-examples/examples/survival_analysis/frailty_models.ipynb, Section: "Fit Model with Shared Frailty terms by Individual"
Authors: Nathaniel Forde
Description: Cox proportional hazards model with individual Gamma frailty terms and
    gender-stratified baseline hazard. Each employee gets their own frailty
    multiplier. Uses Poisson likelihood on discretized monthly intervals with
    gender-stratified baseline hazards (separate for Male/Female).

Changes from original:
- Loaded pre-processed retention data from .npz file (3770 observations).
- Removed sampling and plotting code.
- pm.MutableData replaced with pm.Data.
- Inlined opt_params from pm.find_constrained_prior result.

Benchmark results:
- Original:  logp = -32827.8221, grad norm = 304143.5648, 802.4 us/call (17148 evals)
- Frozen:    logp = -32827.8221, grad norm = 304143.5648, 791.7 us/call (19939 evals)
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
    Male = data["Male"].astype(float)
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

    preds = [
        "sentiment",
        "intention",
        "Low",
        "Medium",
        "Finance",
        "Health",
        "Law",
        "Public/Government",
        "Sales/Marketing",
    ]

    X = np.column_stack([
        sentiment, intention, Low, Medium,
        Finance, Health, Law, Public_Government, Sales_Marketing,
    ])

    # Individual frailty: factor = range(len(retention_df))
    frailty_idx = np.arange(n_employees)
    frailty_labels = np.arange(n_employees)

    # Identify male/female subsets
    is_male = Male == 1.0
    male_idx = np.where(is_male)[0]
    female_idx = np.where(~is_male)[0]

    X_m = X[male_idx]
    X_f = X[female_idx]

    # opt_params from pm.find_constrained_prior(pm.Gamma, lower=0.80, upper=1.30, mass=0.90, ...)
    opt_params = {"alpha": 46.22819238464343, "beta": 44.910852755302585}

    coords = {
        "intervals": intervals,
        "preds": preds,
        "frailty_id": frailty_labels,
        "gender": ["Male", "Female"],
        "women": female_idx,
        "men": male_idx,
        "obs": range(n_employees),
    }

    with pm.Model(coords=coords) as model:
        X_data_m = pm.Data("X_data_m", X_m, dims=("men", "preds"))
        X_data_f = pm.Data("X_data_f", X_f, dims=("women", "preds"))
        lambda0 = pm.Gamma("lambda0", 0.01, 0.01, dims=("intervals", "gender"))
        sigma_frailty = pm.Normal("sigma_frailty", opt_params["alpha"], 1)
        mu_frailty = pm.Normal("mu_frailty", opt_params["beta"], 1)
        frailty = pm.Gamma("frailty", mu_frailty, sigma_frailty, dims="frailty_id")

        beta = pm.Normal("beta", 0, sigma=1, dims="preds")

        # Stratified baseline hazards
        lambda_m = pm.Deterministic(
            "lambda_m",
            pt.outer(pt.exp(pm.math.dot(beta, X_data_m.T)), lambda0[:, 0]),
            dims=("men", "intervals"),
        )
        lambda_f = pm.Deterministic(
            "lambda_f",
            pt.outer(pt.exp(pm.math.dot(beta, X_data_f.T)), lambda0[:, 1]),
            dims=("women", "intervals"),
        )
        lambda_ = pm.Deterministic(
            "lambda_",
            frailty[frailty_idx, None] * pt.concatenate([lambda_f, lambda_m], axis=0),
            dims=("obs", "intervals"),
        )

        mu = pm.Deterministic("mu", exposure * lambda_, dims=("obs", "intervals"))

        obs = pm.Poisson("outcome", mu, observed=quit_, dims=("obs", "intervals"))

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
