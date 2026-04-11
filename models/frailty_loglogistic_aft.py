"""
Model: Log-Logistic AFT Survival Regression (Employee Attrition)
Source: pymc-examples/examples/survival_analysis/frailty_models.ipynb, Section: "Accelerated Failure Time Models"
Authors: Nathaniel Forde
Description: Log-logistic accelerated failure time model for employee attrition data with
    multiple covariates. Uses log-transformed time with logistic likelihood.
    Censored observations handled via a logistic survival function potential.

Changes from original:
- Loaded pre-processed retention data from .npz file (3770 observations).
- Removed sampling, posterior predictive, and plotting code.
- pm.MutableData replaced with pm.Data.

Benchmark results:
- Original:  logp = -3182.3327, grad norm = 1733.2066, 108.6 us/call (100000 evals)
- Frozen:    logp = -3182.3327, grad norm = 1733.2066, 93.8 us/call (100000 evals)
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

    y = np.log(month)
    cens = left == 0.0

    intervals = np.arange(12)
    preds_list = [
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
        "preds": preds_list,
    }

    def logistic_sf(y, mu, s):
        return 1.0 - pm.math.sigmoid((y - mu) / s)

    with pm.Model(coords=coords, check_bounds=False) as model:
        X_data = pm.Data("X_data_obs", X)
        beta = pm.Normal("beta", 0.0, 1, dims="preds")
        mu = pm.Normal("mu", 0, 1)

        s = pm.HalfNormal("s", 5.0)
        eta = pm.Deterministic("eta", pm.math.dot(beta, X_data.T))
        reg = pm.Deterministic("reg", mu + eta)
        y_obs = pm.Logistic("y_obs", mu=reg[~cens], s=s, observed=y[~cens])
        y_cens = pm.Potential("y_cens", logistic_sf(y[cens], reg[cens], s=s))

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
