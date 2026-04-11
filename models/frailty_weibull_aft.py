"""
Model: Weibull AFT Survival Regression (Employee Attrition)
Source: pymc-examples/examples/survival_analysis/frailty_models.ipynb, Section: "Accelerated Failure Time Models"
Authors: Nathaniel Forde
Description: Weibull accelerated failure time model for employee attrition data with
    multiple covariates. Censored observations handled via a Weibull log complementary
    CDF potential. Regression coefficients enter through the Weibull beta parameter.

Changes from original:
- Loaded pre-processed retention data from .npz file (3770 observations).
- Removed sampling, posterior predictive, and plotting code.
- pm.MutableData replaced with pm.Data.

Benchmark results:
- Original:  logp = -581091077.3380, grad norm = 8778187768.8889, 211.9 us/call (68339 evals)
- Frozen:    logp = -581091077.3380, grad norm = 8778187768.8889, 189.5 us/call (79782 evals)
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

    y = month
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

    def weibull_lccdf(x, alpha, beta):
        """Log complementary cdf of Weibull distribution."""
        return -((x / beta) ** alpha)

    with pm.Model(coords=coords, check_bounds=False) as model:
        X_data = pm.Data("X_data_obs", X)
        beta = pm.Normal("beta", 0.0, 1, dims="preds")
        mu = pm.Normal("mu", 0, 1)

        s = pm.HalfNormal("s", 5.0)
        eta = pm.Deterministic("eta", pm.math.dot(beta, X_data.T))
        reg = pm.Deterministic("reg", pt.exp(-(mu + eta) / s))
        y_obs = pm.Weibull("y_obs", beta=reg[~cens], alpha=s, observed=y[~cens])
        y_cens = pm.Potential("y_cens", weibull_lccdf(y[cens], alpha=s, beta=reg[cens]))

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
