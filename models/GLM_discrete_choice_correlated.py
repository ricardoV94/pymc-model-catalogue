"""
Model: Discrete Choice - Multinomial Logit with Correlated Intercepts (LKJ)
Source: pymc-examples/examples/generalized_linear_models/GLM-discrete-choice_models.ipynb, Section: "Experimental Model: Adding Correlation Structure"
Authors: Nathaniel Forde
Description: Multinomial logit model for heating system choice with correlated
    alternative-specific intercepts via LKJ Cholesky prior, plus installation
    and operating cost predictors.

Changes from original:
- Loaded data from .npz file instead of CSV.
- pm.MutableData -> pm.Data.
- Removed sampling, plotting, and prediction code.

Benchmark results:
- Original:  logp = -1459.1913, grad norm = 99492.8688, 101.9 us/call (100000 evals)
- Frozen:    logp = -1459.1913, grad norm = 99492.8688, 100.6 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data = np.load(Path(__file__).parent / "data" / "heating_data.npz")
    observed = data["observed"]
    ic_ec = data["ic_ec"]
    ic_er = data["ic_er"]
    ic_gc = data["ic_gc"]
    ic_gr = data["ic_gr"]
    ic_hp = data["ic_hp"]
    oc_ec = data["oc_ec"]
    oc_er = data["oc_er"]
    oc_gc = data["oc_gc"]
    oc_gr = data["oc_gr"]
    oc_hp = data["oc_hp"]

    N = len(observed)
    coords = {
        "alts_intercepts": ["ec", "er", "gc", "gr"],
        "alts_probs": ["ec", "er", "gc", "gr", "hp"],
        "obs": range(N),
    }

    with pm.Model(coords=coords) as model:
        ## Add data
        ic_ec_data = pm.Data("ic_ec", ic_ec)
        oc_ec_data = pm.Data("oc_ec", oc_ec)
        ic_er_data = pm.Data("ic_er", ic_er)
        oc_er_data = pm.Data("oc_er", oc_er)

        beta_ic = pm.Normal("beta_ic", 0, 1)
        beta_oc = pm.Normal("beta_oc", 0, 1)
        chol, corr, stds = pm.LKJCholeskyCov(
            "chol", n=5, eta=2.0, sd_dist=pm.Exponential.dist(1.0, shape=5)
        )
        alphas = pm.MvNormal("alpha", mu=0, chol=chol, dims="alts_probs")

        u0 = alphas[0] + beta_ic * ic_ec_data + beta_oc * oc_ec_data
        u1 = alphas[1] + beta_ic * ic_er_data + beta_oc * oc_er_data
        u2 = alphas[2] + beta_ic * ic_gc + beta_oc * oc_gc
        u3 = alphas[3] + beta_ic * ic_gr + beta_oc * oc_gr
        u4 = alphas[4] + beta_ic * ic_hp + beta_oc * oc_hp
        s = pm.math.stack([u0, u1, u2, u3, u4]).T

        p_ = pm.Deterministic("p", pm.math.softmax(s, axis=1), dims=("obs", "alts_probs"))
        pm.Categorical("y_cat", p=p_, observed=observed, dims="obs")

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
