"""
Model: Discrete Choice - Multinomial Logit with Alternative-Specific Intercepts
Source: pymc-examples/examples/generalized_linear_models/GLM-discrete-choice_models.ipynb, Section: "Improved Model: Adding Alternative Specific Intercepts"
Authors: Nathaniel Forde
Description: Multinomial logit model for heating system choice with alternative-specific
    intercepts (one per non-pivot alternative) plus installation and operating cost predictors.

Changes from original:
- Loaded data from .npz file instead of CSV.
- Removed sampling and plotting code.

Benchmark results:
- Original:  logp = -1454.0078, grad norm = 99492.7836, 96.6 us/call (100000 evals)
- Frozen:    logp = -1454.0078, grad norm = 99492.7836, 97.1 us/call (100000 evals)
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
        beta_ic = pm.Normal("beta_ic", 0, 1)
        beta_oc = pm.Normal("beta_oc", 0, 1)
        alphas = pm.Normal("alpha", 0, 1, dims="alts_intercepts")

        ## Construct Utility matrix and Pivot using an intercept per alternative
        u0 = alphas[0] + beta_ic * ic_ec + beta_oc * oc_ec
        u1 = alphas[1] + beta_ic * ic_er + beta_oc * oc_er
        u2 = alphas[2] + beta_ic * ic_gc + beta_oc * oc_gc
        u3 = alphas[3] + beta_ic * ic_gr + beta_oc * oc_gr
        u4 = 0 + beta_ic * ic_hp + beta_oc * oc_hp
        s = pm.math.stack([u0, u1, u2, u3, u4]).T

        ## Apply Softmax Transform
        p_ = pm.Deterministic("p", pm.math.softmax(s, axis=1), dims=("obs", "alts_probs"))

        ## Likelihood
        pm.Categorical("y_cat", p=p_, observed=observed, dims="obs")

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
