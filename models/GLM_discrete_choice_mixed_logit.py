"""
Model: Discrete Choice - Mixed Logit with Individual Taste Parameters
Source: pymc-examples/examples/generalized_linear_models/GLM-discrete-choice_models.ipynb, Section: "Choosing Crackers over Repeated Choices: Mixed Logit Model"
Authors: Nathaniel Forde
Description: Mixed multinomial logit model for cracker brand choice with hierarchical
    individual-level taste parameters, display, feature, and price predictors.

Changes from original:
- Loaded data from .npz file instead of CSV.
- Removed sampling and plotting code.

Benchmark results:
- Original:  logp = -3628.8738, grad norm = 1287.7057, 302.4 us/call (48033 evals)
- Frozen:    logp = -3628.8738, grad norm = 1287.7057, 312.9 us/call (50357 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data = np.load(Path(__file__).parent / "data" / "cracker_choice.npz")
    observed = data["observed"]
    person_indx = data["person_indx"]
    n_individuals = int(data["n_individuals"])
    disp_sunshine = data["disp_sunshine"]
    disp_keebler = data["disp_keebler"]
    disp_nabisco = data["disp_nabisco"]
    disp_private = data["disp_private"]
    feat_sunshine = data["feat_sunshine"]
    feat_keebler = data["feat_keebler"]
    feat_nabisco = data["feat_nabisco"]
    feat_private = data["feat_private"]
    price_sunshine = data["price_sunshine"]
    price_keebler = data["price_keebler"]
    price_nabisco = data["price_nabisco"]
    price_private = data["price_private"]

    N = len(observed)
    coords = {
        "alts_intercepts": ["sunshine", "keebler", "nabisco", "private"],
        "alts_probs": ["sunshine", "keebler", "nabisco", "private"],
        "individuals": np.arange(n_individuals),
        "obs": range(N),
    }

    with pm.Model(coords=coords) as model:
        beta_feat = pm.Normal("beta_feat", 0, 1)
        beta_disp = pm.Normal("beta_disp", 0, 1)
        beta_price = pm.Normal("beta_price", 0, 1)
        alphas = pm.Normal("alpha", 0, 1, dims="alts_intercepts")
        ## Hierarchical parameters for individual taste
        beta_individual = pm.Normal(
            "beta_individual", 0, 0.1, dims=("individuals", "alts_intercepts")
        )

        u0 = (
            (alphas[0] + beta_individual[person_indx, 0])
            + beta_disp * disp_sunshine
            + beta_feat * feat_sunshine
            + beta_price * price_sunshine
        )
        u1 = (
            (alphas[1] + beta_individual[person_indx, 1])
            + beta_disp * disp_keebler
            + beta_feat * feat_keebler
            + beta_price * price_keebler
        )
        u2 = (
            (alphas[2] + beta_individual[person_indx, 2])
            + beta_disp * disp_nabisco
            + beta_feat * feat_nabisco
            + beta_price * price_nabisco
        )
        u3 = (
            (0 + beta_individual[person_indx, 2])
            + beta_disp * disp_private
            + beta_feat * feat_private
            + beta_price * price_private
        )
        s = pm.math.stack([u0, u1, u2, u3]).T

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
