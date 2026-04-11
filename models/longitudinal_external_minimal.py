"""
Model: Longitudinal minimal censored Gumbel model (externalizing behaviour)
Source: pymc-examples/examples/time_series/longitudinal_models.ipynb, Section: "A Minimal Model"
Authors: Nathaniel Forde, Osvaldo Martin
Description: Hierarchical intercept-only model of child externalizing behaviour scores with a censored Gumbel likelihood on [0, 68].

Changes from original:
- Load data from .npz instead of pm.get_data csv
- pm.MutableData -> pm.Data (pm.MutableData removed from API)
- Removed sampling/plotting code

Benchmark results:
- Original:  logp = -1087.7876, grad norm = 80.9135, 12.3 us/call (100000 evals)
- Frozen:    logp = -1087.7876, grad norm = 80.9135, 4.7 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm


def build_model():
    data = np.load(Path(__file__).parent / "data" / "external_pp.npz")
    ids_raw = data["ID"]
    external_arr = data["EXTERNAL"]

    id_indx, unique_ids = pd.factorize(ids_raw)
    coords = {"ids": unique_ids, "obs": range(len(external_arr))}

    with pm.Model(coords=coords) as model:
        external = pm.Data("external_data", external_arr + 1e-25)
        global_intercept = pm.Normal("global_intercept", 6, 1)
        global_sigma = pm.HalfNormal("global_sigma", 7)

        subject_intercept_sigma = pm.HalfNormal("subject_intercept_sigma", 5)
        subject_intercept = pm.Normal(
            "subject_intercept", 0, subject_intercept_sigma, dims="ids"
        )
        mu = pm.Deterministic("mu", global_intercept + subject_intercept[id_indx])
        outcome_latent = pm.Gumbel.dist(mu, global_sigma)
        outcome = pm.Censored(
            "outcome", outcome_latent, lower=0, upper=68, observed=external, dims="obs"
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
