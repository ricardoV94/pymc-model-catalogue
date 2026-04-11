"""
Model: Longitudinal growth model with COA and peer effects
Source: pymc-examples/examples/time_series/longitudinal_models.ipynb, Section: "Model controlling for Peer Effects"
Authors: Nathaniel Forde, Osvaldo Martin
Description: Hierarchical growth model of teen alcohol usage adding centered peer score and its interaction with age to the COA growth model.

Changes from original:
- Load data from .npz instead of pm.get_data csv
- pm.MutableData -> pm.Data (pm.MutableData removed from API)
- Removed sampling/plotting code

Benchmark results:
- Original:  logp = -946.0812, grad norm = 229.3536, 6.7 us/call (100000 evals)
- Frozen:    logp = -946.0812, grad norm = 229.3536, 7.1 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm


def build_model():
    data = np.load(Path(__file__).parent / "data" / "alcohol1_pp.npz")
    ids_raw = data["id"]
    age_14_arr = data["age_14"]
    coa_arr = data["coa"]
    cpeer_arr = data["cpeer"]
    alcuse = data["alcuse"]

    id_indx, unique_ids = pd.factorize(ids_raw)
    coords = {"ids": unique_ids, "obs": range(len(alcuse))}

    with pm.Model(coords=coords) as model:
        age_14 = pm.Data("age_14_data", age_14_arr)
        coa = pm.Data("coa_data", coa_arr)
        peer = pm.Data("peer_data", cpeer_arr)

        ## Level 1
        global_intercept = pm.Normal("global_intercept", 0, 1)
        global_sigma = pm.HalfStudentT("global_sigma", 1, 3)
        global_age_beta = pm.Normal("global_age_beta", 0, 1)
        global_coa_beta = pm.Normal("global_coa_beta", 0, 1)
        global_peer_beta = pm.Normal("global_peer_beta", 0, 1)
        global_coa_age_beta = pm.Normal("global_coa_age_beta", 0, 1)
        global_peer_age_beta = pm.Normal("global_peer_age_beta", 0, 1)

        subject_intercept_sigma = pm.HalfNormal("subject_intercept_sigma", 5)
        subject_age_sigma = pm.HalfNormal("subject_age_sigma", 5)

        ## Level 2
        subject_intercept = pm.Normal(
            "subject_intercept", 0, subject_intercept_sigma, dims="ids"
        )
        subject_age_beta = pm.Normal(
            "subject_age_beta", 0, subject_age_sigma, dims="ids"
        )

        growth_model = pm.Deterministic(
            "growth_model",
            (global_intercept + subject_intercept[id_indx])
            + global_coa_beta * coa
            + global_coa_age_beta * (coa * age_14)
            + global_peer_beta * peer
            + global_peer_age_beta * (peer * age_14)
            + (global_age_beta + subject_age_beta[id_indx]) * age_14,
        )
        outcome = pm.Normal(
            "outcome", growth_model, global_sigma, observed=alcuse, dims="obs"
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
