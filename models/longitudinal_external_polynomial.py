"""
Model: Longitudinal polynomial censored Gumbel growth model (externalizing behaviour)
Source: pymc-examples/examples/time_series/longitudinal_models.ipynb, Section: "Adding in Polynomial Time"
Authors: Nathaniel Forde, Osvaldo Martin
Description: Hierarchical quadratic-in-grade growth model of child externalizing behaviour scores with a censored Gumbel likelihood on [0, 68].

Changes from original:
- Load data from .npz instead of pm.get_data csv
- pm.MutableData -> pm.Data (pm.MutableData removed from API)
- Removed sampling/plotting code

Benchmark results:
- Original:  logp = -1100.7584, grad norm = 147.6083, 16.3 us/call (100000 evals)
- Frozen:    logp = -1100.7584, grad norm = 147.6083, 7.5 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm


def build_model():
    data = np.load(Path(__file__).parent / "data" / "external_pp.npz")
    ids_raw = data["ID"]
    external_arr = data["EXTERNAL"]
    grade_arr = data["GRADE"]

    id_indx, unique_ids = pd.factorize(ids_raw)
    coords = {"ids": unique_ids, "obs": range(len(external_arr))}

    with pm.Model(coords=coords) as model:
        grade = pm.Data("grade_data", grade_arr)
        grade2 = pm.Data("grade2_data", grade_arr ** 2)
        external = pm.Data("external_data", external_arr + 1e-25)

        global_intercept = pm.Normal("global_intercept", 6, 2)
        global_sigma = pm.Normal("global_sigma", 7, 1)
        global_beta_grade = pm.Normal("global_beta_grade", 0, 1)
        global_beta_grade2 = pm.Normal("global_beta_grade2", 0, 1)

        subject_intercept_sigma = pm.HalfNormal("subject_intercept_sigma", 1)
        subject_intercept = pm.Normal(
            "subject_intercept", 2, subject_intercept_sigma, dims="ids"
        )

        subject_beta_grade_sigma = pm.HalfNormal("subject_beta_grade_sigma", 1)
        subject_beta_grade = pm.Normal(
            "subject_beta_grade", 0, subject_beta_grade_sigma, dims="ids"
        )

        subject_beta_grade2_sigma = pm.HalfNormal("subject_beta_grade2_sigma", 1)
        subject_beta_grade2 = pm.Normal(
            "subject_beta_grade2", 0, subject_beta_grade2_sigma, dims="ids"
        )

        mu = pm.Deterministic(
            "mu",
            global_intercept
            + subject_intercept[id_indx]
            + (global_beta_grade + subject_beta_grade[id_indx]) * grade
            + (global_beta_grade2 + subject_beta_grade2[id_indx]) * grade2,
        )
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
