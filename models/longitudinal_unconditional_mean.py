"""
Model: Longitudinal unconditional mean model (alcohol use)
Source: pymc-examples/examples/time_series/longitudinal_models.ipynb, Section: "The Unconditional Mean Model"
Authors: Nathaniel Forde, Osvaldo Martin
Description: Hierarchical intercept-only (grand mean) model of teen alcohol usage with per-subject random intercept.

Changes from original:
- Load data from .npz instead of pm.get_data csv
- Removed sampling/plotting code

Benchmark results:
- Original:  logp = -658.2980, grad norm = 210.3274, 8.8 us/call (100000 evals)
- Frozen:    logp = -658.2980, grad norm = 210.3274, 5.9 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm


def build_model():
    data = np.load(Path(__file__).parent / "data" / "alcohol1_pp.npz")
    ids_raw = data["id"]
    alcuse = data["alcuse"]

    id_indx, unique_ids = pd.factorize(ids_raw)
    coords = {"ids": unique_ids, "obs": range(len(alcuse))}

    with pm.Model(coords=coords) as model:
        subject_intercept_sigma = pm.HalfNormal("subject_intercept_sigma", 2)
        subject_intercept = pm.Normal("subject_intercept", 0, subject_intercept_sigma, dims="ids")
        global_sigma = pm.HalfStudentT("global_sigma", 1, 3)
        global_intercept = pm.Normal("global_intercept", 0, 1)
        grand_mean = pm.Deterministic(
            "grand_mean", global_intercept + subject_intercept[id_indx]
        )
        outcome = pm.Normal(
            "outcome", grand_mean, global_sigma, observed=alcuse, dims="obs"
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
