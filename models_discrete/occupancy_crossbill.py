"""
Model: Site Occupancy Model (Red Crossbills in Switzerland)
Source: pymc-examples/examples/case_studies/occupancy.ipynb, Section: "Site occupancy model"
Authors: Philip T. Patton
Description: Site occupancy model for estimating species distributions from
    detection/non-detection data, with logit-linear models for both occurrence
    and detection probabilities. The discrete latent variable z represents
    true occupancy state at each quadrat.

Has discrete variables: Yes (z)

Changes from original:
- Saved preprocessed design matrices and response data to .npz
- Removed marginalization, sampling, plotting, and prediction code

Benchmark results:
- Original:  logp = -677.8664, 22.5 us/call (100000 evals)
- Frozen:    logp = -677.8664, 23.0 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data_path = Path(__file__).parent / "data" / "occupancy_crossbill.npz"
    data = np.load(data_path)

    y = data["y"]
    X = data["X"]
    W = data["W"]
    quadrat_count = int(data["quadrat_count"])
    survey_count = int(data["survey_count"])

    coords = {
        "survey_effects": ["intercept", "date"],
        "quadrat_effects": ["intercept", "forest_cover", "elevation", "elevation2"],
        "quadrats": np.arange(quadrat_count),
        "surveys": np.arange(survey_count),
    }

    with pm.Model(coords=coords) as model:
        # Occurrence probability model
        beta = pm.Normal("beta", 0, 2, dims="quadrat_effects")
        occurrence_probability = pm.math.invlogit(pm.math.dot(X, beta))

        # Detection probability model
        alpha = pm.Normal("alpha", 0, 2, dims="survey_effects")
        detection_probability = pm.math.invlogit(pm.math.dot(W, alpha))

        # Occupied / unoccupied state at each site
        z = pm.Bernoulli("z", occurrence_probability, dims="quadrats")

        # Likelihood
        pm.Bernoulli(
            "y",
            z[:, None] * detection_probability,
            dims=["quadrats", "surveys"],
            observed=y,
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model, discrete=True)
