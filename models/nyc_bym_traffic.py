"""
Model: Besag-York-Mollie (BYM) model for NYC pedestrian traffic accidents
Source: pymc-examples/examples/spatial/nyc_bym.ipynb, Section: "Specifying a BYM model with PyMC"
Authors: Daniel Saunders
Description: A Besag-York-Mollie spatial model of pedestrian traffic accident counts
    across ~1921 NYC census tracts. Combines an ICAR spatially structured random
    effect with an independent random effect via a mixture parameterization, with
    a social fragmentation index predictor and a Poisson likelihood with log
    population offset.

Changes from original:
- Preprocessed edgelist + traffic CSVs into a single .npz file (adjacency matrix,
  counts, log offset, fragment index, area indices, scaling factor) loaded inside
  build_model()
- Removed sampling, posterior predictive, and plotting code

Benchmark results:
- Original:  logp = -6971496.5662, grad norm = 7337748.9594, 36.5 us/call (88371 evals)
- Frozen:    logp = -6971496.5662, grad norm = 7337748.9594, 37.9 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    data = np.load(Path(__file__).parent / "data" / "nyc_bym.npz", allow_pickle=True)
    W_nyc = data["W_nyc"].astype(np.float64)
    y = data["y"]
    log_E = data["log_E"]
    fragment_index = data["fragment_index"]
    area_idx = data["area_idx"]
    scaling_factor = float(data["scaling_factor"])

    coords = {"area_idx": area_idx}

    with pm.Model(coords=coords) as model:
        # intercept
        beta0 = pm.Normal("beta0", 0, 1)

        # fragmentation effect
        beta1 = pm.Normal("beta1", 0, 1)

        # independent random effect
        theta = pm.Normal("theta", 0, 1, dims="area_idx")

        # spatially structured random effect
        phi = pm.ICAR("phi", W=W_nyc, dims="area_idx")

        # joint variance of random effects
        sigma = pm.HalfNormal("sigma", 1)

        # the mixing rate is rho
        rho = pm.Beta("rho", 0.5, 0.5)

        # the bym component - it mixes a spatial and a random effect
        mixture = pm.Deterministic(
            "mixture",
            pt.sqrt(1 - rho) * theta + pt.sqrt(rho / scaling_factor) * phi,
            dims="area_idx",
        )

        # exponential link function to ensure
        # predictions are positive
        mu = pm.Deterministic(
            "mu",
            pt.exp(log_E + beta0 + beta1 * fragment_index + sigma * mixture),
            dims="area_idx",
        )

        y_i = pm.Poisson("y_i", mu, observed=y)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
