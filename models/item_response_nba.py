"""
Model: NBA Foul Analysis with Item Response Theory (Rasch Model)
Source: pymc-examples/examples/case_studies/item_response_nba.ipynb, Section: "PyMC implementation"
Authors: Austin Rochford, Lorenzo Toniazzi
Description: Hierarchical Rasch model for NBA foul call data, estimating latent
    disadvantaged-player skill (theta) and committing-player skill (b) via a
    non-centered parameterization with partial pooling.

Changes from original:
- Saved preprocessed data to .npz (factorized player indices and foul_called)
- Removed sampling, plotting, and posterior analysis code

Benchmark results:
- Original:  logp = -33922.0088, grad norm = 12843.3947, 1430.0 us/call (10543 evals)
- Frozen:    logp = -33922.0088, grad norm = 12843.3947, 1606.8 us/call (9483 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data_path = Path(__file__).parent / "data" / "item_response_nba.npz"
    data = np.load(data_path, allow_pickle=True)

    committing_observed = data["committing_observed"]
    disadvantaged_observed = data["disadvantaged_observed"]
    foul_called = data["foul_called"]
    committing = data["committing"]
    disadvantaged = data["disadvantaged"]

    coords = {"disadvantaged": disadvantaged, "committing": committing}

    with pm.Model(coords=coords) as model:
        # Data
        foul_called_observed = pm.Data("foul_called_observed", foul_called)

        # Hyperpriors
        mu_theta = pm.Normal("mu_theta", 0.0, 100.0)
        sigma_theta = pm.HalfCauchy("sigma_theta", 2.5)
        sigma_b = pm.HalfCauchy("sigma_b", 2.5)

        # Priors (non-centered)
        Delta_theta = pm.Normal("Delta_theta", 0.0, 1.0, dims="disadvantaged")
        Delta_b = pm.Normal("Delta_b", 0.0, 1.0, dims="committing")

        # Deterministic
        theta = pm.Deterministic(
            "theta", Delta_theta * sigma_theta + mu_theta, dims="disadvantaged"
        )
        b = pm.Deterministic("b", Delta_b * sigma_b, dims="committing")
        eta = pm.Deterministic(
            "eta", theta[disadvantaged_observed] - b[committing_observed]
        )

        # Likelihood
        y = pm.Bernoulli("y", logit_p=eta, observed=foul_called_observed)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
