"""
Model: Golf Putting Distance-Angle Model
Source: pymc-examples/examples/case_studies/putting_workflow.ipynb, Section: "A model incorporating distance to hole"
Authors: Colin Carroll, Mark Broadie
Description: Physics-based model for golf putting success probability that combines
    angular accuracy (probability of correct aim) and distance control (probability
    of correct distance) using geometry-derived formulas.

Changes from original:
- Inlined Broadie (2018) golf putting data directly
- Removed sampling, plotting, and model comparison code

Benchmark results:
- Original:  logp = -3136830.7439, grad norm = 1019536.8950, 8.3 us/call (100000 evals)
- Frozen:    logp = -3136830.7439, grad norm = 1019536.8950, 5.8 us/call (100000 evals)
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    def phi(x):
        return 0.5 + 0.5 * pt.erf(x / pt.sqrt(2.0))

    # Golf putting data from Broadie (2018)
    # fmt: off
    distance = np.array([
        0.28, 0.97, 1.93, 2.92, 3.93, 4.94, 5.94, 6.95, 7.95, 8.95,
        9.95, 10.95, 11.95, 12.95, 14.43, 16.43, 18.44, 20.44, 21.95,
        24.39, 28.40, 32.39, 36.39, 40.37, 44.38, 48.37, 52.36, 57.25,
        63.23, 69.18, 75.19,
    ])
    tries = np.array([
        45198, 183020, 169503, 113094, 73855, 53659, 42991, 37050, 33275,
        30836, 28637, 26239, 24636, 22876, 41267, 35712, 31573, 28280,
        13238, 46570, 38422, 31641, 25604, 20366, 15977, 11770, 8708,
        8878, 5492, 3087, 1742,
    ])
    successes = np.array([
        45183, 182899, 168594, 108953, 64740, 41106, 28205, 21334, 16615,
        13503, 11060, 9032, 7687, 6432, 9813, 7196, 5290, 4086,
        1642, 4767, 2980, 1996, 1327, 834, 559, 311, 231,
        204, 103, 35, 24,
    ])
    # fmt: on

    BALL_RADIUS = (1.68 / 2) / 12
    CUP_RADIUS = (4.25 / 2) / 12
    OVERSHOT = 1.0
    DISTANCE_TOLERANCE = 3.0

    coords = {"obs_id": np.arange(len(distance))}

    with pm.Model(coords=coords) as model:
        distance_ = pm.Data("distance", distance, dims="obs_id")
        tries_ = pm.Data("tries", tries, dims="obs_id")
        successes_ = pm.Data("successes", successes, dims="obs_id")

        variance_of_shot = pm.HalfNormal("variance_of_shot")
        variance_of_distance = pm.HalfNormal("variance_of_distance")

        p_good_angle = pm.Deterministic(
            "p_good_angle",
            2 * phi(pt.arcsin((CUP_RADIUS - BALL_RADIUS) / distance_) / variance_of_shot) - 1,
            dims="obs_id",
        )
        p_good_distance = pm.Deterministic(
            "p_good_distance",
            phi((DISTANCE_TOLERANCE - OVERSHOT) / ((distance_ + OVERSHOT) * variance_of_distance))
            - phi(-OVERSHOT / ((distance_ + OVERSHOT) * variance_of_distance)),
            dims="obs_id",
        )

        success = pm.Binomial(
            "success",
            n=tries_,
            p=p_good_angle * p_good_distance,
            observed=successes_,
            dims="obs_id",
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
