"""
Model: Hierarchical Partial Pooling for Baseball Batting Averages
Source: pymc-examples/examples/case_studies/hierarchical_partial_pooling.ipynb, Section: "Approach"
Authors: Vladislavs Dovgalecs, Adrian Seybolt, Christian Luhmann
Description: Beta-Binomial hierarchical model estimating batting averages for 18 baseball players
    plus a new unobserved player, with shared hyperparameters phi (population mean) and kappa
    (concentration) reparameterized via an exponential prior on log(kappa).

Changes from original:
- Inlined Efron-Morris 1975 baseball data (18 players, At-Bats and Hits columns)
- Inlined player names for coords
- Removed sampling and plotting code

Benchmark results:
- Original:  logp = -169.8985, grad norm = 47.3016, 6.0 us/call (100000 evals)
- Frozen:    logp = -169.8985, grad norm = 47.3016, 6.2 us/call (100000 evals)
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

def build_model():
    # Efron-Morris 1975 baseball data
    # fmt: off
    player_names = [
        "Roberto Clemente", "Frank Robinson", "Frank Howard", "Jay Johnstone",
        "Ken Berry", "Jim Spencer", "Don Kessinger", "Luis Alvarado",
        "Ron Santo", "Ron Swaboda", "Rico Petrocelli", "Ellie Rodriguez",
        "George Scott", "Del Unser", "Billy Williams", "Bert Campaneris",
        "Thurman Munson", "Max Alvis",
    ]
    at_bats = np.array([45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45])
    hits = np.array([18, 17, 16, 15, 14, 14, 13, 12, 11, 11, 10, 10, 10, 10, 10, 9, 8, 7])
    # fmt: on

    N = len(hits)
    coords = {"player_names": player_names}

    with pm.Model(coords=coords) as baseball_model:
        phi = pm.Uniform("phi", lower=0.0, upper=1.0)

        kappa_log = pm.Exponential("kappa_log", lam=1.5)
        kappa = pm.Deterministic("kappa", pt.exp(kappa_log))

        theta = pm.Beta("theta", alpha=phi * kappa, beta=(1.0 - phi) * kappa, dims="player_names")
        y = pm.Binomial("y", n=at_bats, p=theta, dims="player_names", observed=hits)

    # Add the new player (0-for-4)
    with baseball_model:
        theta_new = pm.Beta("theta_new", alpha=phi * kappa, beta=(1.0 - phi) * kappa)
        y_new = pm.Binomial("y_new", n=4, p=theta_new, observed=0)

    ip = baseball_model.initial_point()
    baseball_model.rvs_to_initial_values = {rv: None for rv in baseball_model.free_RVs}
    return baseball_model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
