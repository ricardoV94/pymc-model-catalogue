"""
Model: Hierarchical Binomial - Rat Tumor
Source: pymc-examples/examples/generalized_linear_models/GLM-hierarchical-binomial-model.ipynb, Section: "Computing the Posterior using PyMC"
Authors: Demetri Pananos, Junpeng Lao, Raul Maldonado, Farhan Reynaldo
Description: Hierarchical Beta-Binomial model for rat tumor rates with a custom
    joint prior on the Beta hyperparameters alpha and beta, following BDA3 Ch. 5.

Changes from original:
- pm.ConstantData -> pm.Data.
- Inlined rat tumor data arrays.
- Removed sampling and plotting code.

Benchmark results:
- Original:  logp = -649.5511, grad norm = 80.8016, 7.9 us/call (100000 evals)
- Frozen:    logp = -649.5511, grad norm = 80.8016, 5.5 us/call (100000 evals)
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    # fmt: off
    y = np.array([
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,
        1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  1,  5,  2,
        5,  3,  2,  7,  7,  3,  3,  2,  9, 10,  4,  4,  4,  4,  4,  4,  4,
        10,  4,  4,  4,  5, 11, 12,  5,  5,  6,  5,  6,  6,  6,  6, 16, 15,
        15,  9,  4
    ])
    n = np.array([
        20, 20, 20, 20, 20, 20, 20, 19, 19, 19, 19, 18, 18, 17, 20, 20, 20,
        20, 19, 19, 18, 18, 25, 24, 23, 20, 20, 20, 20, 20, 20, 10, 49, 19,
        46, 27, 17, 49, 47, 20, 20, 13, 48, 50, 20, 20, 20, 20, 20, 20, 20,
        48, 19, 19, 19, 22, 46, 49, 20, 20, 23, 19, 22, 20, 20, 20, 52, 46,
        47, 24, 14
    ])
    # fmt: on

    N = len(n)

    def logp_ab(value):
        """prior density"""
        return pt.log(pt.pow(pt.sum(value), -5 / 2))

    coords = {
        "obs_id": np.arange(N),
        "param": ["alpha", "beta"],
    }

    with pm.Model(coords=coords) as model:
        # Uninformative prior for alpha and beta
        n_val = pm.Data("n_val", n)
        ab = pm.HalfNormal("ab", sigma=10, dims="param")
        pm.Potential("p(a, b)", logp_ab(ab))

        X = pm.Deterministic("X", pt.log(ab[0] / ab[1]))
        Z = pm.Deterministic("Z", pt.log(pt.sum(ab)))

        theta = pm.Beta("theta", alpha=ab[0], beta=ab[1], dims="obs_id")

        pm.Binomial("y", p=theta, observed=y, n=n_val)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
