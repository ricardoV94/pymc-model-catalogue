"""
Model: Conditional Autoregressive (CAR) Model for Scotland Lip Cancer
Source: pymc-examples/examples/spatial/conditional_autoregressive_priors.ipynb, Section: "Our third model: a spatial random effects model, with unknown spatial dependence"
Authors: Conor Hassan, Daniel Saunders
Description: Poisson regression for Scottish lip cancer counts (N=56 areas) with an
    independent random effect and a spatially dependent CAR random effect. The spatial
    dependence parameter alpha is estimated via a Beta(1, 1) prior.

Changes from original:
- Inlined the scotland_lips_cancer dataset (CANCER, CEXP, AFF, ADJ columns) as numpy literals.
- Removed sampling and plotting code.

Benchmark results:
- Original:  logp = -346.4586, grad norm = 224.0335, 6.4 us/call (100000 evals)
- Frozen:    logp = -346.4586, grad norm = 224.0335, 6.5 us/call (100000 evals)
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    # Inlined scotland_lips_cancer data (N=56 areas)
    y = np.array(
        [9, 39, 11, 9, 15, 8, 26, 7, 6, 20, 13, 5, 3, 8, 17, 9, 2, 7, 9, 7,
         16, 31, 11, 7, 19, 15, 7, 10, 16, 11, 5, 3, 7, 8, 11, 9, 11, 8, 6, 4,
         10, 8, 2, 6, 19, 3, 2, 3, 28, 6, 1, 1, 1, 1, 0, 0],
        dtype="int64",
    )
    E = np.array(
        [1.38, 8.66, 3.04, 2.53, 4.26, 2.40, 8.11, 2.30, 1.98, 6.63, 4.40, 1.79,
         1.08, 3.31, 7.84, 4.55, 1.07, 4.18, 5.53, 4.44, 10.46, 22.67, 8.77, 5.62,
         15.47, 12.49, 6.04, 8.96, 14.37, 10.20, 4.75, 2.88, 7.03, 8.53, 12.32,
         10.10, 12.68, 9.35, 7.20, 5.27, 18.76, 15.78, 4.32, 14.63, 50.72, 8.20,
         5.59, 9.34, 88.66, 19.62, 3.44, 3.62, 5.74, 7.03, 4.16, 1.76],
        dtype="float64",
    )
    aff = np.array(
        [16, 16, 10, 24, 10, 24, 10, 7, 7, 16, 7, 16, 10, 24, 7, 16, 10, 7, 7, 10,
         7, 16, 10, 7, 1, 1, 7, 7, 10, 10, 7, 24, 10, 7, 7, 0, 10, 1, 16, 0, 1, 16,
         16, 0, 1, 7, 1, 1, 0, 1, 1, 0, 1, 1, 16, 10],
        dtype="float64",
    )

    # Adjacency lists (1-indexed as in the original CSV)
    adj_raw = [
        [5, 9, 11, 19],
        [7, 10],
        [6, 12],
        [18, 20, 28],
        [1, 11, 12, 13, 19],
        [3, 8],
        [2, 10, 13, 16, 17],
        [6],
        [1, 11, 17, 19, 23, 29],
        [2, 7, 16, 22],
        [1, 5, 9, 12],
        [3, 5, 11],
        [5, 7, 17, 19],
        [31, 32, 35],
        [25, 29, 50],
        [7, 10, 17, 21, 22, 29],
        [7, 9, 13, 16, 19, 29],
        [4, 20, 28, 33, 55, 56],
        [1, 5, 9, 13, 17],
        [4, 18, 55],
        [16, 29, 50],
        [10, 16],
        [9, 29, 34, 36, 37, 39],
        [27, 30, 31, 44, 47, 48, 55, 56],
        [15, 26, 29],
        [25, 29, 42, 43],
        [24, 31, 32, 55],
        [4, 18, 33, 45],
        [9, 15, 16, 17, 21, 23, 25, 26, 34, 43, 50],
        [24, 38, 42, 44, 45, 56],
        [14, 24, 27, 32, 35, 46, 47],
        [14, 27, 31, 35],
        [18, 28, 45, 56],
        [23, 29, 39, 40, 42, 43, 51, 52, 54],
        [14, 31, 32, 37, 46],
        [23, 37, 39, 41],
        [23, 35, 36, 41, 46],
        [30, 42, 44, 49, 51, 54],
        [23, 34, 36, 40, 41],
        [34, 39, 41, 49, 52],
        [36, 37, 39, 40, 46, 49, 53],
        [26, 30, 34, 38, 43, 51],
        [26, 29, 34, 42],
        [24, 30, 38, 48, 49],
        [28, 30, 33, 56],
        [31, 35, 37, 41, 47, 53],
        [24, 31, 46, 48, 49, 53],
        [24, 44, 47, 49],
        [38, 40, 41, 44, 47, 48, 52, 53, 54],
        [15, 21, 29],
        [34, 38, 42, 54],
        [34, 40, 49, 54],
        [41, 46, 47, 49],
        [34, 38, 49, 51, 52],
        [18, 20, 24, 27, 56],
        [18, 24, 30, 33, 45, 55],
    ]

    # number of observations
    N = len(y)

    logE = np.log(E)

    # proportion of the population engaged in agriculture, forestry, or fishing
    x = aff / 10.0

    # change to Python indexing (i.e. -1)
    adj = [[j - 1 for j in row] for row in adj_raw]

    # storing the adjacency matrix as a two-dimensional np.array
    adj_matrix = np.zeros((N, N), dtype="int32")
    for area in range(N):
        adj_matrix[area, adj[area]] = 1

    with pm.Model(coords={"area_idx": np.arange(N)}) as car_model:
        beta0 = pm.Normal("beta0", mu=0.0, tau=1.0e-5)
        beta1 = pm.Normal("beta1", mu=0.0, tau=1.0e-5)
        # variance parameter of the independent random effect
        tau_ind = pm.Gamma("tau_ind", alpha=3.2761, beta=1.81)
        # variance parameter of the spatially dependent random effects
        tau_spat = pm.Gamma("tau_spat", alpha=1.0, beta=1.0)

        # prior for alpha
        alpha = pm.Beta("alpha", alpha=1, beta=1)

        # area-specific model parameters
        # independent random effect
        theta = pm.Normal("theta", mu=0, tau=tau_ind, dims="area_idx")
        # spatially dependent random effect
        phi = pm.CAR(
            "phi", mu=np.zeros(N), tau=tau_spat, alpha=alpha, W=adj_matrix, dims="area_idx"
        )

        # exponential of the linear predictor -> the mean of the likelihood
        mu = pm.Deterministic(
            "mu", pt.exp(logE + beta0 + beta1 * x + theta + phi), dims="area_idx"
        )

        # likelihood of the observed data
        y_i = pm.Poisson("y_i", mu=mu, observed=y, dims="area_idx")

    ip = car_model.initial_point()
    car_model.rvs_to_initial_values = {rv: None for rv in car_model.free_RVs}
    return car_model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
