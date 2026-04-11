"""
Model: GLM Baseline without Missing Values (Model0)
Source: pymc-examples/examples/generalized_linear_models/GLM-missing-values-in-covariates.ipynb, Section: "1. Model0: Baseline without Missing Values"
Authors: Jonathan Sedar
Description: Baseline linear regression on synthetic data with 4 predictors (no missing
    values), using Gamma priors on coefficient and noise scales.

Changes from original:
- Inlined synthetic data generation (reproducing notebook's RNG seed=42).
- pm.MutableData -> pm.Data (API fix).
- Removed sampling, plotting, and holdout prediction code.

Benchmark results:
- Original:  logp = -1339569.0249, grad norm = 2679061.1204, 4.2 us/call (100000 evals)
- Frozen:    logp = -1339569.0249, grad norm = 2679061.1204, 3.3 us/call (100000 evals)
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    RNG = np.random.default_rng(seed=42)

    REFVAL_X_MU = dict(a=1, b=1, c=10, d=2)
    REFVAL_X_SIGMA = dict(a=1, b=4, c=1, d=10)
    REFVAL_BETA = dict(intercept=-4, a=10, b=-20, c=30, d=5)

    N = 40
    dfraw_a = RNG.normal(loc=REFVAL_X_MU["a"], scale=REFVAL_X_SIGMA["a"], size=N)
    dfraw_b = RNG.normal(loc=REFVAL_X_MU["b"], scale=REFVAL_X_SIGMA["b"], size=N)
    dfraw_c = RNG.normal(loc=REFVAL_X_MU["c"], scale=REFVAL_X_SIGMA["c"], size=N)
    dfraw_d = RNG.normal(loc=REFVAL_X_MU["d"], scale=REFVAL_X_SIGMA["d"], size=N)

    dfraw_X = np.column_stack([dfraw_a, dfraw_b, dfraw_c, dfraw_d])
    dfraw_y = (
        REFVAL_BETA["intercept"]
        + (dfraw_X * np.array([10, -20, 30, 5])).sum(axis=1)
        + RNG.normal(loc=0, scale=1, size=N)
    )

    # Train split (same logic as notebook: sample n-10 without replacement)
    train_idx = RNG.choice(N, N - 10, replace=False)
    train_idx.sort()

    # Standardize using train stats
    FTS_NUM = ["a", "b", "c", "d"]
    train_X = dfraw_X[train_idx]
    train_y = dfraw_y[train_idx]

    MNS_RAW = np.nanmean(train_X, axis=0)
    SDEVS_RAW = np.nanstd(train_X, axis=0)

    train_X_std = (train_X - MNS_RAW) / SDEVS_RAW
    # Add intercept column
    intercept_col = np.ones(len(train_idx))
    train_Xj = np.column_stack([intercept_col, train_X_std])

    oid = np.array([f"o{str(i).zfill(2)}" for i in train_idx])
    FTS_XJ = ["intercept", "a", "b", "c", "d"]

    COORDS = dict(
        xj_nm=FTS_XJ,
        oid=oid,
    )

    with pm.Model(coords=COORDS) as model:
        # 0. create Data containers for obs (Y, X)
        y = pm.Data("y", train_y, dims="oid")
        xj = pm.Data("xj", train_Xj, dims=("oid", "xj_nm"))

        # 2. define priors for contiguous data
        b_s = pm.Gamma("beta_sigma", alpha=10, beta=10)  # E ~ 1
        bj = pm.Normal("beta_j", mu=0, sigma=b_s, dims="xj_nm")

        # 4. define evidence
        epsilon = pm.Gamma("epsilon", alpha=50, beta=50)  # encourage E ~ 1
        lm = pt.dot(xj, bj.T)
        pm.Normal("yhat", mu=lm, sigma=epsilon, observed=y, dims="oid")

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
