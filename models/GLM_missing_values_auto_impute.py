"""
Model: GLM with Auto-Imputed Missing Values (ModelA)
Source: pymc-examples/examples/generalized_linear_models/GLM-missing-values-in-covariates.ipynb, Section: "2. ModelA: Auto-impute Missing Values"
Authors: Jonathan Sedar
Description: Linear regression with automatic imputation of missing covariate values
    using hierarchical Normal priors on the missing data, demonstrating PyMC's
    built-in missing value handling.

Changes from original:
- Inlined synthetic data generation (reproducing notebook's RNG seed=42).
- pm.MutableData -> pm.Data (API fix).
- Removed sampling, plotting, and holdout prediction code.

Benchmark results:
- Original:  logp = -1365460.1059, grad norm = 2730693.0819, 7.1 us/call (100000 evals)
- Frozen:    logp = -1365460.1059, grad norm = 2730693.0819, 6.4 us/call (100000 evals)
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

    # Introduce 40% missing values in columns c and d
    idx_all = np.arange(N)
    missing_c = RNG.choice(idx_all, int(N * 0.4), replace=False)
    missing_d = RNG.choice(idx_all, int(N * 0.4), replace=False)

    df_c = dfraw_c.copy()
    df_d = dfraw_d.copy()
    df_c[missing_c] = np.nan
    df_d[missing_d] = np.nan

    # Train split (same logic as notebook: sample n-10 without replacement)
    train_idx = RNG.choice(N, N - 10, replace=False)
    train_idx.sort()

    train_a = dfraw_a[train_idx]
    train_b = dfraw_b[train_idx]
    train_c = df_c[train_idx]
    train_d = df_d[train_idx]
    train_y = dfraw_y[train_idx]

    # Standardize using train stats (nanmean/nanstd for cols with missing)
    FTS_NUM = np.column_stack([train_a, train_b, train_c, train_d])
    MNS = np.nanmean(FTS_NUM, axis=0)
    SDEVS = np.nanstd(FTS_NUM, axis=0)

    train_a_std = (train_a - MNS[0]) / SDEVS[0]
    train_b_std = (train_b - MNS[1]) / SDEVS[1]
    train_c_std = (train_c - MNS[2]) / SDEVS[2]
    train_d_std = (train_d - MNS[3]) / SDEVS[3]

    # Build Xj (non-missing features: intercept, a, b)
    intercept_col = np.ones(len(train_idx))
    train_Xj = np.column_stack([intercept_col, train_a_std, train_b_std])

    # Build Xk (features with missing values: c, d)
    train_Xk = np.column_stack([train_c_std, train_d_std])

    oid = np.array([f"o{str(i).zfill(2)}" for i in train_idx])
    FTS_XJ = ["intercept", "a", "b"]
    FTS_XK = ["c", "d"]

    COORDS = dict(
        xj_nm=FTS_XJ,
        xk_nm=FTS_XK,
        oid=oid,
    )

    with pm.Model(coords=COORDS) as model:
        # 0. create Data containers for obs (Y, X)
        y = pm.Data("y", train_y, dims="oid")
        xj = pm.Data("xj", train_Xj, dims=("oid", "xj_nm"))

        # 1. create auto-imputing likelihood for missing data values
        # NOTE: there's no way to put a nan-containing array into pm.Data,
        # so train_Xk has to go in directly
        xk_mu = pm.Normal("xk_mu", mu=0.0, sigma=1, dims="xk_nm")
        xk = pm.Normal(
            "xk", mu=xk_mu, sigma=1.0, observed=train_Xk, dims=("oid", "xk_nm")
        )

        # 2. define priors for contiguous and auto-imputed data
        b_s = pm.Gamma("beta_sigma", alpha=10, beta=10)  # E ~ 1
        bj = pm.Normal("beta_j", mu=0, sigma=b_s, dims="xj_nm")
        bk = pm.Normal("beta_k", mu=0, sigma=b_s, dims="xk_nm")

        # 4. define evidence
        epsilon = pm.Gamma("epsilon", alpha=50, beta=50)  # encourage E ~ 1
        lm = pt.dot(xj, bj.T) + pt.dot(xk, bk.T)
        pm.Normal("yhat", mu=lm, sigma=epsilon, observed=y, dims="oid")

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
