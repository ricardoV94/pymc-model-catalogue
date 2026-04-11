"""
Model: Robust Regression - Student-T Likelihood
Source: pymc-examples/examples/generalized_linear_models/GLM-robust-with-outlier-detection.ipynb, Section: "4. Simple Linear Model with Robust Student-T Likelihood"
Authors: Jon Sedar, Thomas Wiecki, Raul Maldonado, Oriol Abril-Pla
Description: Linear regression with Student-T likelihood for robustness to outliers,
    using InverseGamma prior on degrees of freedom, on the Hogg 2010 standardized dataset.

Changes from original:
- Inlined and standardized the Hogg 2010 dataset.
- Removed sampling and plotting code.

Benchmark results:
- Original:  logp = -30.8034, grad norm = 31.8900, 4.0 us/call (100000 evals)
- Frozen:    logp = -30.8034, grad norm = 31.8900, 3.9 us/call (100000 evals)
"""

import numpy as np
import pymc as pm


def build_model():
    # Hogg 2010 data (hardcoded from paper)
    # fmt: off
    raw = np.array([
        [1, 201, 592, 61, 9, -0.84],
        [2, 244, 401, 25, 4, 0.31],
        [3, 47, 583, 38, 11, 0.64],
        [4, 287, 402, 15, 7, -0.27],
        [5, 203, 495, 21, 5, -0.33],
        [6, 58, 173, 15, 9, 0.67],
        [7, 210, 479, 27, 4, -0.02],
        [8, 202, 504, 14, 4, -0.05],
        [9, 198, 510, 30, 11, -0.84],
        [10, 158, 416, 16, 7, -0.69],
        [11, 165, 393, 14, 5, 0.30],
        [12, 201, 442, 25, 5, -0.46],
        [13, 157, 317, 52, 5, -0.03],
        [14, 131, 311, 16, 6, 0.50],
        [15, 166, 400, 34, 6, 0.73],
        [16, 160, 337, 31, 5, -0.52],
        [17, 186, 423, 42, 9, 0.90],
        [18, 125, 334, 26, 8, 0.40],
        [19, 218, 533, 16, 6, -0.78],
        [20, 146, 344, 22, 5, -0.56],
    ])
    # fmt: on
    x_raw = raw[:, 1]
    y_raw = raw[:, 2]
    sigma_y_raw = raw[:, 3]

    # Standardize (mean center and divide by 2 sd)
    x_mean, x_std = x_raw.mean(), x_raw.std()
    y_mean, y_std = y_raw.mean(), y_raw.std()
    x = (x_raw - x_mean) / (2 * x_std)
    y = (y_raw - y_mean) / (2 * y_std)
    sigma_y = sigma_y_raw / (2 * y_std)

    datapoint_ids = [f"p{int(i)}" for i in raw[:, 0]]
    coords = {"coefs": ["intercept", "slope"], "datapoint_id": datapoint_ids}

    with pm.Model(coords=coords) as model:
        # define weakly informative Normal priors to give Ridge regression
        beta = pm.Normal("beta", mu=0, sigma=10, dims="coefs")

        # define linear model
        y_est = beta[0] + beta[1] * x

        # define prior for StudentT degrees of freedom
        # InverseGamma has nice properties:
        # it's continuous and has support x in (0, inf)
        nu = pm.InverseGamma("nu", alpha=1, beta=1)

        # define Student T likelihood
        pm.StudentT(
            "y", mu=y_est, sigma=sigma_y, nu=nu, observed=y, dims="datapoint_id"
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
