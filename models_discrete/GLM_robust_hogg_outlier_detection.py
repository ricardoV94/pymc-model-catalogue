"""
Model: Robust Regression - Hogg Signal vs Noise Outlier Detection
Source: pymc-examples/examples/generalized_linear_models/GLM-robust-with-outlier-detection.ipynb, Section: "5. Linear Model with Custom Likelihood to Distinguish Outliers: Hogg Method"
Authors: Jon Sedar, Thomas Wiecki, Raul Maldonado, Oriol Abril-Pla
Description: Linear regression with custom mixture likelihood for inlier/outlier
    classification using the Hogg 2010 method. Uses Bernoulli indicators for outlier
    class membership and a Potential for the mixture log-likelihood.

Has discrete variables: Yes (is_outlier)

Changes from original:
- Inlined and standardized the Hogg 2010 dataset.
- pm.ConstantData -> pm.Data.
- Removed sampling, plotting, and outlier classification code.

Benchmark results:
- Original:  logp = -197.9222, 4.8 us/call (100000 evals)
- Frozen:    logp = -197.9222, 4.0 us/call (100000 evals)
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
        # state input data as shared vars
        tsv_x = pm.Data("tsv_x", x, dims="datapoint_id")
        tsv_y = pm.Data("tsv_y", y, dims="datapoint_id")
        tsv_sigma_y = pm.Data("tsv_sigma_y", sigma_y, dims="datapoint_id")

        # weakly informative Normal priors (L2 ridge reg) for inliers
        beta = pm.Normal("beta", mu=0, sigma=10, dims="coefs")

        # linear model for mean for inliers
        y_est_in = beta[0] + beta[1] * tsv_x

        # very weakly informative mean for all outliers
        y_est_out = pm.Normal("y_est_out", mu=0, sigma=10, initval=pm.floatX(0.0))

        # very weakly informative prior for additional variance for outliers
        sigma_y_out = pm.HalfNormal("sigma_y_out", sigma=10, initval=pm.floatX(1.0))

        # create in/outlier distributions to get a logp evaluated on the observed y
        inlier_logp = pm.logp(pm.Normal.dist(mu=y_est_in, sigma=tsv_sigma_y), tsv_y)
        outlier_logp = pm.logp(
            pm.Normal.dist(mu=y_est_out, sigma=tsv_sigma_y + sigma_y_out), tsv_y
        )

        # frac_outliers only needs to span [0, .5]
        frac_outliers = pm.Uniform("frac_outliers", lower=0.0, upper=0.5)
        is_outlier = pm.Bernoulli(
            "is_outlier",
            p=frac_outliers,
            initval=(np.random.rand(len(x)) < 0.4).astype(int),
            dims="datapoint_id",
        )

        # non-sampled Potential evaluates the Normal.dist.logp's
        pm.Potential(
            "obs",
            ((1 - is_outlier) * inlier_logp).sum() + (is_outlier * outlier_logp).sum(),
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model, discrete=True)
