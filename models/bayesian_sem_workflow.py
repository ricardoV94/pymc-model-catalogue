"""
Model: Bayesian Structural Equation Model (Job Satisfaction)
Source: pymc-examples/examples/case_studies/bayesian_sem_workflow.ipynb, Section: "Structural Regressions"
Authors: Nathaniel Ford
Description: Full structural equation model with four latent constructs (satisfaction,
    well-being, dysfunction, constructive thought) linked by directed causal paths,
    confirmatory factor loadings with first-loading-fixed-to-1 identification,
    and LKJ-Cholesky correlated latent factors.

Changes from original:
- Inlined all utility functions (make_lambda, make_Lambda, make_B)
- Inlined data generation from covariance matrix with fixed seed
- Removed sampling, plotting, and diagnostic code

Benchmark results:
- Original:  logp = -5731.8386, grad norm = 632.4267, 120.7 us/call (73197 evals)
- Frozen:    logp = -5731.8386, grad norm = 632.4267, 123.1 us/call (95453 evals)
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    # Data generation from known covariance structure
    # fmt: off
    stds = np.array(
        [0.939, 1.017, 0.937, 0.562, 0.760, 0.524, 0.585, 0.609, 0.731, 0.711, 1.124, 1.001]
    )
    n = len(stds)

    corr_values = [
        1.000,
        .668, 1.000,
        .635, .599, 1.000,
        .263, .261, .164, 1.000,
        .290, .315, .247, .486, 1.000,
        .207, .245, .231, .251, .449, 1.000,
       -.206, -.182, -.195, -.309, -.266, -.142, 1.000,
       -.280, -.241, -.238, -.344, -.305, -.230,  .753, 1.000,
       -.258, -.244, -.185, -.255, -.255, -.215,  .554,  .587, 1.000,
        .080,  .096,  .094, -.017,  .151,  .141, -.074, -.111,  .016, 1.000,
        .061,  .028, -.035, -.058, -.051, -.003, -.040, -.040, -.018,  .284, 1.000,
        .113,  .174,  .059,  .063,  .138,  .044, -.119, -.073, -.084,  .563,  .379, 1.000,
    ]
    # fmt: on

    corr_matrix = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i + 1):
            corr_matrix[i, j] = corr_values[idx]
            corr_matrix[j, i] = corr_values[idx]
            idx += 1

    cov_matrix = np.outer(stds, stds) * corr_matrix

    FEATURE_COLUMNS = [
        "JW1", "JW2", "JW3", "UF1", "UF2", "FOR",
        "DA1", "DA2", "DA3", "EBA", "ST", "MI",
    ]

    rng = np.random.default_rng(42)
    sample_df = rng.multivariate_normal(np.zeros(12), cov_matrix, size=263)
    observed_data = sample_df

    coords = {
        "obs": list(range(len(sample_df))),
        "indicators": FEATURE_COLUMNS,
        "indicators_1": ["JW1", "JW2", "JW3"],
        "indicators_2": ["UF1", "UF2", "FOR"],
        "indicators_3": ["DA1", "DA2", "DA3"],
        "indicators_4": ["EBA", "ST", "MI"],
        "latent": ["satisfaction", "well being", "dysfunctional", "constructive"],
        "latent1": ["satisfaction", "well being", "dysfunctional", "constructive"],
        "paths": [
            "dysfunctional ~ constructive",
            "well being ~ dysfunctional",
            "well being ~ constructive",
            "satisfaction ~ well being",
            "satisfaction ~ dysfunction",
            "satisfaction ~ constructive",
        ],
    }

    LATENT_ORDER_LKUP = dict(
        zip(coords["latent"], range(len(coords["latent"])))
    )

    with pm.Model(coords=coords) as model:
        # --- Factor loadings ---
        # Each factor has first loading fixed to 1 for identification
        lambdas_1_ = pm.Normal("lambdas1_", 1, 0.5, dims="indicators_1")
        lambdas1 = pm.Deterministic(
            "lambdas1", pt.set_subtensor(lambdas_1_[0], 1), dims="indicators_1"
        )

        lambdas_2_ = pm.Normal("lambdas2_", 1, 0.5, dims="indicators_2")
        lambdas2 = pm.Deterministic(
            "lambdas2", pt.set_subtensor(lambdas_2_[0], 1), dims="indicators_2"
        )

        lambdas_3_ = pm.Normal("lambdas3_", 1, 0.5, dims="indicators_3")
        lambdas3 = pm.Deterministic(
            "lambdas3", pt.set_subtensor(lambdas_3_[0], 1), dims="indicators_3"
        )

        lambdas_4_ = pm.Normal("lambdas4_", 1, 0.5, dims="indicators_4")
        lambdas4 = pm.Deterministic(
            "lambdas4", pt.set_subtensor(lambdas_4_[0], 1), dims="indicators_4"
        )

        # Full loading matrix Lambda (12 x 4)
        Lambda = pt.zeros((12, 4))
        Lambda = pt.set_subtensor(Lambda[0:3, 0], lambdas1)
        Lambda = pt.set_subtensor(Lambda[3:6, 1], lambdas2)
        Lambda = pt.set_subtensor(Lambda[6:9, 2], lambdas3)
        Lambda = pt.set_subtensor(Lambda[9:12, 3], lambdas4)
        Lambda = pm.Deterministic("Lambda", Lambda, dims=("indicators", "latent"))

        latent_dim = len(coords["latent"])

        # Latent factor covariance via LKJ-Cholesky
        sd_dist = pm.Exponential.dist(1.0, shape=latent_dim)
        chol, _, _ = pm.LKJCholeskyCov(
            "chol_cov", n=latent_dim, eta=2, sd_dist=sd_dist, compute_corr=True
        )
        gamma = pm.MvNormal("gamma", 0, chol=chol, dims=("obs", "latent"))

        # Structural regression coefficients (B matrix)
        coefs = pm.Normal(
            "mu_betas", [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], dims="paths"
        )
        zeros = pt.zeros((4, 4))
        # dysfunctional ~ constructive
        zeros = pt.set_subtensor(
            zeros[LATENT_ORDER_LKUP["dysfunctional"], LATENT_ORDER_LKUP["constructive"]],
            coefs[0],
        )
        # well being ~ dysfunctional
        zeros = pt.set_subtensor(
            zeros[LATENT_ORDER_LKUP["well being"], LATENT_ORDER_LKUP["dysfunctional"]],
            coefs[1],
        )
        # well being ~ constructive
        zeros = pt.set_subtensor(
            zeros[LATENT_ORDER_LKUP["well being"], LATENT_ORDER_LKUP["constructive"]],
            coefs[2],
        )
        # satisfaction ~ well being
        zeros = pt.set_subtensor(
            zeros[LATENT_ORDER_LKUP["satisfaction"], LATENT_ORDER_LKUP["well being"]],
            coefs[3],
        )
        # satisfaction ~ dysfunction
        zeros = pt.set_subtensor(
            zeros[LATENT_ORDER_LKUP["satisfaction"], LATENT_ORDER_LKUP["dysfunctional"]],
            coefs[4],
        )
        # satisfaction ~ constructive
        B = pt.set_subtensor(
            zeros[LATENT_ORDER_LKUP["satisfaction"], LATENT_ORDER_LKUP["constructive"]],
            coefs[5],
        )
        B = pm.Deterministic("B_", B, dims=("latent", "latent1"))

        I = pt.eye(latent_dim)
        # Clean Causal Influence of Shocks
        eta = pm.Deterministic(
            "eta",
            pt.slinalg.solve(I - B + 1e-8 * I, gamma.T).T,
            dims=("obs", "latent"),
        )
        # Influence of Exogenous indicator variables
        mu = pt.dot(eta, Lambda.T)

        # Error Terms
        sds = pm.InverseGamma("Psi", 5, 10, dims="indicators")
        Psi_cov = pt.diag(sds)
        _ = pm.MvNormal(
            "likelihood",
            mu=mu,
            cov=Psi_cov,
            observed=observed_data,
            dims=("obs", "indicators"),
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
