"""
Model: Confirmatory Factor Analysis with Structural Equation Model (Indirect Effects)
Source: pymc-examples/examples/case_studies/CFA_SEM.ipynb, Section: "Indirect Effects SEM"
Authors: Nathaniel Forde
Description: Full SEM with five latent factors (academic self-efficacy, social self-efficacy,
    friend support, parent support, life satisfaction) where SE and LS are regressed on
    support factors with correlated residuals, and factor loadings use first-fixed-to-1
    identification.

Changes from original:
- Saved observed data to .npz
- Inlined the make_indirect_sem function with specific priors (SEM0 configuration)
- Removed sampling, plotting, and model comparison code

Benchmark results:
- Original:  logp = -12207.7977, grad norm = 1449.9846, 84.1 us/call (100000 evals)
- Frozen:    logp = -12207.7977, grad norm = 1449.9846, 79.1 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    data_path = Path(__file__).parent / "data" / "sem_data.npz"
    data = np.load(data_path)
    observed_data = data["observed_data"]
    n_obs = int(data["n_obs"])

    drivers = [
        "se_acad_p1", "se_acad_p2", "se_acad_p3",
        "se_social_p1", "se_social_p2", "se_social_p3",
        "sup_friends_p1", "sup_friends_p2", "sup_friends_p3",
        "sup_parents_p1", "sup_parents_p2", "sup_parents_p3",
        "ls_p1", "ls_p2", "ls_p3",
    ]

    coords = {
        "obs": list(range(n_obs)),
        "indicators": drivers,
        "indicators_1": ["se_acad_p1", "se_acad_p2", "se_acad_p3"],
        "indicators_2": ["se_social_p1", "se_social_p2", "se_social_p3"],
        "indicators_3": ["sup_friends_p1", "sup_friends_p2", "sup_friends_p3"],
        "indicators_4": ["sup_parents_p1", "sup_parents_p2", "sup_parents_p3"],
        "indicators_5": ["ls_p1", "ls_p2", "ls_p3"],
        "latent": ["SUP_F", "SUP_P"],
        "latent1": ["SUP_F", "SUP_P"],
        "latent_regression": [
            "SUP_F->SE_ACAD", "SUP_P->SE_ACAD",
            "SUP_F->SE_SOC", "SUP_P->SE_SOC",
        ],
        "regression": ["SE_ACAD", "SE_SOCIAL", "SUP_F", "SUP_P"],
    }

    # Priors configuration (SEM0)
    gamma_prior = 1
    lambda_prior = [1, 2]
    beta_r_prior = 3
    beta_r2_prior = 3
    eta_prior = 1

    obs_idx = list(range(n_obs))

    with pm.Model(coords=coords) as model:
        Psi = pm.Gamma("Psi", gamma_prior, 0.5, dims="indicators")

        # Factor loadings with first-fixed-to-1 identification
        lambdas_ = pm.Normal(
            "lambdas_1", lambda_prior[0], lambda_prior[1], dims="indicators_1"
        )
        lambdas_1 = pm.Deterministic(
            "lambdas1", pt.set_subtensor(lambdas_[0], 1), dims="indicators_1"
        )
        lambdas_ = pm.Normal(
            "lambdas_2", lambda_prior[0], lambda_prior[1], dims="indicators_2"
        )
        lambdas_2 = pm.Deterministic(
            "lambdas2", pt.set_subtensor(lambdas_[0], 1), dims="indicators_2"
        )
        lambdas_ = pm.Normal(
            "lambdas_3", lambda_prior[0], lambda_prior[1], dims="indicators_3"
        )
        lambdas_3 = pm.Deterministic(
            "lambdas3", pt.set_subtensor(lambdas_[0], 1), dims="indicators_3"
        )
        lambdas_ = pm.Normal(
            "lambdas_4", lambda_prior[0], lambda_prior[1], dims="indicators_4"
        )
        lambdas_4 = pm.Deterministic(
            "lambdas4", pt.set_subtensor(lambdas_[0], 1), dims="indicators_4"
        )
        lambdas_ = pm.Normal(
            "lambdas_5", lambda_prior[0], lambda_prior[1], dims="indicators_5"
        )
        lambdas_5 = pm.Deterministic(
            "lambdas5", pt.set_subtensor(lambdas_[0], 1), dims="indicators_5"
        )

        kappa = 0
        sd_dist = pm.Gamma.dist(gamma_prior, 0.5, shape=2)
        chol, _, _ = pm.LKJCholeskyCov(
            "chol_cov", n=2, eta=eta_prior, sd_dist=sd_dist, compute_corr=True
        )
        cov = pm.Deterministic("cov", chol.dot(chol.T), dims=("latent", "latent1"))
        ksi = pm.MvNormal("ksi", kappa, chol=chol, dims=("obs", "latent"))

        # Regression Components
        beta_r = pm.Normal("beta_r", 0, beta_r_prior, dims="latent_regression")
        beta_r2 = pm.Normal("beta_r2", 0, beta_r2_prior, dims="regression")
        sd_dist1 = pm.Gamma.dist(1, 0.5, shape=2)
        resid_chol, _, _ = pm.LKJCholeskyCov(
            "resid_chol", n=2, eta=3, sd_dist=sd_dist1, compute_corr=True
        )
        _ = pm.Deterministic("resid_cov", resid_chol.dot(resid_chol.T))
        sigmas_resid = pm.MvNormal("sigmas_resid", 1, chol=resid_chol)
        sigma_regr = pm.HalfNormal("sigma_regr", 1)

        # SE_ACAD ~ SUP_FRIENDS + SUP_PARENTS
        regression_se_acad = pm.Normal(
            "regr_se_acad",
            beta_r[0] * ksi[obs_idx, 0] + beta_r[1] * ksi[obs_idx, 1],
            pm.math.abs(sigmas_resid[0]),
        )
        # SE_SOCIAL ~ SUP_FRIENDS + SUP_PARENTS
        regression_se_social = pm.Normal(
            "regr_se_social",
            beta_r[2] * ksi[obs_idx, 0] + beta_r[3] * ksi[obs_idx, 1],
            pm.math.abs(sigmas_resid[1]),
        )

        # LS ~ SE_ACAD + SE_SOCIAL + SUP_FRIEND + SUP_PARENTS
        regression = pm.Normal(
            "regr",
            beta_r2[0] * regression_se_acad
            + beta_r2[1] * regression_se_social
            + beta_r2[2] * ksi[obs_idx, 0]
            + beta_r2[3] * ksi[obs_idx, 1],
            sigma_regr,
        )

        tau = pm.Normal("tau", 3, 0.5, dims="indicators")
        m0 = tau[0] + regression_se_acad * lambdas_1[0]
        m1 = tau[1] + regression_se_acad * lambdas_1[1]
        m2 = tau[2] + regression_se_acad * lambdas_1[2]
        m3 = tau[3] + regression_se_social * lambdas_2[0]
        m4 = tau[4] + regression_se_social * lambdas_2[1]
        m5 = tau[5] + regression_se_social * lambdas_2[2]
        m6 = tau[6] + ksi[obs_idx, 0] * lambdas_3[0]
        m7 = tau[7] + ksi[obs_idx, 0] * lambdas_3[1]
        m8 = tau[8] + ksi[obs_idx, 0] * lambdas_3[2]
        m9 = tau[9] + ksi[obs_idx, 1] * lambdas_4[0]
        m10 = tau[10] + ksi[obs_idx, 1] * lambdas_4[1]
        m11 = tau[11] + ksi[obs_idx, 1] * lambdas_4[2]
        m12 = tau[12] + regression * lambdas_5[0]
        m13 = tau[13] + regression * lambdas_5[1]
        m14 = tau[14] + regression * lambdas_5[2]

        mu = pm.Deterministic(
            "mu",
            pm.math.stack(
                [m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14]
            ).T,
        )
        _ = pm.Normal(
            "likelihood", mu, Psi, observed=observed_data
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
