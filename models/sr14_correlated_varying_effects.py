"""
Model: Correlated Varying Effects (LKJ) Contraception Model
Source: pymc-examples/examples/statistical_rethinking_lectures/14-Correlated_Features.ipynb, Section: "Correlated Features"
Authors: Dustin Stansbury
Description: Multilevel Bernoulli model of contraception use with correlated district-level
    varying intercepts and slopes. Uses LKJCholeskyCov to model the correlation between
    district intercept (baseline contraception rate) and district slope (urban effect).
    Non-centered parameterization via Cholesky factor.

Changes from original:
- Loaded data from npz instead of using utils.load_data
- Removed sampling and plotting code

Benchmark results:
- Original:  logp = -1456.5846, grad norm = 215.6513, 49.0 us/call (100000 evals)
- Frozen:    logp = -1456.5846, grad norm = 215.6513, 47.5 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data = np.load(
        Path(__file__).parent / "data" / "sr_bangladesh.npz", allow_pickle=True
    )
    USES_CONTRACEPTION = data["use_contraception"].astype(int)
    district = data["district"]
    urban = data["urban"]

    # Factorize district
    unique_districts = np.unique(district)
    district_map = {d: i for i, d in enumerate(unique_districts)}
    DISTRICT_ID = np.array([district_map[d] for d in district])

    DISTRICT = np.arange(1, 62).astype(int)

    # Factorize urban
    unique_urban = np.sort(np.unique(urban))
    URBAN_ID = np.searchsorted(unique_urban, urban)

    N_CORRELATED_FEATURES = 2
    N_DISTRICTS = 61
    ETA = 4

    with pm.Model(coords={"district": DISTRICT}) as model:
        urban_ = pm.Data("urban", URBAN_ID)

        # Priors - Feature correlation
        chol, corr, stds = pm.LKJCholeskyCov(
            "Rho",
            eta=ETA,
            n=N_CORRELATED_FEATURES,
            sd_dist=pm.Exponential.dist(1, size=N_CORRELATED_FEATURES),
        )

        # Uncentered parameterization
        z = pm.Normal("z", 0, 1, shape=(N_DISTRICTS, N_CORRELATED_FEATURES))
        v = pm.Deterministic("v", chol.dot(z.T).T)

        alpha_bar = pm.Normal("alpha_bar", 0, 1)
        alpha = pm.Deterministic("alpha", alpha_bar + v[:, 0])

        beta_bar = pm.Normal("beta_bar", 0, 1)
        beta = pm.Deterministic("beta", beta_bar + v[:, 1])

        # Record values for reporting
        pm.Deterministic("feature_cov", chol.dot(chol.T))
        pm.Deterministic("feature_corr", corr)
        pm.Deterministic("feature_std", stds)
        pm.Deterministic(
            "p_C", pm.math.invlogit(alpha + beta), dims="district"
        )
        pm.Deterministic(
            "p_C_urban", pm.math.invlogit(alpha + beta), dims="district"
        )
        pm.Deterministic(
            "p_C_rural", pm.math.invlogit(alpha), dims="district"
        )

        # Likelihood
        p = pm.math.invlogit(
            alpha[DISTRICT_ID] + beta[DISTRICT_ID] * urban_
        )
        pm.Bernoulli("C", p=p, observed=USES_CONTRACEPTION)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
