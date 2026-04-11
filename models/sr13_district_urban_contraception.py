"""
Model: District-Urban Multilevel Contraception Model
Source: pymc-examples/examples/statistical_rethinking_lectures/13-Multilevel_Adventures.ipynb, Section: "Varying Slopes"
Authors: Dustin Stansbury
Description: Multilevel Bernoulli model of contraception use in Bangladesh, with district-level
    varying intercepts and varying slopes for urban/rural effect. Both intercepts and slopes
    use non-centered parameterization for numerical stability.

Changes from original:
- Loaded data from npz instead of using utils.load_data
- Removed sampling and plotting code

Benchmark results:
- Original:  logp = -1456.4950, grad norm = 215.6490, 46.1 us/call (100000 evals)
- Frozen:    logp = -1456.4950, grad norm = 215.6490, 45.5 us/call (100000 evals)
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

    # District 54 has no data but we create its dim
    DISTRICT = np.arange(1, 62).astype(int)

    # Factorize urban
    unique_urban = np.sort(np.unique(urban))
    URBAN_CODED = np.searchsorted(unique_urban, urban)

    with pm.Model(coords={"district": DISTRICT}) as model:
        urban_ = pm.Data("urban", URBAN_CODED)

        # Priors - District offset
        alpha_bar = pm.Normal("alpha_bar", 0, 1)
        sigma = pm.Exponential("sigma", 1)

        # Uncentered parameterization for intercepts
        z_alpha = pm.Normal("z_alpha", 0, 1, dims="district")
        alpha = alpha_bar + z_alpha * sigma

        # District / urban interaction
        beta_bar = pm.Normal("beta_bar", 0, 1)
        tau = pm.Exponential("tau", 1)

        # Uncentered parameterization for slopes
        z_beta = pm.Normal("z_beta", 0, 1, dims="district")
        beta = beta_bar + z_beta * tau

        # Record p(contraceptive)
        p_C = pm.Deterministic("p_C", pm.math.invlogit(alpha + beta))
        p_C_urban = pm.Deterministic(
            "p_C_urban", pm.math.invlogit(alpha + beta)
        )
        p_C_rural = pm.Deterministic(
            "p_C_rural", pm.math.invlogit(alpha)
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
