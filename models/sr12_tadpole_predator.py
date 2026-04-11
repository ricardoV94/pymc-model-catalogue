"""
Model: Multilevel Tadpole Survival with Predator Effects
Source: pymc-examples/examples/statistical_rethinking_lectures/12-Multilevel_Models.ipynb, Section: "Predator Model"
Authors: Dustin Stansbury
Description: Multilevel Binomial model of reed frog tadpole survival across tanks with
    a predator covariate. Tank-specific intercepts are drawn from a shared Normal
    distribution with learned variance, providing partial pooling. Predator presence
    shifts survival probability on the log-odds scale.

Changes from original:
- Loaded data from npz instead of using utils.load_data
- Factorized predator variable inline
- Removed sampling, LOO-CV, and plotting code

Benchmark results:
- Original:  logp = -424.2154, grad norm = 78.8353, 3.4 us/call (100000 evals)
- Frozen:    logp = -424.2154, grad norm = 78.8353, 3.4 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data = np.load(
        Path(__file__).parent / "data" / "sr_reedfrogs.npz", allow_pickle=True
    )
    N_SURVIVED = data["surv"].astype(float)
    N_TRIALS = data["density"].astype(float)
    pred = data["pred"]

    N_TANKS = len(N_SURVIVED)

    # Factorize predator
    unique_pred = []
    pred_id = np.empty(len(pred), dtype=int)
    for i, p in enumerate(pred):
        if p not in unique_pred:
            unique_pred.append(p)
        pred_id[i] = unique_pred.index(p)

    TANK_ID = np.arange(N_TANKS)

    with pm.Model() as model:
        # Global Priors
        sigma = pm.Exponential("sigma", 1)
        alpha_bar = pm.Normal("alpha_bar", 0, 1.5)

        # Predator-specific prior
        beta_predator = pm.Normal("beta_predator", 0, 0.5)

        # Tank-specific prior
        alpha = pm.Normal("alpha", alpha_bar, sigma, shape=N_TANKS)

        # Record p_survived for visualization
        p_survived = pm.math.invlogit(
            alpha[TANK_ID] + beta_predator * pred_id
        )
        p_survived = pm.Deterministic("p_survived", p_survived)

        # Likelihood
        pm.Binomial(
            "survived", n=N_TRIALS, p=p_survived, observed=N_SURVIVED
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
