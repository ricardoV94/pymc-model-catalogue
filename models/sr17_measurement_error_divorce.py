"""
Model: Divorce Rate with Measurement Error in Both Marriage and Divorce
Source: pymc-examples/examples/statistical_rethinking_lectures/17-Measurement_and_Misclassification.ipynb, Section: "Full Measurement Error Model"
Authors: Dustin Stansbury
Description: Structural model of divorce rate incorporating measurement error in both
    divorce rate and marriage rate observations. True latent divorce and marriage rates
    are modeled, with observed values drawn from distributions centered on latent values
    with known standard errors.

Changes from original:
- Loaded data inline from npz instead of using utils.load_data
- Standardized data inline
- Removed sampling and plotting code

Benchmark results:
- Original:  logp = -591.3441, grad norm = 185.0391, 5.5 us/call (100000 evals)
- Frozen:    logp = -591.3441, grad norm = 185.0391, 6.0 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data = np.load(
        Path(__file__).parent / "data" / "sr_waffle_divorce.npz", allow_pickle=True
    )
    divorce_raw = data["Divorce"]
    marriage_raw = data["Marriage"]
    age_raw = data["MedianAgeMarriage"]
    divorce_se_raw = data["Divorce_SE"]
    marriage_se_raw = data["Marriage_SE"]
    state = data["Location"]

    # Standardize
    def standardize(x):
        c = x - np.nanmean(x)
        return c / np.nanstd(c)

    DIVORCE_STD = divorce_raw.std()
    DIVORCE = standardize(divorce_raw)
    MARRIAGE = standardize(marriage_raw)
    AGE = standardize(age_raw)
    DIVORCE_SE = divorce_se_raw / DIVORCE_STD

    MARRIAGE_STD = marriage_raw.std()
    MARRIAGE_SE = marriage_se_raw / MARRIAGE_STD

    STATE = state.tolist()
    coords = {"state": STATE}

    with pm.Model(coords=coords) as model:
        # Marriage Rate Model
        sigma_marriage = pm.Exponential("sigma_M", 1)
        alpha_marriage = pm.Normal("alpha_M", 0, 0.2)
        beta_AM = pm.Normal("beta_AM", 0, 0.5)

        # True marriage parameter
        mu_marriage = alpha_marriage + beta_AM * AGE
        M = pm.Normal("M", mu_marriage, sigma_marriage, dims="state")

        # Marriage measurement model
        pm.Normal("M_star", M, MARRIAGE_SE, observed=MARRIAGE)

        # Divorce Model
        sigma_divorce = pm.Exponential("sigma_D", 1)
        alpha_divorce = pm.Normal("alpha_D", 0, 0.2)
        beta_AD = pm.Normal("beta_AD", 0, 0.5)
        beta_MD = pm.Normal("beta_MD", 0, 0.5)

        # True divorce parameter
        STATE_ID = np.arange(len(STATE))
        mu_divorce = alpha_divorce + beta_AD * AGE + beta_MD * M[STATE_ID]
        D = pm.Normal("D", mu_divorce, sigma_divorce, dims="state")

        # Divorce measurement model
        pm.Normal("D_star", D, DIVORCE_SE, observed=DIVORCE)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
