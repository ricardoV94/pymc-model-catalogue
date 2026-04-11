"""
Model: Full Luxury Bayes Height-Weight Model
Source: pymc-examples/examples/statistical_rethinking_lectures/04-Categories_&_Curves.ipynb, Section: "Full Luxury Bayes"
Authors: Dustin Stansbury
Description: Joint model of height and weight for adults, stratified by sex. Models both
    height and weight processes simultaneously, allowing sex-specific intercepts and slopes.

Changes from original:
- Loaded and filtered data inline instead of using utils.load_data and pd.factorize
- Removed sampling and plotting code

Benchmark results:
- Original:  logp = -4143.7259, grad norm = 1628.9474, 9.8 us/call (100000 evals)
- Frozen:    logp = -4143.7259, grad norm = 1628.9474, 8.4 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data = np.load(Path(__file__).parent / "data" / "sr_howell.npz")
    height = data["height"]
    weight = data["weight"]
    age = data["age"]
    male = data["male"]

    # Filter to adults
    adults = age >= 18
    H = height[adults]
    W = weight[adults]
    M = male[adults]

    # Factorize sex: male=1 -> "M" (index 0), male=0 -> "F" (index 1)
    sex_labels = np.array(["M" if s else "F" for s in M])
    unique_sex = []
    sex_id = np.empty(len(sex_labels), dtype=int)
    for i, s in enumerate(sex_labels):
        if s not in unique_sex:
            unique_sex.append(s)
        sex_id[i] = unique_sex.index(s)
    SEX = unique_sex

    with pm.Model(coords={"SEX": SEX}) as model:
        S = pm.Data("S", sex_id)
        H_ = pm.Data("H", H)
        Hbar = pm.Data("Hbar", H.mean())

        # Height Model
        tau = pm.Uniform("tau", 0, 10)
        h = pm.Normal("h", 160, 10, dims="SEX")
        nu = h[S]
        pm.Normal("H_obs", nu, tau, observed=H)

        # Weight Model
        alpha = pm.Normal("alpha", 60, 10, dims="SEX")
        beta = pm.Uniform("beta", 0, 1, dims="SEX")
        sigma = pm.Uniform("sigma", 0, 10)
        mu = alpha[S] + beta[S] * (H_ - Hbar)
        pm.Normal("W_obs", mu, sigma, observed=W)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
