"""
Model: Bayesian Estimation Supersedes the T-Test (BEST)
Source: pymc-examples/examples/case_studies/BEST.ipynb, Section: "Example: Drug trial evaluation"
Authors: Andrew Straw, Thomas Wiecki, Chris Fonnesbeck, Andres Suarez
Description: Robust Bayesian estimation for comparing two groups using Student-t likelihoods,
    with shared degrees-of-freedom and separate means/standard deviations per group.

Changes from original:
- Inlined IQ data arrays directly
- Removed xarray Dataset construction used only for plotting
- Removed sampling and plotting code

Benchmark results:
- Original:  logp = -270.3988, grad norm = 15.8755, 5.6 us/call (100000 evals)
- Frozen:    logp = -270.3988, grad norm = 15.8755, 6.3 us/call (100000 evals)
"""

import numpy as np
import pymc as pm

def build_model():
    # fmt: off
    iq_drug = np.array([
        101, 100, 102, 104, 102, 97, 105, 105, 98, 101, 100, 123, 105, 103,
        100, 95, 102, 106, 109, 102, 82, 102, 100, 102, 102, 101, 102, 102,
        103, 103, 97, 97, 103, 101, 97, 104, 96, 103, 124, 101, 101, 100,
        101, 101, 104, 100, 101
    ])

    iq_placebo = np.array([
        99, 101, 100, 101, 102, 100, 97, 101, 104, 101, 102, 102, 100, 105,
        88, 101, 100, 104, 100, 100, 100, 101, 102, 103, 97, 101, 101, 100,
        101, 99, 101, 100, 100, 101, 100, 99, 101, 100, 102, 99, 100, 99
    ])
    # fmt: on

    # Pooled empirical statistics for prior hyperparameters
    all_data = np.concatenate([iq_drug, iq_placebo])
    mu_m = float(all_data.mean())
    mu_s = float(all_data.std()) * 2

    sigma_low = 10**-1
    sigma_high = 10

    with pm.Model() as model:
        group1_mean = pm.Normal("group1_mean", mu=mu_m, sigma=mu_s)
        group2_mean = pm.Normal("group2_mean", mu=mu_m, sigma=mu_s)

        group1_std = pm.Uniform("group1_std", lower=sigma_low, upper=sigma_high)
        group2_std = pm.Uniform("group2_std", lower=sigma_low, upper=sigma_high)

        nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29.0)
        nu = pm.Deterministic("nu", nu_minus_one + 1)
        nu_log10 = pm.Deterministic("nu_log10", np.log10(nu))

        lambda_1 = group1_std**-2
        lambda_2 = group2_std**-2
        group1 = pm.StudentT("drug", nu=nu, mu=group1_mean, lam=lambda_1, observed=iq_drug)
        group2 = pm.StudentT("placebo", nu=nu, mu=group2_mean, lam=lambda_2, observed=iq_placebo)

        diff_of_means = pm.Deterministic("difference of means", group1_mean - group2_mean)
        diff_of_stds = pm.Deterministic("difference of stds", group1_std - group2_std)
        effect_size = pm.Deterministic(
            "effect size", diff_of_means / np.sqrt((group1_std**2 + group2_std**2) / 2)
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip

if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
