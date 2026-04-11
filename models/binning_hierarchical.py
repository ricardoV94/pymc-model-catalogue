"""
Model: Hierarchical Gaussian Parameter Estimation from Binned Data
Source: pymc-examples/examples/case_studies/binning.ipynb, Section: "Example 5: Hierarchical estimation"
Authors: Eric Ma, Benjamin T. Vincent
Description: Hierarchical model estimating population-level Gaussian parameters from
    two studies with different bin cutpoints. Uses non-centered parameterization
    for study-level means and Gamma priors for study-level standard deviations.

Changes from original:
- Inlined simulated data generation with fixed random seed
- Precomputed bin counts inline
- Removed sampling, plotting, and comparison code

Benchmark results:
- Original:  logp = -1646.4837, grad norm = 2859.8275, 7.2 us/call (100000 evals)
- Frozen:    logp = -1646.4837, grad norm = 2859.8275, 7.1 us/call (100000 evals)
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    # Simulated data: two studies from same Gaussian population
    rng = np.random.default_rng(1234)
    true_mu, true_sigma = -2, 2
    x1 = rng.normal(loc=true_mu, scale=true_sigma, size=1500)
    x2 = rng.normal(loc=true_mu, scale=true_sigma, size=2000)

    # Different cutpoints per study
    d1 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    d2 = np.array([-5.0, -3.5, -2.0, -0.5, 1.0, 2.5])

    # Compute bin counts
    def data_to_bincounts(data, cutpoints):
        bins = np.digitize(data, bins=cutpoints)
        counts = np.bincount(bins, minlength=len(cutpoints) + 1)
        return counts

    c1 = data_to_bincounts(x1, d1)
    c2 = data_to_bincounts(x2, d2)

    coords = {
        "study": np.array([0, 1]),
        "bin1": np.arange(len(c1)),
        "bin2": np.arange(len(c2)),
    }

    with pm.Model(coords=coords) as model:
        # Population level priors
        mu_pop_mean = pm.Normal("mu_pop_mean", 0.0, 1.0)
        mu_pop_variance = pm.HalfNormal("mu_pop_variance", sigma=1)

        # Study level priors (non-centered)
        x = pm.Normal("x", dims="study")
        mu = pm.Deterministic("mu", x * mu_pop_variance + mu_pop_mean, dims="study")

        sigma = pm.Gamma("sigma", alpha=2, beta=1, dims="study")

        # Study 1
        probs1 = pm.math.exp(pm.logcdf(pm.Normal.dist(mu=mu[0], sigma=sigma[0]), d1))
        probs1 = pt.extra_ops.diff(pm.math.concatenate([np.array([0]), probs1, np.array([1])]))
        probs1 = pm.Deterministic("normal1_cdf", probs1, dims="bin1")

        # Study 2
        probs2 = pm.math.exp(pm.logcdf(pm.Normal.dist(mu=mu[1], sigma=sigma[1]), d2))
        probs2 = pt.extra_ops.diff(pm.math.concatenate([np.array([0]), probs2, np.array([1])]))
        probs2 = pm.Deterministic("normal2_cdf", probs2, dims="bin2")

        # Likelihood
        pm.Multinomial("counts1", p=probs1, n=c1.sum(), observed=c1, dims="bin1")
        pm.Multinomial("counts2", p=probs2, n=c2.sum(), observed=c2, dims="bin2")

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
