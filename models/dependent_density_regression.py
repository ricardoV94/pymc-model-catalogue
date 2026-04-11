"""
Model: Dependent Density Regression
Source: pymc-examples/examples/mixture_models/dependent_density_regression.ipynb, Section: full model
Authors: Austin Rochford
Description: A dependent density regression model for LIDAR data using a truncated
    Dirichlet process with probit stick-breaking weights and linear conditional
    component means. Uses NormalMixture likelihood with K=20 components, where
    mixture weights and component means depend on the predictor via probit and
    linear functions respectively.

Changes from original:
- Saved standardized LIDAR data to .npz file instead of loading via URL
- Removed sampling, posterior predictive, plotting, and animation code
- Removed Slice step specification (sampling configuration)

Benchmark results:
- Original:  logp = -654.4541, grad norm = 114.1663, 167.9 us/call (74911 evals)
- Frozen:    logp = -654.4541, grad norm = 114.1663, 175.8 us/call (73705 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm
import pytensor.tensor as pt

def build_model():
    data = np.load(Path(__file__).parent / "data" / "lidar.npz")
    std_range = data["std_range"]
    std_logratio = data["std_logratio"]

    N = len(std_range)
    K = 20

    def norm_cdf(z):
        return 0.5 * (1 + pt.erf(z / np.sqrt(2)))

    def stick_breaking(v):
        return v * pt.concatenate(
            [pt.ones_like(v[:, :1]), pt.extra_ops.cumprod(1 - v[:, :-1], axis=1)], axis=1
        )

    with pm.Model(coords={"N": np.arange(N), "K": np.arange(K) + 1}) as model:
        alpha = pm.Normal("alpha", 0, 5, dims="K")
        beta = pm.Normal("beta", 0, 5, dims="K")
        x = pm.Data("x", std_range, dims="N")
        v = norm_cdf(alpha + pt.outer(x, beta))
        w = pm.Deterministic("w", stick_breaking(v), dims=["N", "K"])

        gamma = pm.Normal("gamma", 0, 3, dims="K")
        delta = pm.Normal("delta", 0, 3, dims="K")
        mu = pm.Deterministic("mu", gamma + pt.outer(x, delta), dims=("N", "K"))

        sigma = pm.HalfNormal("sigma", 3, dims="K")
        y = pm.Data("y", std_logratio, dims="N")
        obs = pm.NormalMixture("obs", w, mu, sigma=sigma, observed=y, dims="N")

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
