"""
Model: Weibull AFT Parameterization 1
Source: pymc-examples/examples/survival_analysis/weibull_aft.ipynb, Section: "Parameterization 1"
Authors: Junpeng Lao, George Ho, Chris Fonnesbeck
Description: Intuitive parameterization of Weibull accelerated failure time model using
    pm.Censored. Models normalized survival times with Weibull distribution, where
    alpha and beta are derived from normal priors via exponential transforms.
    Right-censored observations handled via pm.Censored upper bounds.

Changes from original:
- Saved sampled flchain data (500 observations, seed=8927) as .npz file.
- Removed sampling and plotting code.

Benchmark results:
- Original:  logp = -358.7274, grad norm = 231.4427, 66.4 us/call (100000 evals)
- Frozen:    logp = -358.7274, grad norm = 231.4427, 66.1 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm
import pytensor.tensor as pt

def build_model():
    data = np.load(Path(__file__).parent / "data" / "flchain_500.npz")
    y = data["y"]
    censored = data["censored"]

    # Normalize event time between 0 and 1
    y_norm = y / np.max(y)

    # If censored then observed event time else maximum time
    right_censored = [x if x > 0 else np.max(y_norm) for x in y_norm * censored]

    with pm.Model() as model:
        alpha_sd = 1.0

        mu = pm.Normal("mu", mu=0, sigma=1)
        alpha_raw = pm.Normal("a0", mu=0, sigma=0.1)
        alpha = pm.Deterministic("alpha", pt.exp(alpha_sd * alpha_raw))
        beta = pm.Deterministic("beta", pt.exp(mu / alpha))
        beta_backtransformed = pm.Deterministic("beta_backtransformed", beta * np.max(y))

        latent = pm.Weibull.dist(alpha=alpha, beta=beta)
        y_obs = pm.Censored("Censored_likelihood", latent, upper=right_censored, observed=y_norm)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
