"""
Model: Weibull AFT Parameterization 2
Source: pymc-examples/examples/survival_analysis/weibull_aft.ipynb, Section: "Parameterization 2"
Authors: Junpeng Lao, George Ho, Chris Fonnesbeck
Description: Stan-inspired parameterization of Weibull accelerated failure time model.
    Uses Gamma prior on shape parameter r and Normal prior on alpha, with beta
    derived as exp(-alpha/r). Right-censored observations handled via pm.Censored.

Changes from original:
- Saved sampled flchain data (500 observations, seed=8927) as .npz file.
- Removed sampling and plotting code.

Benchmark results:
- Original:  logp = -376.4659, grad norm = 196.8491, 69.9 us/call (100000 evals)
- Frozen:    logp = -376.4659, grad norm = 196.8491, 69.8 us/call (100000 evals)
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
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        r = pm.Gamma("r", alpha=2, beta=1)
        beta = pm.Deterministic("beta", pt.exp(-alpha / r))
        beta_backtransformed = pm.Deterministic("beta_backtransformed", beta * np.max(y))

        latent = pm.Weibull.dist(alpha=r, beta=beta)
        y_obs = pm.Censored("Censored_likelihood", latent, upper=right_censored, observed=y_norm)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
