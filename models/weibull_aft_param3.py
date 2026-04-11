"""
Model: Weibull AFT Parameterization 3 (Gumbel)
Source: pymc-examples/examples/survival_analysis/weibull_aft.ipynb, Section: "Parameterization 3"
Authors: Junpeng Lao, George Ho, Chris Fonnesbeck
Description: Gumbel-based parameterization of Weibull accelerated failure time model.
    Models log-transformed survival times with a Gumbel distribution instead of
    modeling the survival function directly. Right-censored observations handled
    via pm.Censored.

Changes from original:
- Saved sampled flchain data (500 observations, seed=8927) as .npz file.
- Removed sampling and plotting code.

Benchmark results:
- Original:  logp = -6849.4780, grad norm = 6838.4437, 21.1 us/call (100000 evals)
- Frozen:    logp = -6849.4780, grad norm = 6838.4437, 21.2 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm

def build_model():
    data = np.load(Path(__file__).parent / "data" / "flchain_500.npz")
    y = data["y"]
    censored = data["censored"]

    logtime = np.log(y)

    # If censored then observed event time else maximum time
    right_censored = [x if x > 0 else np.max(logtime) for x in logtime * censored]

    with pm.Model() as model:
        s = pm.HalfNormal("s", tau=3.0)
        gamma = pm.Normal("gamma", mu=0, sigma=5)

        latent = pm.Gumbel.dist(mu=gamma, beta=s)
        y_obs = pm.Censored("Censored_likelihood", latent, upper=right_censored, observed=logtime)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
