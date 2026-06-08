"""
Model: Election '88 — hierarchical logistic regression
Source: Gorinova, Moore & Hoffman, "Automatic Reparameterisation of Probabilistic
    Programs" (arXiv:1906.03028, NeurIPS 2019), Section 6 (Election '88);
    data from Gelman & Hill (2007). Reference implementation:
    https://github.com/mgorinova/autoreparam
Authors: Maria I. Gorinova, Dave Moore, Matthew D. Hoffman
Description: Logistic model of 1988 US presidential vote intention from survey
    responses, with a state-level varying intercept (hierarchical Normal) and
    individual demographic covariates (black, female). The varying-intercept
    hierarchy gives the funnel geometry the paper studies.

Data: black, female, state (0-based index, 51 states), y (binary), N = 11566,
    preprocessed by scripts/prep_autoreparam_data.py.

Changes from original:
- Loaded preprocessed arrays from .npz; state indices converted to 0-based.
- Removed sampling/plotting code.

Benchmark results:
- Original:  logp = -8083.6000, grad norm = 450.3396, 583.4 us/call (33707 evals)
- Frozen:    logp = -8083.6000, grad norm = 450.3396, 633.4 us/call (32089 evals)
"""

import numpy as np
import pymc as pm
from pathlib import Path


def build_model():
    data = np.load(Path(__file__).parent / "data" / "election88.npz")
    black = data["black"]
    female = data["female"]
    state = data["state"]
    y = data["y"]
    n_state = int(state.max()) + 1

    with pm.Model() as model:
        mua = pm.Normal("mua", mu=0.0, sigma=100.0)
        log_sigma_a = pm.Normal("log_sigma_a", mu=0.0, sigma=10.0)
        a = pm.Normal("a", mu=mua, sigma=pm.math.exp(log_sigma_a), shape=n_state)
        b1 = pm.Normal("b1", mu=0.0, sigma=100.0)  # black
        b2 = pm.Normal("b2", mu=0.0, sigma=100.0)  # female
        logits = a[state] + b1 * black + b2 * female
        pm.Bernoulli("y", logit_p=logits, observed=y)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
