"""
Model: German Credit — hierarchical logistic regression (LogNormal-centred scales)
Source: Gorinova, Moore & Hoffman, "Automatic Reparameterisation of Probabilistic
    Programs" (arXiv:1906.03028, NeurIPS 2019), Section 6 (German credit).
    Reference implementation: https://github.com/mgorinova/autoreparam
Authors: Maria I. Gorinova, Dave Moore, Matthew D. Hoffman
Description: Logistic regression on the UCI Statlog German credit data with a
    hierarchical prior on the per-coefficient scales. This produces a "stacked"
    funnel: a global funnel (overall_log_scale -> beta_log_scales) on top of a
    local funnel (beta_log_scales -> beta). In the paper this is the headline
    case where BOTH fully-centred and fully-non-centred HMC perform poorly and a
    mixed/partial parameterisation wins.

Data: design matrix X (1000 x 62: constant + standardized numerics + one-hot
    categoricals) and binary label y, preprocessed by
    scripts/prep_autoreparam_data.py replicating autoreparam's
    load_german_credit_data + in-model one-hot encoding.

Changes from original:
- Loaded preprocessed design matrix from .npz instead of reading german.data.
- Removed sampling/plotting code.

Benchmark results:
- Original:  logp = -810.3171, grad norm = 578.7297, 110.2 us/call (87544 evals)
- Frozen:    logp = -810.3171, grad norm = 578.7297, 105.5 us/call (100000 evals)
"""

import numpy as np
import pymc as pm
from pathlib import Path


def build_model():
    data = np.load(Path(__file__).parent / "data" / "german_credit.npz")
    X = data["X"]
    y = data["y"]
    num_features = X.shape[1]

    with pm.Model() as model:
        overall_log_scale = pm.Normal("overall_log_scale", mu=0.0, sigma=10.0)
        beta_log_scales = pm.Normal(
            "beta_log_scales", mu=overall_log_scale, sigma=1.0, shape=num_features
        )
        beta = pm.Normal(
            "beta", mu=0.0, sigma=pm.math.exp(beta_log_scales), shape=num_features
        )
        logits = pm.math.dot(X, beta)
        pm.Bernoulli("y", logit_p=logits, observed=y)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
