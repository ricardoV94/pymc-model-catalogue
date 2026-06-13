"""
Model: German Credit — hierarchical logistic regression (Gamma scales)
Source: Gorinova, Moore & Hoffman, "Automatic Reparameterisation of Probabilistic
    Programs" (arXiv:1906.03028, NeurIPS 2019), Section 6 (German credit).
    Reference implementation: https://github.com/mgorinova/autoreparam
    (get_german_credit_gammascale).
Authors: Maria I. Gorinova, Dave Moore, Matthew D. Hoffman
Description: Logistic regression on the UCI Statlog German credit data with
    heavy-tailed per-coefficient scales. Unlike the LogNormal variant, the
    per-feature scales come from an independent Gamma(0.5, 0.5) prior (a sparse /
    shrinkage-like prior), so there is no global-scale funnel — only the local
    beta <-> scale funnel, with heavy tails. In the reference implementation
    `beta_log_scales = log(s)` with `s ~ Gamma(0.5, 0.5)` and the coefficient
    scale is `exp(overall_log_scale + beta_log_scales) = exp(overall_log_scale)*s`.

Data: design matrix X (1000 x 62: constant + standardized numerics + one-hot
    categoricals) and binary label y, preprocessed by
    scripts/prep_autoreparam_data.py.

Changes from original:
- Loaded preprocessed design matrix from .npz instead of reading german.data.
- `s ~ Gamma(0.5, 0.5)` directly (NUTS samples it on the log scale), equivalent
    to the reference's `beta_log_scales = Invert(Exp)(Gamma(0.5, 0.5))`.
- Removed sampling/plotting code.

Benchmark results:
- Original:  logp = -841.3171, grad norm = 582.0413, 91.0 us/call (60531 evals)
- Frozen:    logp = -841.3171, grad norm = 582.0413, 90.2 us/call (100000 evals)
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
        beta_scales = pm.Gamma("beta_scales", alpha=0.5, beta=0.5, shape=num_features)
        beta = pm.Normal(
            "beta",
            mu=0.0,
            sigma=pm.math.exp(overall_log_scale) * beta_scales,
            shape=num_features,
        )
        logits = pm.math.dot(X, beta)
        pm.Bernoulli("y", logit_p=logits, observed=y)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
