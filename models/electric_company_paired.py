"""
Model: Electric Company — paired hierarchical Gaussian
Source: Gorinova, Moore & Hoffman, "Automatic Reparameterisation of Probabilistic
    Programs" (arXiv:1906.03028, NeurIPS 2019), Section 6 (Electric Company);
    data from Gelman & Hill (2007). Reference implementation:
    https://github.com/mgorinova/autoreparam
Authors: Maria I. Gorinova, Dave Moore, Matthew D. Hoffman
Description: Paired causal analysis of the effect of an educational TV show on
    192 classrooms across G=4 grades, arranged in P=96 treated/control pairs.
    A grade-level mean feeds a pair-level varying intercept (hierarchical Normal
    funnel); the treatment effect and observation noise vary by grade.

Data: grade, pair, grade_pair (all 0-based), treatment, y; N = 192, n_pair = 96,
    n_grade = 4, preprocessed by scripts/prep_autoreparam_data.py.

Changes from original:
- Loaded preprocessed arrays from .npz; index arrays converted to 0-based.
- Follows the reference implementation: grade mean mu ~ N(0,1) is scaled by 100
    so the pair intercept a_p ~ N(100*mu_{g[p]}, 1) can reach the post-test
    scale (~44-122); b_g ~ N(0,100); observation noise is exp(log_sigma_g),
    log_sigma_g ~ N(0,1). (The paper's Section 6 equations drop the 100 factor,
    which would leave the prior far from the data scale.)
- Removed sampling/plotting code.

Benchmark results:
- Original:  logp = -936462.1172, grad norm = 984987.8625, 11.3 us/call (100000 evals)
- Frozen:    logp = -936462.1172, grad norm = 984987.8625, 10.9 us/call (100000 evals)
"""

import numpy as np
import pymc as pm
from pathlib import Path


def build_model():
    data = np.load(Path(__file__).parent / "data" / "electric.npz")
    grade = data["grade"]
    pair = data["pair"]
    grade_pair = data["grade_pair"]
    treatment = data["treatment"]
    y = data["y"]
    n_grade = int(grade.max()) + 1
    n_pair = int(pair.max()) + 1

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0, shape=n_grade)
        a = pm.Normal("a", mu=100.0 * mu[grade_pair], sigma=1.0, shape=n_pair)
        b = pm.Normal("b", mu=0.0, sigma=100.0, shape=n_grade)
        log_sigma = pm.Normal("log_sigma", mu=0.0, sigma=1.0, shape=n_grade)
        y_hat = a[pair] + b[grade] * treatment
        pm.Normal("y", mu=y_hat, sigma=pm.math.exp(log_sigma[grade]), observed=y)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
