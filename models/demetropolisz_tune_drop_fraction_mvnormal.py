"""
Model: 10-Dimensional MvNormal Target (DEMetropolisZ sampler tuning)
Source: pymc-examples/examples/samplers/DEMetropolisZ_tune_drop_fraction.ipynb, Section: "Target Distribution"
Authors: Michael Osthege, Greg Brunkhorst
Description: A 10-dimensional multivariate normal target density with zero means and
    a covariance matrix where the first 5 dimensions have variances 1..5 with some
    added correlation, used as a toy target to explore DEMetropolisZ tuning parameters.

Changes from original:
- Inlined gen_mvnormal_params helper into build_model
- Removed sampling/plotting code and target_sample draw

Benchmark results:
- Original:  logp = -9.9941, grad norm = 0.0000, 1.9 us/call (100000 evals)
- Frozen:    logp = -9.9941, grad norm = 0.0000, 1.9 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    D = 10
    # means=zero
    mu = np.zeros(D)
    # sigmas=1 to start
    cov = np.eye(D)
    # add variance and covariance in the first 5 dimensions
    cov[:5, :5] = np.array(
        [
            [1, 0.5, 0, 0, 0],
            [0.5, 2, 2, 0, 0],
            [0, 2, 3, 0, 0],
            [0, 0, 0, 4, 4],
            [0, 0, 0, 4, 5],
        ]
    )

    with pm.Model() as model:
        x = pm.MvNormal("x", mu=mu, cov=cov, shape=(len(mu),))

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
