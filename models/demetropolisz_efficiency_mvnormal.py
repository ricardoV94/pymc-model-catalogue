"""
Model: 10-Dimensional MvNormal Target (DEMetropolisZ efficiency comparison)
Source: pymc-examples/examples/samplers/DEMetropolisZ_EfficiencyComparison.ipynb, Section: "D-dimensional MvNormal Target Distribution and PyMC Model"
Authors: Michael Osthege, Greg Brunkhorst
Description: A 10-dimensional multivariate normal target density with zero means and
    a covariance matrix where the first 5 dimensions have variances 1..5 with some
    added correlation, used as a toy target for sampler efficiency comparisons.

Changes from original:
- Inlined gen_mvnormal_params helper into build_model
- Removed sampling/plotting code
- Extracted only the D=10 instance (the 50-D instance uses the same model structure)

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
    # sigma**2 = 1 to start
    cov = np.eye(D)
    # manually adjust the first 5 dimensions
    # sigma**2 in the first 5 dimensions = 1, 2, 3, 4, 5
    # with a little covariance added
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
