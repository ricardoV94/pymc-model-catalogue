"""
Model: Gamma likelihood with Exponential priors
Source: pymc-examples/examples/variational_inference/variational_api_quickstart.ipynb, Section: "Distributional Approximations"
Authors: Maxim Kochurov, Chris Fonnesbeck
Description: Simple Gamma likelihood with Exponential(0.1) priors on shape (alpha) and rate (beta),
    fitted to 200 samples drawn from Gamma(2, 0.5). Used as the intro example in the VI API tour.

Changes from original:
- Inlined data generation with fixed seed (np.random.seed(42)) as in the notebook
- Removed VI/NUTS fitting, callback, and plotting code

Benchmark results:
- Original:  logp = -342.8209, grad norm = 340.4372, 2.1 us/call (100000 evals)
- Frozen:    logp = -342.8209, grad norm = 340.4372, 2.4 us/call (100000 evals)
"""

import numpy as np
import pymc as pm


def build_model():
    np.random.seed(42)
    gamma_data = np.random.gamma(2, 0.5, size=200)

    with pm.Model() as model:
        alpha = pm.Exponential("alpha", 0.1)
        beta = pm.Exponential("beta", 0.1)

        y = pm.Gamma("y", alpha, beta, observed=gamma_data)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
