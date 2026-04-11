"""
Model: Beta-Binomial with Uniform Prior
Source: pymc-examples/examples/diagnostics_and_criticism/Bayes_factor.ipynb, Section: "Savage-Dickey Density Ratio"
Authors: Osvaldo Martin
Description: Beta-binomial coin-flip model with a uniform Beta(1,1) prior on the bias
    parameter, used for Bayes factor computation via the Savage-Dickey density ratio.

Changes from original:
- Inlined the small data array (100 coin flips)
- Removed sampling, plotting, and Bayes factor computation code

Benchmark results:
- Original:  logp = -70.7010, grad norm = 0.0000, 2.4 us/call (100000 evals)
- Frozen:    logp = -70.7010, grad norm = 0.0000, 2.6 us/call (100000 evals)
"""

import numpy as np
import pymc as pm

def build_model():
    y = np.repeat([1, 0], [50, 50])  # 50 "heads" and 50 "tails"

    with pm.Model() as model:
        a = pm.Beta("a", 1, 1)
        yl = pm.Bernoulli("yl", a, observed=y)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
