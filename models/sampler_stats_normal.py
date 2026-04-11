"""
Model: Standard Normal (Sampler Statistics Demo)
Source: pymc-examples/examples/diagnostics_and_criticism/sampler-stats.ipynb, Section: "Sampler Statistics"
Authors: Meenal Jhajharia, Christian Luhmann
Description: A simple 10-dimensional standard normal model used to demonstrate and
    explain NUTS sampler statistics and diagnostics.

Changes from original:
- Removed sampling, diagnostics inspection, and plotting code

Benchmark results:
- Original:  logp = -9.1894, grad norm = 0.0000, 2.5 us/call (100000 evals)
- Frozen:    logp = -9.1894, grad norm = 0.0000, 2.7 us/call (100000 evals)
"""

import numpy as np
import pymc as pm

def build_model():
    with pm.Model() as model:
        mu1 = pm.Normal("mu1", mu=0, sigma=1, shape=10)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
