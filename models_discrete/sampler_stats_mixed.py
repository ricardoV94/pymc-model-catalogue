"""
Model: Mixed Bernoulli-Normal (Multiple Samplers Demo)
Source: pymc-examples/examples/diagnostics_and_criticism/sampler-stats.ipynb, Section: "Multiple samplers"
Authors: Meenal Jhajharia, Christian Luhmann
Description: A simple model with one Bernoulli and one Normal variable, used to demonstrate
    how PyMC handles multiple samplers (BinaryMetropolis + Metropolis) and merges their
    sampler statistics.

Has discrete variables: Yes (mu1 - Bernoulli)

Changes from original:
- Preserved coords and dims from the original
- Removed sampling, diagnostics inspection, and plotting code

Benchmark results:
- Original:  logp = -1.1421, 2.5 us/call (100000 evals)
- Frozen:    logp = -1.1421, 2.4 us/call (100000 evals)
"""

import numpy as np
import pymc as pm

def build_model():
    coords = {"step": ["BinaryMetropolis", "Metropolis"], "obs": ["mu1"]}

    with pm.Model(coords=coords) as model:
        mu1 = pm.Bernoulli("mu1", p=0.8)
        mu2 = pm.Normal("mu2", mu=0, sigma=1, dims="obs")

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model, discrete=True)
