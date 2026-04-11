"""
Model: Beta-Bernoulli-Binomial Compound Step Demo
Source: pymc-examples/examples/samplers/sampling_compound_step.ipynb, Section: "Compound steps by default"
Authors: Thomas Wiecki, Meenal Jhajharia
Description: A simple hierarchical model with a continuous Beta probability, a discrete
    Bernoulli indexing variable, and an observed Binomial likelihood. Used to demonstrate
    PyMC's compound step assignment (NUTS for the continuous variable, BinaryGibbsMetropolis
    for the discrete variable).

Has discrete variables: Yes (ni - Bernoulli)

Changes from original:
- Removed sampling and plotting code

Benchmark results:
- Original:  logp = -5.2577, 1.7 us/call (100000 evals)
- Frozen:    logp = -5.2577, 1.7 us/call (100000 evals)
"""

import numpy as np
import pymc as pm
import pytensor


def build_model():
    n_ = pytensor.shared(np.asarray([10, 15]))

    with pm.Model() as model:
        p = pm.Beta("p", 1.0, 1.0)
        ni = pm.Bernoulli("ni", 0.5)
        k = pm.Binomial("k", p=p, n=n_[ni], observed=4)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model, discrete=True)
