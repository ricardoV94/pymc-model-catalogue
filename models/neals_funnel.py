"""
Model: Neal's Funnel
Source: Gorinova, Moore & Hoffman, "Automatic Reparameterisation of Probabilistic
    Programs" (arXiv:1906.03028, NeurIPS 2019), Figure 1; originally Neal (2003).
    Reference implementation: https://github.com/mgorinova/autoreparam
Authors: Maria I. Gorinova, Dave Moore, Matthew D. Hoffman
Description: The canonical reparameterisation stress test. The centred
    parameterisation has a pathological funnel geometry (the scale of `x` depends
    exponentially on `z`), which most gradient-based samplers struggle with.
    Prior-only joint (no observed data): logp is the joint log-density of z and x.

Changes from original:
- None (dataless / synthetic model).

Benchmark results:
- Original:  logp = -2.9365, grad norm = 0.5000, 4.7 us/call (100000 evals)
- Frozen:    logp = -2.9365, grad norm = 0.5000, 4.7 us/call (100000 evals)
"""

import pymc as pm


def build_model():
    with pm.Model() as model:
        z = pm.Normal("z", mu=0.0, sigma=3.0)
        x = pm.Normal("x", mu=0.0, sigma=pm.math.exp(z / 2.0))

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
