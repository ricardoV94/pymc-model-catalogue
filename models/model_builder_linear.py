"""
Model: ModelBuilder Linear Regression
Source: pymc-examples/examples/howto/model_builder.ipynb, Section: "Standard syntax"
Authors: Shashank Kirtania, Thomas Wiecki, Michal Raczycki
Description: Simple linear regression (a + b*x with HalfNormal noise) used as a
    placeholder model to demonstrate the pymc_extras ModelBuilder sklearn-style
    wrapper. The underlying pm.Model is extracted here; the ModelBuilder subclass
    machinery is ignored as it is just API sugar.

Changes from original:
- Dropped the ModelBuilder subclass and inlined the underlying pm.Model
- Inlined small generated data
- Removed sampling / predict / plotting code

Benchmark results:
- Original:  logp = -177.4419, grad norm = 114.0914, 2.4 us/call (100000 evals)
- Frozen:    logp = -177.4419, grad norm = 114.0914, 2.2 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    rng = np.random.default_rng(8927)
    x = np.linspace(start=0, stop=1, num=100)
    y = 0.3 * x + 0.5 + rng.normal(0, 1, len(x))

    with pm.Model() as model:
        x_data = pm.Data("x_data", x)
        y_data = pm.Data("y_data", y)

        # priors
        a = pm.Normal("a", mu=0.0, sigma=1.0)
        b = pm.Normal("b", mu=0.0, sigma=1.0)
        eps = pm.HalfNormal("eps", 1.0)

        pm.Normal("y", mu=a + b * x_data, sigma=eps, shape=x_data.shape, observed=y_data)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
