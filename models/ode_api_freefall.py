"""
Model: Freefall ODE with inferred initial condition
Source: pymc-examples/examples/ode_models/ODE_API_introduction.ipynb, Section: "A Differential Equation For Freefall"
Authors: Demetri Pananos, Oriol Abril-Pla, Hector Munoz, Chris Fonnesbeck
Description: Single-state ODE y' = m*g - gamma*y modeling an object in freefall with
    air resistance, using pm.ode.DifferentialEquation. Priors on the drag coefficient
    gamma, the acceleration g, and the initial velocity y0.

Changes from original:
- pymc3 -> pymc (pm.ode.DifferentialEquation)
- Inlined synthetic data generation (21 observations)
- Removed sampling and plotting code
- Extracted the most complete freefall model (model3), which additionally puts a
  prior on the initial condition and on g, superseding the two earlier intermediate
  freefall models (model, model2) in the same notebook.

Benchmark results:
- Original:  logp = -23314.2029, grad norm = 118762.1339, 1611.0 us/call (9132 evals)
- Frozen:    logp = -23314.2029, grad norm = 118762.1339, 1616.1 us/call (8949 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    from scipy.integrate import odeint

    def freefall(y, t, p):
        return 2.0 * p[1] - p[0] * y[0]

    # For reproducibility of synthetic data, matching the notebook
    rng = np.random.default_rng(20394)
    times = np.arange(0, 10, 0.5)
    gamma_true, g_true, y0_true, sigma_true = 0.4, 9.8, -2, 2
    y = odeint(freefall, t=times, y0=y0_true, args=tuple([[gamma_true, g_true]]))
    yobs = rng.normal(y, 2)

    ode_model = pm.ode.DifferentialEquation(
        func=freefall, times=times, n_states=1, n_theta=2, t0=0
    )

    with pm.Model() as model:
        sigma = pm.HalfCauchy("sigma", 1)
        gamma = pm.Lognormal("gamma", 0, 1)
        g = pm.Lognormal("g", pm.math.log(10), 2)
        # Initial condition prior. We think it is at rest, but will allow for
        # perturbations in initial velocity.
        y0 = pm.Normal("y0", 0, 2)

        ode_solution = ode_model(y0=[y0], theta=[gamma, g])

        Y = pm.Normal("Y", mu=ode_solution, sigma=sigma, observed=yobs)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
