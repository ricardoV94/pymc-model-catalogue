"""
Model: Enzymatic reaction ODE with partially-observed states
Source: pymc-examples/examples/ode_models/ODE_API_shapes_and_benchmarking.ipynb, Section: "Demo Scenario: Simple enzymatic reaction"
Authors: Michael Osthege (Raul-ing Average), Thomas Wiecki, Oriol Abril-Pla, Chris Fonnesbeck
Description: Two-state chemistry ODE (substrate S and product P) with shared Vmax/K_S
    parameters, where S and P are observed at different (overlapping) time indices.
    Demonstrates slicing the flattened DifferentialEquation output for partially-
    observed multi-state systems.

Changes from original:
- pymc3 -> pymc (pm.ode.DifferentialEquation)
- Fixed a typo in the original notebook (`y_hpt` -> `y_hat`) that would have made
  the original `get_model()` function raise NameError.
- Removed the module-level `Chem` helper class; the reaction right-hand-side is
  inlined inside build_model().
- Inlined synthetic data generation
- Removed theano test-value config and the theano-based benchmark/graph-printing
  harness; this file uses the catalogue's standard run_benchmark helper.

Benchmark results:
- Original:  logp = -60.3474, grad norm = 47.6790, 2325.9 us/call (6344 evals)
- Frozen:    logp = -60.3474, grad norm = 47.6790, 2327.4 us/call (6280 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    from scipy.integrate import odeint

    def reaction(y, t, p):
        S, P = y[0], y[1]
        vmax, K_S = p[0], p[1]
        dPdt = vmax * (S / K_S + S)
        dSdt = -dPdt
        return [dSdt, dPdt]

    rng = np.random.default_rng(23489)

    times = np.arange(0, 10, 0.5)
    red = np.arange(5, len(times))
    blue = np.arange(12)

    y0_true = (10, 2)
    theta_true = (0.5, 2)
    sigma_true = 0.2

    y_true = odeint(reaction, t=times, y0=y0_true, args=(theta_true,))
    y_obs_1 = rng.normal(y_true[red, 0], sigma_true)
    y_obs_2 = rng.normal(y_true[blue, 1], sigma_true)

    with pm.Model() as model:
        sigma = pm.HalfCauchy("sigma", 1)
        vmax = pm.Lognormal("vmax", 0, 1)
        K_S = pm.Lognormal("K_S", 0, 1)
        s0 = pm.Normal("red_0", mu=10, sigma=2)

        y_hat = pm.ode.DifferentialEquation(
            func=reaction,
            times=times,
            n_states=len(y0_true),
            n_theta=len(theta_true),
        )(y0=[s0, y0_true[1]], theta=[vmax, K_S], return_sens=False)

        red_hat = y_hat.T[0][red]
        blue_hat = y_hat.T[1][blue]

        Y_red = pm.Normal("Y_red", mu=red_hat, sigma=sigma, observed=y_obs_1)
        Y_blue = pm.Normal("Y_blue", mu=blue_hat, sigma=sigma, observed=y_obs_2)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
