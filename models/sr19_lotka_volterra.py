"""
Model: Lotka-Volterra Predator-Prey Population Dynamics
Source: pymc-examples/examples/statistical_rethinking_lectures/19-Generalized_Linear_Madness.ipynb, Section: "Lotka-Volterra Model"
Authors: Dustin Stansbury
Description: Bayesian ODE model of hare-lynx population dynamics using Lotka-Volterra
    differential equations. ODE is solved via scipy.integrate.odeint wrapped in a
    pytensor op. Priors are informed by least-squares initialization. Observation model
    accounts for trapping probability and measurement error.

Changes from original:
- Loaded data inline from npz instead of using utils.load_data
- Inlined the ODE function and pytensor wrapper
- Moved least-squares initialization into build_model
- Removed sampling and plotting code

Note: Uses as_op (black-box ODE solver), so gradients are not available.
    Benchmark should use discrete=True mode (logp only, no gradient).

Benchmark results:
- Original:  logp = -246.3392, 1452.9 us/call (11860 evals)
- Frozen:    logp = -246.3392, 1179.8 us/call (9904 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    from pytensor.compile.ops import as_op
    from scipy.integrate import odeint
    from scipy.optimize import least_squares

    def lotka_volterra_diffeq(X, t, theta):
        H, L = X
        bH, mH, bL, mL, _, _ = theta
        dH_dt = H * (bH - mH * L)
        dL_dt = L * (bL * H - mL)
        return [dH_dt, dL_dt]

    data = np.load(Path(__file__).parent / "data" / "sr_lynx_hare.npz")
    TIMES = data["Year"]
    H_obs = data["Hare"]
    L_obs = data["Lynx"]

    def ode_residuals(theta):
        sol = odeint(
            func=lotka_volterra_diffeq, y0=theta[-2:], t=TIMES, args=(theta,)
        )
        residuals = np.column_stack([H_obs, L_obs]) - sol
        return residuals.flatten()

    theta_0 = np.array([0.5, 0.025, 0.025, 0.75, H_obs[0], L_obs[0]])
    lstsq_solution = least_squares(ode_residuals, x0=theta_0)
    lstsq = dict(
        zip(["bH", "mH", "bL", "mL", "H0", "L0"], lstsq_solution.x)
    )

    @as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
    def pytensor_lotka_volterra(theta):
        return odeint(
            func=lotka_volterra_diffeq, y0=theta[-2:], t=TIMES, args=(theta,)
        )

    with pm.Model() as model:
        # Initial population priors
        initial_H = pm.LogNormal(
            "initial_H", np.log(10), 1, initval=lstsq["H0"]
        )
        initial_L = pm.LogNormal(
            "initial_L", np.log(10), 1, initval=lstsq["L0"]
        )

        # Hare param priors
        bH = pm.TruncatedNormal(
            "bH", mu=lstsq["bH"], sigma=0.1, lower=0.0, initval=lstsq["bH"]
        )
        mH = pm.TruncatedNormal(
            "mH", mu=lstsq["mH"], sigma=0.05, lower=0.0, initval=lstsq["mH"]
        )

        # Lynx param priors
        bL = pm.TruncatedNormal(
            "bL", mu=lstsq["bL"], sigma=0.05, lower=0.0, initval=lstsq["bL"]
        )
        mL = pm.TruncatedNormal(
            "mL", mu=lstsq["mL"], sigma=0.1, lower=0.0, initval=lstsq["mL"]
        )

        # Run dynamical system
        ode_solution = pytensor_lotka_volterra(
            pm.math.stack([bH, mH, bL, mL, initial_H, initial_L])
        )

        # Observation model
        p_trapped_H = pm.Beta("p_trapped_H", 40, 200)
        p_trapped_L = pm.Beta("p_trapped_L", 40, 200)

        # Measurement error variance
        sigma_H = pm.Exponential("sigma_H", 1)
        sigma_L = pm.Exponential("sigma_L", 1)

        # Hare likelihood
        population_H = ode_solution[:, 0]
        mu_H = pm.math.log(population_H * p_trapped_H)
        pm.LogNormal("H", mu=mu_H, sigma=sigma_H, observed=H_obs)

        # Lynx likelihood
        population_L = ode_solution[:, 1]
        mu_L = pm.math.log(population_L * p_trapped_L)
        pm.LogNormal("L", mu=mu_L, sigma=sigma_L, observed=L_obs)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    # as_op means no gradients available; use discrete mode
    run_benchmark(build_model, discrete=True)
