"""
Model: Lotka-Volterra ODE with pm.ode.DifferentialEquation
Source: pymc-examples/examples/ode_models/ODE_Lotka_Volterra_multiple_ways.ipynb, Section: "PyMC ODE Module"
Authors: Greg Brunkhorst
Description: Lotka-Volterra predator-prey ODE fit to Hudson's Bay Company lynx/hare
    data using PyMC's built-in DifferentialEquation wrapper (scipy.odeint under the
    hood with finite-difference gradients).

Changes from original:
- Inlined data
- Removed sampling/plotting code
- Added ip capture + initval clearing boilerplate

Benchmark results:
- Original:  logp = -137.4171, grad norm = 88.2992, 7935.0 us/call (1902 evals)
- Frozen:    logp = -137.4171, grad norm = 88.2992, 7953.1 us/call (1858 evals)
"""

import numpy as np
import pymc as pm

from pymc.ode import DifferentialEquation


def build_model():
    year = np.arange(1900.0, 1921.0, 1.0)
    lynx = np.array(
        [
            4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3, 9.1, 7.4,
            8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6,
        ]
    )
    hare = np.array(
        [
            30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4,
            27.1, 40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7,
        ]
    )
    observed = np.stack([hare, lynx], axis=1)

    # Least-squares informed priors (from the notebook)
    theta = np.array([0.52, 0.026, 0.84, 0.026, 34.0, 5.9])

    def rhs_pymcode(y, t, p):
        dX_dt = p[0] * y[0] - p[1] * y[0] * y[1]
        dY_dt = -p[2] * y[1] + p[3] * y[0] * y[1]
        return [dX_dt, dY_dt]

    ode_model = DifferentialEquation(
        func=rhs_pymcode, times=year, n_states=2, n_theta=4, t0=year[0]
    )

    with pm.Model() as model:
        alpha = pm.TruncatedNormal("alpha", mu=theta[0], sigma=0.1, lower=0, initval=theta[0])
        beta = pm.TruncatedNormal("beta", mu=theta[1], sigma=0.01, lower=0, initval=theta[1])
        gamma = pm.TruncatedNormal("gamma", mu=theta[2], sigma=0.1, lower=0, initval=theta[2])
        delta = pm.TruncatedNormal("delta", mu=theta[3], sigma=0.01, lower=0, initval=theta[3])
        xt0 = pm.TruncatedNormal("xto", mu=theta[4], sigma=1, lower=0, initval=theta[4])
        yt0 = pm.TruncatedNormal("yto", mu=theta[5], sigma=1, lower=0, initval=theta[5])
        sigma = pm.HalfNormal("sigma", 10)

        ode_solution = ode_model(y0=[xt0, yt0], theta=[alpha, beta, gamma, delta])

        pm.Normal("Y_obs", mu=ode_solution, sigma=sigma, observed=observed)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
