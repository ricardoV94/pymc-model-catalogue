"""
Model: Lotka-Volterra ODE via scipy.odeint wrapped with @as_op
Source: pymc-examples/examples/ode_models/ODE_Lotka_Volterra_multiple_ways.ipynb, Section: "PyMC Model Specification for Gradient-Free Bayesian Inference"
Authors: Greg Brunkhorst
Description: Lotka-Volterra predator-prey ODE fit to Hudson's Bay Company lynx/hare
    data. The forward model calls scipy.integrate.odeint on a numba-njit'd RHS and
    is wrapped as a PyTensor Op via @as_op (no gradients; intended for gradient-free
    samplers but still benchmarkable for logp).

Has discrete variables: No, but the @as_op wrapper produces a PyTensor Op with
    no defined gradient (pullback not implemented), so it is benchmarked via
    compile_logp (no dlogp) alongside the discrete/simulator models.

Changes from original:
- Inlined data
- Inlined the @as_op wrapper and njit RHS inside build_model()
- Removed sampling/plotting code
- Added ip capture + initval clearing boilerplate

Benchmark results:
- Original:  logp = -137.4171, 262.2 us/call (44962 evals)
- Frozen:    logp = -137.4171, 262.8 us/call (45123 evals)
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from numba import njit
from pytensor.compile.ops import as_op
from scipy.integrate import odeint


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

    theta_init = np.array([0.52, 0.026, 0.84, 0.026, 34.0, 5.9])

    @njit
    def rhs(X, t, theta):
        x, y = X
        alpha, beta, gamma, delta, xt0, yt0 = theta
        dx_dt = alpha * x - beta * x * y
        dy_dt = -gamma * y + delta * x * y
        return [dx_dt, dy_dt]

    @as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
    def pytensor_forward_model_matrix(theta):
        return odeint(func=rhs, y0=theta[-2:], t=year, args=(theta,))

    with pm.Model() as model:
        alpha = pm.TruncatedNormal("alpha", mu=theta_init[0], sigma=0.1, lower=0, initval=theta_init[0])
        beta = pm.TruncatedNormal("beta", mu=theta_init[1], sigma=0.01, lower=0, initval=theta_init[1])
        gamma = pm.TruncatedNormal("gamma", mu=theta_init[2], sigma=0.1, lower=0, initval=theta_init[2])
        delta = pm.TruncatedNormal("delta", mu=theta_init[3], sigma=0.01, lower=0, initval=theta_init[3])
        xt0 = pm.TruncatedNormal("xto", mu=theta_init[4], sigma=1, lower=0, initval=theta_init[4])
        yt0 = pm.TruncatedNormal("yto", mu=theta_init[5], sigma=1, lower=0, initval=theta_init[5])
        sigma = pm.HalfNormal("sigma", 10)

        ode_solution = pytensor_forward_model_matrix(
            pm.math.stack([alpha, beta, gamma, delta, xt0, yt0])
        )

        pm.Normal("Y_obs", mu=ode_solution, sigma=sigma, observed=observed)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model, discrete=True)
