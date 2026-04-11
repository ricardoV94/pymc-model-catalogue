"""
Model: Lotka-Volterra ODE via pytensor.scan forward Euler
Source: pymc-examples/examples/ode_models/ODE_Lotka_Volterra_multiple_ways.ipynb, Section: "Simulate with Pytensor Scan"
Authors: Greg Brunkhorst
Description: Lotka-Volterra predator-prey ODE fit to Hudson's Bay Company lynx/hare
    data. The ODE is integrated by forward Euler inside a pytensor.scan (100 steps
    per year) giving fully symbolic gradients for NUTS.

Changes from original:
- Inlined data
- Inlined the scan-based inference model builder inside build_model()
- Removed sampling/plotting code
- Added ip capture + initval clearing boilerplate

Benchmark results:
- Original:  logp = -136.8846, grad norm = 85.1594, 746.1 us/call (20354 evals)
- Frozen:    logp = -136.8846, grad norm = 85.1594, 738.4 us/call (20635 evals)
"""

import numpy as np
import pymc as pm
import pytensor


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

    steps_year = 100
    years = 21
    n_steps = years * steps_year
    dt = 1 / steps_year

    segment = [True] + [False] * (steps_year - 1)
    boolist_idxs = []
    for _ in range(years):
        boolist_idxs += segment

    with pm.Model() as model:
        alpha = pm.TruncatedNormal("alpha", mu=theta_init[0], sigma=0.1, lower=0, initval=theta_init[0])
        beta = pm.TruncatedNormal("beta", mu=theta_init[1], sigma=0.01, lower=0, initval=theta_init[1])
        gamma = pm.TruncatedNormal("gamma", mu=theta_init[2], sigma=0.1, lower=0, initval=theta_init[2])
        delta = pm.TruncatedNormal("delta", mu=theta_init[3], sigma=0.01, lower=0, initval=theta_init[3])
        xt0 = pm.TruncatedNormal("xto", mu=theta_init[4], sigma=1, lower=0, initval=theta_init[4])
        yt0 = pm.TruncatedNormal("yto", mu=theta_init[5], sigma=1, lower=0, initval=theta_init[5])
        sigma = pm.HalfNormal("sigma", 10)

        def ode_update_function(x, y, alpha, beta, gamma, delta):
            x_new = x + (alpha * x - beta * x * y) * dt
            y_new = y + (-gamma * y + delta * x * y) * dt
            return x_new, y_new

        result, updates = pytensor.scan(
            fn=ode_update_function,
            outputs_info=[xt0, yt0],
            non_sequences=[alpha, beta, gamma, delta],
            n_steps=n_steps,
        )

        final_result = pm.math.stack([result[0], result[1]], axis=1)
        annual_value = final_result[np.array(boolist_idxs), :]

        pm.Normal("Y_obs", mu=annual_value, sigma=sigma, observed=observed)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
