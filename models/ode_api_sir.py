"""
Model: Non-dimensionalized SIR epidemiological ODE
Source: pymc-examples/examples/ode_models/ODE_API_introduction.ipynb, Section: "Non-linear Differential Equations"
Authors: Demetri Pananos, Oriol Abril-Pla, Hector Munoz, Chris Fonnesbeck
Description: Two-state SIR ODE (dS/dt = -beta*S*I, dI/dt = beta*S*I - lambda*I)
    with a Lognormal observational model, parameterised via R0 (truncated below 1)
    and lambda so that beta = R0 * lambda is a deterministic.

Changes from original:
- pymc3 -> pymc (pm.ode.DifferentialEquation)
- Replaced deprecated `pm.Bound(pm.Normal, lower=1)` with
  `pm.Truncated("R0", pm.Normal.dist(2, 3), lower=1)`, the current PyMC API for
  a lower-bounded Normal prior.
- Added `initval=2.0` to the `lambda` Lognormal so the default initial point
  produces finite logp. Current PyMC defaults Lognormal to its *mean*
  exp(mu + sigma**2/2) = exp(0.693 + 2) ≈ 14.78, which drives the SIR
  integrator to collapse S→0 and yields nan in log(sir_curves).
- Inlined synthetic data generation (19 observations of 2 states)
- Removed sampling and plotting code

Benchmark results:
- Original:  logp = 7.6338, grad norm = 39.3160, 2273.1 us/call (6023 evals)
- Frozen:    logp = 7.6338, grad norm = 39.3160, 2286.4 us/call (5887 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    from scipy.integrate import odeint

    def SIR(y, t, p):
        ds = -p[0] * y[0] * y[1]
        di = p[0] * y[0] * y[1] - p[1] * y[1]
        return [ds, di]

    rng = np.random.default_rng(20394)
    times = np.arange(0, 5, 0.25)
    beta_true, gamma_true = 4, 1.0
    y = odeint(SIR, t=times, y0=[0.99, 0.01], args=((beta_true, gamma_true),), rtol=1e-8)
    yobs = rng.lognormal(mean=np.log(y[1::]), sigma=[0.2, 0.3])

    sir_model = pm.ode.DifferentialEquation(
        func=SIR,
        times=np.arange(0.25, 5, 0.25),
        n_states=2,
        n_theta=2,
        t0=0,
    )

    with pm.Model() as model:
        sigma = pm.HalfCauchy("sigma", 1, shape=2)

        # R0 is bounded below by 1 because we see an epidemic has occurred
        R0 = pm.Truncated("R0", pm.Normal.dist(2, 3), lower=1)
        lam = pm.Lognormal("lambda", pm.math.log(2), 2, initval=2.0)
        beta = pm.Deterministic("beta", lam * R0)

        sir_curves = sir_model(y0=[0.99, 0.01], theta=[beta, lam])

        Y = pm.Lognormal("Y", mu=pm.math.log(sir_curves), sigma=sigma, observed=yobs)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
