"""
Model: SMC-ABC Lotka-Volterra via Simulator
Source: pymc-examples/examples/samplers/SMC-ABC_Lotka-Volterra_example.ipynb, Section: "Lotka-Volterra"
Authors: Osvaldo Martin
Description: Likelihood-free inference of the Lotka-Volterra predator-prey ODE parameters
    (a: prey growth rate, b: predation rate) using a pm.Simulator whose synthetic data
    comes from scipy.integrate.odeint, with HalfNormal priors and epsilon=10.

Has discrete variables: No, but uses pm.Simulator which has no defined gradient and
    a *random* logp (the ODE is resimulated with fresh noise on every call), so it is
    benchmarked via compile_logp (no dlogp) alongside the discrete models.

Changes from original:
- Saved the synthetic observed dataset (100 x 2 ODE integration plus Gaussian noise, seed 42)
  to .npz
- Removed sampling and plotting code
- Inlined the competition_model simulator (and dX_dt ODE RHS and the other fixed parameters
  c, d, X0, t) inside build_model()
- Squeezed a, b to scalars inside competition_model: current PyMC's Simulator passes
  params with extra batch dims, which would otherwise make dX_dt return a 3D array
  and trip scipy.integrate.odeint's 1-D-output check.

Benchmark results:
- Original:  logp = -328.2938, 229.3 us/call (53572 evals)
- Frozen:    logp = -328.2938, 237.6 us/call (53169 evals)
  (logp is stochastic via pm.Simulator; runs will vary)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    from scipy.integrate import odeint

    observed = np.load(Path(__file__).parent / "data" / "smc_abc_lotka_volterra.npz")["observed"]

    # Definition of fixed parameters
    c = 1.5
    d = 0.75

    # initial population of rabbits and foxes
    X0 = [10.0, 5.0]
    # size of data
    size = 100
    # time lapse
    time = 15
    t = np.linspace(0, time, size)

    # Lotka - Volterra equation
    def dX_dt(X, t, a, b, c, d):
        """Return the growth rate of fox and rabbit populations."""
        return np.array([a * X[0] - b * X[0] * X[1], -c * X[1] + d * b * X[0] * X[1]])

    # simulator function
    def competition_model(rng, a, b, size=None):
        a = float(np.asarray(a).squeeze())
        b = float(np.asarray(b).squeeze())
        return odeint(dX_dt, y0=X0, t=t, rtol=0.01, args=(a, b, c, d))

    with pm.Model() as model:
        a = pm.HalfNormal("a", 1.0)
        b = pm.HalfNormal("b", 1.0)

        sim = pm.Simulator("sim", competition_model, params=(a, b), epsilon=10, observed=observed)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model, discrete=True)
