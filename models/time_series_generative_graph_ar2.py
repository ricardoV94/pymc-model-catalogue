"""
Model: AR(2) via pytensor.scan generative graph (observed)
Source: pymc-examples/examples/time_series/Time_Series_Generative_Graph.ipynb, Section: "Posterior"
Authors: Jesse Grabowski, Juan Orduz, Ricardo Vieira
Description: AR(2) time series model defined from a generative graph using pytensor.scan
    inside a pm.CustomDist, rather than pm.AR. The model has Normal priors on the
    autoregressive coefficients, a HalfNormal prior on the noise scale, and a Normal
    prior on the two-lag initial state. Observed data is attached via pm.observe.

Changes from original:
- Observed data (`ar_obs`) is generated deterministically with a numpy AR(2) simulation
  using a fixed seed, rather than drawing from pm.sample_prior_predictive (which would
  require running the full prior sampler at build time).
- Removed sampling, prior/posterior predictive, and plotting code.

Benchmark results:
- Original:  logp = -211.7216, grad norm = 561.8939, 34.0 us/call (100000 evals)
- Frozen:    logp = -211.7216, grad norm = 561.8939, 33.3 us/call (100000 evals)
"""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from pymc.pytensorf import collect_default_updates


def build_model():
    lags = 2
    timeseries_length = 100

    # Deterministic synthetic observed AR(2) series (stand-in for the prior draw
    # used in the original notebook).
    rng = np.random.default_rng(42)
    rho_true = np.array([0.6, -0.2])
    sigma_true = 0.5
    ar_obs = np.zeros(timeseries_length)
    ar_obs[:lags] = rng.normal(scale=0.5, size=lags)
    for t in range(lags, timeseries_length):
        ar_obs[t] = (
            rho_true[0] * ar_obs[t - 1]
            + rho_true[1] * ar_obs[t - 2]
            + rng.normal(scale=sigma_true)
        )
    test_data = ar_obs[lags:]

    def ar_step(x_tm2, x_tm1, rho, sigma):
        mu = x_tm1 * rho[0] + x_tm2 * rho[1]
        x = mu + pm.Normal.dist(sigma=sigma)
        return x, collect_default_updates([x])

    def ar_dist(ar_init, rho, sigma, size):
        ar_steps, _ = pytensor.scan(
            fn=ar_step,
            outputs_info=[{"initial": ar_init, "taps": range(-lags, 0)}],
            non_sequences=[rho, sigma],
            n_steps=timeseries_length - lags,
            strict=True,
        )
        return ar_steps

    coords = {
        "lags": range(-lags, 0),
        "steps": range(timeseries_length - lags),
        "timeseries_length": range(timeseries_length),
    }
    with pm.Model(coords=coords, check_bounds=False) as model:
        rho = pm.Normal(name="rho", mu=0, sigma=0.2, dims=("lags",))
        sigma = pm.HalfNormal(name="sigma", sigma=0.2)

        ar_init = pm.Normal(name="ar_init", sigma=0.5, dims=("lags",))

        ar_steps = pm.CustomDist(
            "ar_steps",
            ar_init,
            rho,
            sigma,
            dist=ar_dist,
            dims=("steps",),
            observed=test_data,
        )

        ar = pm.Deterministic(
            name="ar",
            var=pt.concatenate([ar_init, ar_steps], axis=-1),
            dims=("timeseries_length",),
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
