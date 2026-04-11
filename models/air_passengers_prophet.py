"""
Model: Prophet-like model for Air Passengers (linear trend + Fourier seasonality)
Source: pymc-examples/examples/time_series/Air_passengers-Prophet_with_Bayesian_workflow.ipynb, Section: "Part 2: enter seasonality"
Authors: Marco Gorelli, Danh Phan
Description: A Prophet-like PyMC model for the monthly Air Passengers dataset with a linear
    trend component combined with multiplicative Fourier seasonality (order 10, so 20 Fourier
    features). The final model uses narrower priors determined via prior predictive checks.

Changes from original:
- Saved precomputed t (scaled time), y (scaled passengers), and Fourier feature matrix
  to .npz instead of regenerating from the CSV inline
- Removed sampling and plotting code

Benchmark results:
- Original:  logp = -1502.3919, grad norm = 8305.1644, 3.6 us/call (100000 evals)
- Frozen:    logp = -1502.3919, grad norm = 8305.1644, 3.8 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data = np.load(Path(__file__).parent / "data" / "air_passengers_prophet.npz")
    t = data["t"]
    y = data["y"]
    fourier_features = data["fourier_features"]

    n_order = 10
    coords = {"fourier_features": np.arange(2 * n_order)}
    with pm.Model(check_bounds=False, coords=coords) as model:
        α = pm.Normal("α", mu=0, sigma=0.5)
        β = pm.Normal("β", mu=0, sigma=0.5)
        trend = pm.Deterministic("trend", α + β * t)

        β_fourier = pm.Normal("β_fourier", mu=0, sigma=0.1, dims="fourier_features")
        seasonality = pm.Deterministic(
            "seasonality", pm.math.dot(β_fourier, fourier_features.T)
        )

        μ = trend * (1 + seasonality)
        σ = pm.HalfNormal("σ", sigma=0.1)
        pm.Normal("likelihood", mu=μ, sigma=σ, observed=y)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
