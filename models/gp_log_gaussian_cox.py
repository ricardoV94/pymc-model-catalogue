"""
Model: Log-Gaussian Cox Process for spatial point patterns
Source: pymc-examples/examples/gaussian_processes/log-gaussian-cox-process.ipynb, Section: "Modeling spatial point patterns with a marked log-Gaussian Cox process"
Authors: Christopher Krapu, Chris Fonnesbeck
Description: Latent GP with Matern52 covariance models log-intensity of a spatial
    Poisson process on a 14x9 grid of cells, with anemone count data.

Changes from original:
- Saved anemone data to .npz instead of loading via pm.get_data
- Removed sampling, plotting, conditional predictions
- Inlined grid construction and histogram computation
- Added ip capture and initval clearing

Benchmark results:
- Original:  logp = -405.8429, grad norm = 316.8569, 825.2 us/call (15969 evals)
- Frozen:    logp = -405.8429, grad norm = 316.8569, 829.5 us/call (19131 evals)
"""

from itertools import product
from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    RANDOM_SEED = 42
    rng = np.random.default_rng(RANDOM_SEED)

    data = np.load(
        Path(__file__).parent / "data" / "anemones.npz", allow_pickle=True
    )
    xy = np.column_stack([data["x"], data["y"]]).astype(float)

    # Jitter slightly so no points fall on cell boundaries
    eps = 1e-3
    xy = xy + rng.standard_normal(xy.shape) * eps

    resolution = 20
    area_per_cell = resolution**2 / 100

    cells_x = int(280 / resolution)
    cells_y = int(180 / resolution)

    quadrat_x = np.linspace(0, 280, cells_x + 1)
    quadrat_y = np.linspace(0, 180, cells_y + 1)

    centroids = np.asarray(
        list(product(quadrat_x[:-1] + 10, quadrat_y[:-1] + 10))
    )

    cell_counts, _, _ = np.histogram2d(
        xy[:, 0], xy[:, 1], [quadrat_x, quadrat_y]
    )
    cell_counts = cell_counts.ravel().astype(int)

    coords = {"cell": np.arange(cell_counts.size)}

    with pm.Model(coords=coords) as model:
        mu = pm.Normal("mu", sigma=3)
        rho = pm.Uniform("rho", lower=25, upper=300)
        variance = pm.InverseGamma("variance", alpha=1, beta=1)
        cov_func = variance * pm.gp.cov.Matern52(2, ls=rho)
        mean_func = pm.gp.mean.Constant(mu)

        gp = pm.gp.Latent(mean_func=mean_func, cov_func=cov_func)

        log_intensity = gp.prior("log_intensity", X=centroids, dims="cell")
        intensity = pm.math.exp(log_intensity)

        rates = intensity * area_per_cell
        counts = pm.Poisson(
            "counts", mu=rates, observed=cell_counts, dims="cell"
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
