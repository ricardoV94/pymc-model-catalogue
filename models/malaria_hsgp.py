"""
Model: Malaria Prevalence HSGP (Gambia)
Source: pymc-examples/examples/spatial/malaria_prevalence.ipynb, Section: "Model Specification"
Authors: Jonathan Dekermanjian
Description: Binomial regression of malaria prevalence across 65 sampled villages in
    the Gambia, with logit-linked linear elevation covariate plus a 2D Hilbert-Space
    Gaussian Process approximation (Matern 3/2 kernel) on village coordinates.

Changes from original:
- Preprocessed raster-joined elevation and aggregated village counts saved to
  data/malaria_gambia.npz instead of loading CSV + TIFF raster with geopandas/rasterio
- Removed sampling, plotting, posterior predictive, and out-of-sample prediction code
- Added ip capture and initval clearing

Benchmark results:
- Original:  logp = -1871.6529, grad norm = 309.0386, 1717.5 us/call (8844 evals)
- Frozen:    logp = -1871.6529, grad norm = 309.0386, 51.3 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data = np.load(Path(__file__).parent / "data" / "malaria_gambia.npz")
    elev = data["elev"]
    pos = data["pos"]
    n = data["n"]
    lonlat = data["lonlat"]

    # Standardize elevation values
    elev_std = (elev - np.mean(elev)) / np.std(elev)

    with pm.Model() as hsgp_model:
        _X = pm.Data("X", lonlat)
        _elev = pm.Data("elevation", elev_std)

        ls = pm.Gamma("ls", mu=20, sigma=5)
        cov_func = pm.gp.cov.Matern32(2, ls=ls)
        m0, m1, c = 40, 40, 2.5
        gp = pm.gp.HSGP(m=[m0, m1], c=c, cov_func=cov_func)
        s = gp.prior("s", X=_X)

        beta_0 = pm.Normal("beta_0", 0, 1)
        beta_1 = pm.Normal("beta_1", 0, 1)

        p_logit = pm.Deterministic("p_logit", beta_0 + beta_1 * _elev + s)
        p = pm.Deterministic("p", pm.math.invlogit(p_logit))
        pm.Binomial("likelihood", n=n, logit_p=p_logit, observed=pos)

    ip = hsgp_model.initial_point()
    hsgp_model.rvs_to_initial_values = {rv: None for rv in hsgp_model.free_RVs}
    return hsgp_model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
