"""
Model: Gaussian Process for CO2 at Mauna Loa
Source: pymc-examples/examples/gaussian_processes/GP-MaunaLoa.ipynb, Section: "The model in PyMC"
Authors: Bill Engels, Chris Fonnesbeck
Description: Additive marginal GP model for atmospheric CO2 at Mauna Loa with
    seasonal (Periodic x Matern52), medium-term (RatQuad), and long-term trend
    (ExpQuad) components, plus correlated noise (Matern32 + WhiteNoise).

Changes from original:
- Saved preprocessed CO2 data to .npz instead of loading CSV
- Removed MAP estimation, prediction, and plotting
- Added ip capture and initval clearing

Benchmark results:
- Original:  logp = 974.7189, grad norm = 465.5312, 133617.0 us/call (105 evals)
- Frozen:    logp = 974.7189, grad norm = 465.5312, 133566.5 us/call (116 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data = np.load(Path(__file__).parent / "data" / "mauna_loa_co2.npz")
    t = data["t"][:, None]
    y = data["y_n"]

    with pm.Model() as model:
        # yearly periodic component x long term trend
        eta_per = pm.HalfCauchy("eta_per", beta=2, initval=1.0)
        ell_pdecay = pm.Gamma("ell_pdecay", alpha=10, beta=0.075)
        period = pm.Normal("period", mu=1, sigma=0.05)
        ell_psmooth = pm.Gamma("ell_psmooth", alpha=4, beta=3)
        cov_seasonal = (
            eta_per**2
            * pm.gp.cov.Periodic(1, period, ell_psmooth)
            * pm.gp.cov.Matern52(1, ell_pdecay)
        )
        gp_seasonal = pm.gp.Marginal(cov_func=cov_seasonal)

        # small/medium term irregularities
        eta_med = pm.HalfCauchy("eta_med", beta=0.5, initval=0.1)
        ell_med = pm.Gamma("ell_med", alpha=2, beta=0.75)
        alpha = pm.Gamma("alpha", alpha=5, beta=2)
        cov_medium = eta_med**2 * pm.gp.cov.RatQuad(1, ell_med, alpha)
        gp_medium = pm.gp.Marginal(cov_func=cov_medium)

        # long term trend
        eta_trend = pm.HalfCauchy("eta_trend", beta=2, initval=2.0)
        ell_trend = pm.Gamma("ell_trend", alpha=4, beta=0.1)
        cov_trend = eta_trend**2 * pm.gp.cov.ExpQuad(1, ell_trend)
        gp_trend = pm.gp.Marginal(cov_func=cov_trend)

        # noise model
        eta_noise = pm.HalfNormal("eta_noise", sigma=0.5, initval=0.05)
        ell_noise = pm.Gamma("ell_noise", alpha=2, beta=4)
        sigma = pm.HalfNormal("sigma", sigma=0.25, initval=0.05)
        cov_noise = (
            eta_noise**2 * pm.gp.cov.Matern32(1, ell_noise)
            + pm.gp.cov.WhiteNoise(sigma)
        )

        # The Gaussian process is a sum of these three components
        gp = gp_seasonal + gp_medium + gp_trend

        # Marginal likelihood
        y_ = gp.marginal_likelihood("y", X=t, y=y, sigma=cov_noise)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
