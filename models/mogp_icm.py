"""
Model: Multi-output GP with ICM (Intrinsic Coregionalization Model)
Source: pymc-examples/examples/gaussian_processes/MOGP-Coregion-Hadamard.ipynb, Section: "Intrinsic Coregionalization Model (ICM)"
Authors: Danh Phan, Bill Engels, Chris Fonnesbeck
Description: Multi-output GP using the Intrinsic Coregionalization Model (ICM)
    with Hadamard product. Models average spin rates of 5 MLB pitchers using
    ExpQuad x Coregion kernel structure with Marginal GP.

Changes from original:
- Saved preprocessed fastball data to .npz instead of loading CSV
- Removed sampling, plotting, conditional predictions
- Added ip capture and initval clearing

Benchmark results:
- Original:  logp = -13529871.5370, grad norm = 24397029.5981, 6096.2 us/call (2362 evals)
- Frozen:    logp = -13529871.5370, grad norm = 24397029.5981, 6377.6 us/call (2134 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    def get_icm(input_dim, kernel, W=None, kappa=None, B=None, active_dims=None):
        coreg = pm.gp.cov.Coregion(
            input_dim=input_dim, W=W, kappa=kappa, B=B, active_dims=active_dims
        )
        return kernel * coreg

    data = np.load(
        Path(__file__).parent / "data" / "fastball_spin_rates.npz",
        allow_pickle=True,
    )
    X = data["X"]
    Y = data["Y"]
    n_outputs = data["n_outputs"].item()

    rng = np.random.default_rng(8927)

    with pm.Model() as model:
        # Priors
        ell = pm.Gamma("ell", alpha=2, beta=0.5)
        eta = pm.Gamma("eta", alpha=3, beta=1)
        kernel = eta**2 * pm.gp.cov.ExpQuad(input_dim=2, ls=ell, active_dims=[0])
        sigma = pm.HalfNormal("sigma", sigma=3)

        # Get the ICM kernel
        W = pm.Normal(
            "W",
            mu=0,
            sigma=3,
            shape=(n_outputs, 2),
            initval=rng.standard_normal((n_outputs, 2)),
        )
        kappa = pm.Gamma("kappa", alpha=1.5, beta=1, shape=n_outputs)
        B = pm.Deterministic("B", pt.dot(W, W.T) + pt.diag(kappa))
        cov_icm = get_icm(
            input_dim=2, kernel=kernel, B=B, active_dims=[1]
        )

        # Define a Multi-output GP
        mogp = pm.gp.Marginal(cov_func=cov_icm)
        y_ = mogp.marginal_likelihood("f", X, Y, sigma=sigma)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
