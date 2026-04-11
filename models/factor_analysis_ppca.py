"""
Model: Identified Probabilistic PCA (Factor Analysis)
Source: pymc-examples/examples/case_studies/factor_analysis.ipynb, Section: "Alternative parametrization"
Authors: Chris Hartl, Christopher Krapu, Oriol Abril-Pla, Erik Werner
Description: Factor analysis with an identified lower-triangular loading matrix W and
    latent factors F, using a positive-increasing diagonal constraint for identifiability.

Changes from original:
- Inlined simulated data generation with fixed random seed
- Removed sampling, plotting, and post-hoc F recovery code

Benchmark results:
- Original:  logp = -143470.3991, grad norm = 278891.9526, 9.8 us/call (100000 evals)
- Frozen:    logp = -143470.3991, grad norm = 278891.9526, 9.9 us/call (100000 evals)
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    def expand_packed_block_triangular(d, k, packed, diag=None):
        assert d >= k

        out = pt.zeros((d, k), dtype="float64")
        if diag is None:
            idxs = np.tril_indices(d, m=k)
            out = pt.set_subtensor(out[idxs], packed)
        else:
            idxs = np.tril_indices(d, k=-1, m=k)
            out = pt.set_subtensor(out[idxs], packed)
            idxs = (np.arange(k), np.arange(k))
            out = pt.set_subtensor(out[idxs], diag)
        return out

    rng = np.random.default_rng(31415)
    n = 250
    k_true = 4
    d = 10
    err_sd = 2
    M = rng.binomial(1, 0.25, size=(k_true, n))
    Q = np.hstack(
        [rng.exponential(2 * k_true - k, size=(d, 1)) for k in range(k_true)]
    ) * rng.binomial(1, 0.75, size=(d, k_true))
    Y = np.round(1000 * Q @ M + rng.standard_normal(size=(d, n)) * err_sd) / 1000

    k = 2

    coords = {
        "latent_columns": np.arange(k),
        "rows": np.arange(n),
        "observed_columns": np.arange(d),
    }

    # Number of off-diagonal elements in lower-triangular d x k block
    n_od = int(k * d - k * (k - 1) / 2 - k)

    with pm.Model(coords=coords) as model:
        # Loading matrix W with identifiability constraints
        z = pm.HalfNormal("W_z", 1.0, dims="latent_columns")
        b = pm.Normal("W_b", 0.0, 1.0, shape=(n_od,), dims="packed_dim")
        L = expand_packed_block_triangular(d, k, b, pt.ones(k))
        W = pm.Deterministic(
            "W", L @ pt.diag(pt.extra_ops.cumsum(z)), dims=("observed_columns", "latent_columns")
        )

        F = pm.Normal("F", dims=("latent_columns", "rows"))
        sigma = pm.HalfNormal("sigma", 1.0)
        X = pm.Normal("X", mu=W @ F, sigma=sigma, observed=Y, dims=("observed_columns", "rows"))

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
