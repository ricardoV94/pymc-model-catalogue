"""
Model: Multivariate Normal Imputation via Potential (Employee Satisfaction)
Source: pymc-examples/examples/howto/Missing_Data_Imputation.ipynb, Section: "Bayesian Imputation"
Authors: Nathaniel Forde
Description: Joint MvNormal imputation of worksat/empower/lmx using an LKJ prior on the
    covariance and a Uniform latent vector for the NaN entries, combined with the observed
    data via pt.set_subtensor and scored with a Potential term.

Changes from original:
- Inlined data loaded from data/employee_missing.npz (float64) instead of pm.get_data csv.
- Removed sampling / plotting / prior predictive code.

Benchmark results:
- Original:  logp = -451600.4888, grad norm = 756097.9802, 88.6 us/call (100000 evals)
- Frozen:    logp = -451600.4888, grad norm = 756097.9802, 83.6 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    d = np.load(Path(__file__).parent / "data" / "employee_missing.npz")
    data_values = np.column_stack([d["worksat"], d["empower"], d["lmx"]]).astype(
        np.float64
    )

    with pm.Model() as model:
        # Priors
        mus = pm.Normal("mus", 0, 1, size=3)
        cov_flat_prior, _, _ = pm.LKJCholeskyCov(
            "cov", n=3, eta=1.0, sd_dist=pm.Exponential.dist(1)
        )
        # Create a vector of flat variables for the unobserved components of the MvNormal
        x_unobs = pm.Uniform(
            "x_unobs", 0, 100, shape=(np.isnan(data_values).sum(),)
        )

        # Create the symbolic value of x, combining observed data and unobserved variables
        x = pt.as_tensor(data_values)
        x = pm.Deterministic(
            "x", pt.set_subtensor(x[np.isnan(data_values)], x_unobs)
        )

        # Add a Potential with the logp of the variable conditioned on `x`
        pm.Potential(
            "x_logp",
            pm.logp(rv=pm.MvNormal.dist(mus, chol=cov_flat_prior), value=x),
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
