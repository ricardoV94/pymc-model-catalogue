"""
Model: Phylogenetic Gaussian Process Regression of Primate Brain Size
Source: pymc-examples/examples/statistical_rethinking_lectures/16-Gaussian_Processes.ipynb, Section: "Full Phylogenetic Model"
Authors: Dustin Stansbury
Description: GP regression of primate brain size on body mass and social group size,
    using phylogenetic distance as the GP kernel input (Matern 1/2 / Ornstein-Uhlenbeck).
    Controls for shared evolutionary history via phylogenetic covariance.

Changes from original:
- Loaded data inline from npz instead of using utils.load_data
- Implemented standardize inline
- Defined MeanBodyMassSocialGroupSize class inline
- Removed sampling and plotting code

Benchmark results:
- Original:  logp = -245.3057, grad norm = 155.8732, 1582.4 us/call (9559 evals)
- Frozen:    logp = -245.3057, grad norm = 155.8732, 1984.8 us/call (6913 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data = np.load(
        Path(__file__).parent / "data" / "sr_primates301.npz", allow_pickle=True
    )
    name = data["name"]
    brain = data["brain"]
    body = data["body"]
    group_size = data["group_size"]
    distance_matrix = data["distance_matrix"]

    # Filter to complete cases (brain, body, group_size all non-NaN)
    mask = ~(np.isnan(brain) | np.isnan(body) | np.isnan(group_size))
    idx = np.where(mask)[0]

    name_cc = name[idx]
    brain_cc = brain[idx]
    body_cc = body[idx]
    group_size_cc = group_size[idx]

    # Filter distance matrix
    D_full = distance_matrix[np.ix_(idx, idx)]
    D = D_full / D_full.max()

    # Standardize log-transformed values
    def standardize(x):
        c = x - np.nanmean(x)
        return c / np.nanstd(c)

    G = standardize(np.log(group_size_cc))
    M = standardize(np.log(body_cc))
    B = standardize(np.log(brain_cc))

    PRIMATE = name_cc.tolist()
    coords = {"primate": PRIMATE}

    class MeanBodyMassSocialGroupSize(pm.gp.mean.Linear):
        """Custom mean function that separates covariates from phylogeny"""

        def __init__(self, alpha, beta_G, beta_M):
            self.alpha = alpha
            self.beta_G = beta_G
            self.beta_M = beta_M

        def __call__(self, X):
            return self.alpha + self.beta_G * G + self.beta_M * M

    with pm.Model(coords=coords) as model:
        # Priors
        alpha = pm.Normal("alpha", 0, 1)
        sigma = pm.Exponential("sigma", 1)

        beta_M = pm.Normal("beta_M", 0, 1)
        beta_G = pm.Normal("beta_G", 0, 1)

        # Define the mean function
        mean_func = MeanBodyMassSocialGroupSize(alpha, beta_G, beta_M)

        # Phylogenetic distance covariance
        eta_squared = pm.TruncatedNormal("eta_squared", 1, 0.25, lower=0.01)
        rho = pm.TruncatedNormal("rho", 3, 0.25, lower=0.01)

        # Ornstein-Uhlenbeck kernel (Matern 1/2)
        cov_func = eta_squared * pm.gp.cov.Matern12(1, ls=rho)

        # Init the GP
        gp = pm.gp.Marginal(mean_func=mean_func, cov_func=cov_func)

        # Likelihood
        gp.marginal_likelihood("B", X=D, y=B, sigma=sigma)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
