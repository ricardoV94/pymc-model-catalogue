"""
Model: Probabilistic Matrix Factorization (MovieLens 100k)
Source: pymc-examples/examples/case_studies/probabilistic_matrix_factorization.ipynb, Section: "Probabilistic Matrix Factorization"
Authors: Mack Sweeney, Colin Carroll, Rob Zinkov
Description: Bayesian probabilistic matrix factorization for collaborative filtering
    on movie ratings, with Gaussian priors on user and item latent factor matrices U and V.

Changes from original:
- Extracted model from PMF class into standalone build_model function
- Saved dense rating matrix to .npz
- Removed MAP estimation, sampling, baseline comparisons, and plotting

Benchmark results:
- Original:  logp = -1420540.2085, grad norm = 103.9623, 10168.3 us/call (1468 evals)
- Frozen:    logp = -1420540.2085, grad norm = 103.9623, 10090.3 us/call (1299 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data_path = Path(__file__).parent / "data" / "ml_100k_ratings.npz"
    dense_data = np.load(data_path)["dense_data"]

    dim = 10  # Number of latent factors
    alpha = 2  # Fixed precision for likelihood
    std = 0.01  # Initialization noise

    rng = np.random.default_rng(827)

    n, m = dense_data.shape

    # Mean value imputation for initialization stats
    nan_mask = np.isnan(dense_data)
    train = dense_data.copy()
    train[nan_mask] = train[~nan_mask].mean()

    alpha_u = 1 / train.var(axis=1).mean()
    alpha_v = 1 / train.var(axis=0).mean()

    obs_user_idx, obs_movie_idx = np.where(~nan_mask)
    obs_user_idx = obs_user_idx.astype("int64")
    obs_movie_idx = obs_movie_idx.astype("int64")
    obs_ratings = dense_data[~nan_mask]

    with pm.Model(
        coords={
            "user": np.arange(n),
            "movie": np.arange(m),
            "latent_factor": np.arange(dim),
            "obs_id": np.arange(obs_ratings.shape[0]),
        }
    ) as model:
        user_idx_ = pm.Data("user_idx", obs_user_idx, dims="obs_id")
        movie_idx_ = pm.Data("movie_idx", obs_movie_idx, dims="obs_id")
        ratings_ = pm.Data("ratings", obs_ratings, dims="obs_id")

        U = pm.Normal(
            "U",
            mu=0,
            sigma=1 / np.sqrt(alpha_u),
            dims=("user", "latent_factor"),
            initval=rng.standard_normal(size=(n, dim)) * std,
        )
        V = pm.Normal(
            "V",
            mu=0,
            sigma=1 / np.sqrt(alpha_v),
            dims=("movie", "latent_factor"),
            initval=rng.standard_normal(size=(m, dim)) * std,
        )
        mu = (U[user_idx_] * V[movie_idx_]).sum(axis=-1)
        pm.Normal(
            "R",
            mu=mu,
            sigma=np.sqrt(1.0 / alpha),
            dims="obs_id",
            observed=ratings_,
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
