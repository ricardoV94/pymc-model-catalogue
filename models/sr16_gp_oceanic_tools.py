"""
Model: Gaussian Process Model of Oceanic Tool Complexity with Population
Source: pymc-examples/examples/statistical_rethinking_lectures/16-Gaussian_Processes.ipynb, Section: "Distance + Population Model"
Authors: Dustin Stansbury
Description: GP latent model of tool counts across Oceanic societies. Uses inter-island
    geographic distance as GP input with ExpQuad kernel, combined with a population-driven
    innovation rate. Models spatial autocorrelation in technological complexity.

Changes from original:
- Loaded data from npz instead of using utils.load_data
- Removed sampling and plotting code

Benchmark results:
- Original:  logp = -275.3721, grad norm = 693.4104, 8.1 us/call (100000 evals)
- Frozen:    logp = -275.3721, grad norm = 693.4104, 7.6 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data = np.load(
        Path(__file__).parent / "data" / "sr_kline_islands.npz", allow_pickle=True
    )
    culture = data["culture"]
    total_tools = data["total_tools"].astype(int)
    logpop = data["logpop"].astype(float)
    island_distances = data["island_distances"].astype(float)

    # Factorize culture
    CULTURE = culture.tolist()
    CULTURE_ID = np.arange(len(CULTURE))
    TOOLS = total_tools

    coords = {"culture": CULTURE}

    with pm.Model(coords=coords) as model:
        population = pm.Data(
            "log_population", logpop, dims="culture"
        )
        culture_id = pm.Data("CULTURE_ID", CULTURE_ID)

        # Priors
        alpha_bar = pm.Exponential("alpha_bar", 1)
        gamma = pm.Exponential("gamma", 1)
        beta = pm.Exponential("beta", 1)
        eta_squared = pm.Exponential("eta_squared", 2)
        rho_squared = pm.Exponential("rho_squared", 2)

        # Gaussian Process
        kernel_function = eta_squared * pm.gp.cov.ExpQuad(
            input_dim=1, ls=rho_squared
        )
        GP = pm.gp.Latent(cov_func=kernel_function)
        alpha = GP.prior("alpha", X=island_distances, dims="culture")

        # Likelihood
        lambda_T = (
            alpha_bar / gamma * population[culture_id] ** beta
        ) * pm.math.exp(alpha[culture_id])
        pm.Poisson("T", lambda_T, observed=TOOLS, dims="culture")

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
