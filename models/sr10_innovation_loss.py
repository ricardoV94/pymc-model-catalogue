"""
Model: Oceanic Technology Innovation-Loss Model
Source: pymc-examples/examples/statistical_rethinking_lectures/10-Counts_&_Hidden_Confounds.ipynb, Section: "Innovation/Loss Model"
Authors: Dustin Stansbury
Description: Scientific model of tool complexity in Oceanic societies based on a balance
    between innovation (population-driven) and loss (constant rate). Contact level affects
    both innovation rate and elasticity of population effect. Uses Poisson likelihood
    with a difference-equation equilibrium as mean rate.

Changes from original:
- Loaded data inline from npz instead of using utils.load_data
- Removed sampling, LOO-CV, and plotting code

Benchmark results:
- Original:  logp = -225.9967, grad norm = 519.0117, 2.9 us/call (100000 evals)
- Frozen:    logp = -225.9967, grad norm = 519.0117, 2.7 us/call (100000 evals)
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
    population = data["population"].astype(float)

    # Factorize contact level from Kline2 dataset
    # In the original, contact is loaded from the CSV; here we inline it
    # Kline2 contact: low, low, low, high, low, low, high, low, high, low
    contact_labels = np.array(
        ["low", "low", "low", "high", "low", "low", "high", "low", "high", "low"]
    )
    unique_contacts = []
    contact_level = np.empty(len(contact_labels), dtype=int)
    for i, c in enumerate(contact_labels):
        if c not in unique_contacts:
            unique_contacts.append(c)
        contact_level[i] = unique_contacts.index(c)
    CONTACT = unique_contacts

    TOOLS = total_tools
    POPULATION = population

    ETA = 4
    with pm.Model(coords={"contact": CONTACT}) as model:
        pop = pm.Data("population", POPULATION, dims="obs_id")
        cl = pm.Data("contact_level", contact_level, dims="obs_id")

        # Priors
        alpha = pm.Exponential("alpha", ETA, dims="contact")
        beta = pm.Exponential("beta", ETA, dims="contact")
        gamma = pm.Exponential("gamma", ETA)

        # Likelihood using difference equation equilibrium as mean Poisson rate
        T_hat = (alpha[cl] * (pop ** beta[cl])) / gamma
        pm.Poisson("tools", T_hat, observed=TOOLS, dims="obs_id")

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
