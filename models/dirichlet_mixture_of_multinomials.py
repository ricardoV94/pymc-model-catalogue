"""
Model: Dirichlet-Multinomial (Marginalized)
Source: pymc-examples/examples/mixture_models/dirichlet_mixture_of_multinomials.ipynb,
    Section: "Dirichlet-Multinomial Model - Marginalized"
Authors: Byron J. Smith, Abhipsha Das, Oriol Abril-Pla
Description: A Dirichlet-multinomial model for overdispersed categorical count data
    (tree species observed across forests). Uses the marginalized DirichletMultinomial
    distribution with a Dirichlet prior on expected fractions and a Lognormal prior
    on the concentration parameter.

Changes from original:
- Inlined simulated observed_counts data (generated with RANDOM_SEED=8927)
- Removed sampling, posterior predictive, plotting, model comparison, and
  the intermediate multinomial and explicit Dirichlet-multinomial models

Benchmark results:
- Original:  logp = -128.7002, grad norm = 51.4717, 6.8 us/call (100000 evals)
- Frozen:    logp = -128.7002, grad norm = 51.4717, 6.7 us/call (100000 evals)
"""

import numpy as np
import pymc as pm

def build_model():
    trees = ["pine", "oak", "ebony", "rosewood", "mahogany"]
    forests = [
        "sunderbans", "amazon", "arashiyama", "trossachs", "valdivian",
        "bosc de poblet", "font groga", "monteverde", "primorye", "daintree",
    ]
    k = len(trees)
    total_count = 50

    observed_counts = np.array([
        [21,  9, 11,  6,  3],
        [36,  7,  6,  1,  0],
        [ 8, 31,  1, 10,  0],
        [25,  4, 17,  4,  0],
        [43,  6,  1,  0,  0],
        [28, 10, 12,  0,  0],
        [21, 16, 10,  3,  0],
        [16, 32,  2,  0,  0],
        [45,  4,  1,  0,  0],
        [35,  5,  2,  8,  0],
    ])

    coords = {"tree": trees, "forest": forests}
    with pm.Model(coords=coords) as model:
        frac = pm.Dirichlet("frac", a=np.ones(k), dims="tree")
        conc = pm.Lognormal("conc", mu=1, sigma=1)
        counts = pm.DirichletMultinomial(
            "counts", n=total_count, a=frac * conc, observed=observed_counts, dims=("forest", "tree")
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
