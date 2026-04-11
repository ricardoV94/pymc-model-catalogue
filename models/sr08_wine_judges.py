"""
Model: Wine Judges Item Response Model
Source: pymc-examples/examples/statistical_rethinking_lectures/08-Markov_Chain_Monte_Carlo.ipynb, Section: "Including Judge Effects"
Authors: Dustin Stansbury
Description: Item-response style model of standardized wine scores for the 2012 Wines dataset,
    with additive wine quality, wine origin, and judge harshness effects, modulated
    multiplicatively by a per-judge discrimination parameter. This is the most complete
    model in the lecture; the "simple" and "wine origin" models that precede it are
    intermediate building blocks.

Changes from original:
- Loaded data inline from a local .npz file instead of rethinking's utils loader
- Removed sampling, diagnostics, and plotting code

Benchmark results:
- Original:  logp = -293.8960, grad norm = 36.1598, 4.0 us/call (100000 evals)
- Frozen:    logp = -293.8960, grad norm = 36.1598, 3.8 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data = np.load(
        Path(__file__).parent / "data" / "sr08_wines2012.npz", allow_pickle=True
    )
    judge = data["judge"]
    wine = data["wine"]
    score = data["score"]
    wine_amer = data["wine_amer"]

    def standardize(x):
        return (x - np.nanmean(x)) / np.nanstd(x)

    SCORES = standardize(score)

    # Categorical judge ID
    JUDGE, JUDGE_IDX = np.unique(judge, return_inverse=True)
    JUDGE_ID = JUDGE_IDX.astype(np.int64)

    # Categorical wine ID
    WINE, WINE_IDX = np.unique(wine, return_inverse=True)
    WINE_ID = WINE_IDX.astype(np.int64)

    # Categorical wine origin (preserve first-seen order like pd.factorize(sort=False))
    origin_labels = np.array(["US" if w == 1.0 else "FR" for w in wine_amer])
    _, first_idx = np.unique(origin_labels, return_index=True)
    WINE_ORIGIN = origin_labels[np.sort(first_idx)]
    origin_to_id = {o: i for i, o in enumerate(WINE_ORIGIN)}
    WINE_ORIGIN_ID = np.array([origin_to_id[o] for o in origin_labels], dtype=np.int64)

    with pm.Model(
        coords={"wine": WINE, "wine_origin": WINE_ORIGIN, "judge": JUDGE}
    ) as model:

        # Judge effects
        D = pm.Exponential("D", 1, dims="judge")  # Judge Discrimination (multiplicative)
        H = pm.Normal("H", 0, 1, dims="judge")  # Judge Harshness (additive)

        # Wine Origin effect
        O = pm.Normal("O", 0, 1, dims="wine_origin")

        # Wine Quality effect
        Q = pm.Normal("Q", 0, 1, dims="wine")

        # Score
        sigma = pm.Exponential("sigma", 1)

        # Likelihood
        mu = (O[WINE_ORIGIN_ID] + Q[WINE_ID] - H[JUDGE_ID]) * D[JUDGE_ID]
        S = pm.Normal("S", mu, sigma, observed=SCORES)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
