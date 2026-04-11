"""
Model: Cat Adoption Survival Model with Right Censoring
Source: pymc-examples/examples/statistical_rethinking_lectures/09-Modeling_Events.ipynb, Section: "Survival Analysis"
Authors: Dustin Stansbury
Description: Exponential survival model for cat adoption times at Austin Animal Center,
    stratified by color (Black vs Other). Uses right censoring for cats not adopted.

Changes from original:
- Loaded data inline from npz instead of using utils.load_data
- Removed sampling and plotting code

Benchmark results:
- Original:  logp = -54312.1412, grad norm = 183.8183, 1820.5 us/call (8813 evals)
- Frozen:    logp = -54312.1412, grad norm = 183.8183, 2309.7 us/call (7909 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data = np.load(
        Path(__file__).parent / "data" / "sr_austin_cats.npz", allow_pickle=True
    )
    color = data["color"]
    out_event = data["out_event"]
    days_to_event = data["days_to_event"].astype(float)

    # Factorize color: Black vs Other
    cat_color_labels = np.array(["Other" if c != "Black" else c for c in color])
    unique_colors = []
    cat_color_id = np.empty(len(cat_color_labels), dtype=int)
    for i, c in enumerate(cat_color_labels):
        if c not in unique_colors:
            unique_colors.append(c)
        cat_color_id[i] = unique_colors.index(c)
    CAT_COLOR = unique_colors

    # Factorize adoption
    adopted_labels = np.array(
        ["Other" if e != "Adoption" else "Adopted" for e in out_event]
    )
    unique_adopted = []
    adopted_id = np.empty(len(adopted_labels), dtype=int)
    for i, a in enumerate(adopted_labels):
        if a not in unique_adopted:
            unique_adopted.append(a)
        adopted_id[i] = unique_adopted.index(a)

    LAMBDA = 50

    # Right censoring: non-adopted cats are censored at max days
    right_censoring = days_to_event.copy()
    right_censoring[adopted_id == 0] = days_to_event.max()

    with pm.Model(coords={"cat_color": CAT_COLOR}) as model:
        # Priors
        gamma = 1 / LAMBDA
        alpha = pm.Exponential("alpha", gamma, dims="cat_color")

        # Likelihood
        log_adoption_rate = 1 / alpha[cat_color_id]
        pm.Censored(
            "adopted",
            pm.Exponential.dist(lam=log_adoption_rate),
            lower=None,
            upper=right_censoring,
            observed=days_to_event,
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
