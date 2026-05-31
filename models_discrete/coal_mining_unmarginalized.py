"""
Model: Coal Mining Disasters Switchpoint (unmarginalized)
Source: pymc-examples/examples/howto/marginalizing-models.ipynb, Section: "Coal mining model"
Authors: Rob Zinkov
Description: Classic coal mining disasters switchpoint model with the discrete `switchpoint`
    kept as a free variable (no marginalization). Free variables are the discrete
    `switchpoint` DiscreteUniform, the continuous `early_rate` / `late_rate` Exponential
    priors, and a Poisson likelihood on the disaster counts. The counts contain NaNs, so
    PyMC also imputes the two missing observations as a discrete `disasters_unobserved`
    free RV. Discrete model — benchmarked logp-only, no gradient.

    Companion to `coal_mining_marginalized.py`, which marginalizes `switchpoint` out via
    `pymc_extras.marginalize`. This file is the same model before that call.

Changes from original:
- Inlined coal mining disaster data
- Removed sampling, plotting, and `recover_marginals` call

Benchmark results:
- Original:  logp = -231.8252, 2.1 us/call (100000 evals)
- Frozen:    logp = -231.8252, 2.2 us/call (100000 evals)
"""

import numpy as np
import pandas as pd
import pymc as pm


def build_model():
    # fmt: off
    disaster_data = pd.Series(
        [4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
        3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
        2, 2, 3, 4, 2, 1, 3, np.nan, 2, 1, 1, 1, 1, 3, 0, 0,
        1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
        0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
        3, 3, 1, np.nan, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]
    )
    # fmt: on
    years = np.arange(1851, 1962)

    with pm.Model() as model:
        switchpoint = pm.DiscreteUniform("switchpoint", lower=years.min(), upper=years.max())
        early_rate = pm.Exponential("early_rate", 1.0)
        late_rate = pm.Exponential("late_rate", 1.0)
        rate = pm.math.switch(switchpoint >= years, early_rate, late_rate)
        disasters = pm.Poisson("disasters", rate, observed=disaster_data)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model, discrete=True)
