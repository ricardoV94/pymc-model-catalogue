"""
Model: Multinomial logistic regression on Iris
Source: pymc-examples/examples/variational_inference/variational_api_quickstart.ipynb, Section: "Multilabel logistic regression"
Authors: Maxim Kochurov, Chris Fonnesbeck
Description: Multinomial logistic regression fit to the Iris dataset (4 features, 3 classes)
    using a wide Normal(0, 1e2) prior on the 4x3 coefficient matrix and Normal(0, 1e4)
    intercepts. Softmax link with Categorical observed likelihood.

Changes from original:
- Pre-computed the sklearn train/test split with random_state=0 and saved the train
  arrays to models/data/iris_train.npz (sklearn is not a dependency here)
- Replaced pytensor.shared wrappers (Xt, yt) with plain numpy arrays, since they are not
  mutated in the benchmark (the VI test-set replacement trick is not part of the logp graph)
- Removed VI/SVGD fitting, callbacks, and plotting code

Benchmark results:
- Original:  logp = -219.7217, grad norm = 152.2337, 11.8 us/call (100000 evals)
- Frozen:    logp = -219.7217, grad norm = 152.2337, 11.8 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    data = np.load(Path(__file__).parent / "data" / "iris_train.npz")
    X_train = data["X_train"].astype(np.float64)
    y_train = data["y_train"].astype(np.int64)

    with pm.Model() as model:
        # Coefficients for features
        beta = pm.Normal("β", 0, sigma=1e2, shape=(4, 3))
        # Intercepts
        a = pm.Normal("a", sigma=1e4, shape=(3,))
        p = pt.special.softmax(pt.dot(X_train, beta) + a, axis=-1)

        observed = pm.Categorical("obs", p=p, observed=y_train)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
