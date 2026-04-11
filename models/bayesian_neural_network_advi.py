"""
Model: Bayesian Neural Network (2-hidden-layer tanh MLP, Bernoulli output)
Source: pymc-examples/examples/variational_inference/bayesian_neural_network_advi.ipynb, Section: "Model specification"
Authors: Thomas Wiecki, updated by Chris Fonnesbeck
Description: Two-hidden-layer feed-forward neural network (5 tanh units per layer)
    with Normal(0, 1) priors on all weights and a Bernoulli likelihood, fit to the
    two-moons binary classification dataset (500 training points, 2 features).

Changes from original:
- Inlined moons dataset (generated deterministically and saved to data/bayesian_neural_network_moons.npz)
- Removed pm.Minibatch wrapper and total_size argument so the full training set is
    used as a single batch (deterministic logp for benchmarking). The underlying NN
    graph is otherwise identical.
- Removed ADVI fitting, posterior predictive, and plotting code.

Benchmark results:
- Original:  logp = -346.4279, grad norm = 469.3765, 66.3 us/call (100000 evals)
- Frozen:    logp = -346.4279, grad norm = 469.3765, 67.0 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data = np.load(Path(__file__).parent / "data" / "bayesian_neural_network_moons.npz")
    X_train = data["X_train"]
    Y_train = data["Y_train"]

    floatX = X_train.dtype
    n_hidden = 5

    # Initial values for the weights (match original notebook: rng seed 9927)
    rng = np.random.default_rng(9927)
    init_1 = rng.standard_normal(size=(X_train.shape[1], n_hidden)).astype(floatX)
    init_2 = rng.standard_normal(size=(n_hidden, n_hidden)).astype(floatX)
    init_out = rng.standard_normal(size=n_hidden).astype(floatX)

    coords = {
        "hidden_layer_1": np.arange(n_hidden),
        "hidden_layer_2": np.arange(n_hidden),
        "train_cols": np.arange(X_train.shape[1]),
        "obs_id": np.arange(X_train.shape[0]),
    }

    with pm.Model(coords=coords) as model:
        X_data = pm.Data("X_data", X_train, dims=("obs_id", "train_cols"))
        Y_data = pm.Data("Y_data", Y_train, dims="obs_id")

        # Weights from input to hidden layer
        weights_in_1 = pm.Normal(
            "w_in_1", 0, sigma=1, initval=init_1, dims=("train_cols", "hidden_layer_1")
        )

        # Weights from 1st to 2nd layer
        weights_1_2 = pm.Normal(
            "w_1_2", 0, sigma=1, initval=init_2, dims=("hidden_layer_1", "hidden_layer_2")
        )

        # Weights from hidden layer to output
        weights_2_out = pm.Normal(
            "w_2_out", 0, sigma=1, initval=init_out, dims="hidden_layer_2"
        )

        # Build neural network using tanh activation
        act_1 = pm.math.tanh(pm.math.dot(X_data, weights_in_1))
        act_2 = pm.math.tanh(pm.math.dot(act_1, weights_1_2))
        act_out = pm.math.sigmoid(pm.math.dot(act_2, weights_2_out))

        # Binary classification -> Bernoulli likelihood
        pm.Bernoulli("out", act_out, observed=Y_data, dims="obs_id")

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
