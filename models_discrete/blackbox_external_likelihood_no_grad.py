"""
Model: Blackbox external likelihood wrapped as a custom PyTensor Op (no gradient)
Source: pymc-examples/examples/howto/blackbox_external_likelihood_numpy.ipynb, Section: "PyTensor Op without gradients"
Authors: Matt Pitkin, Jørgen Midtbø, Oriol Abril, Ricardo Vieira
Description: Simple linear regression whose log-likelihood is implemented as a
    numpy function and wrapped with a custom PyTensor `Op` (`LogLike`) without
    any defined gradient. Used via `pm.CustomDist` with `logp=...`.

Has discrete variables: No, but the custom `LogLike` Op has no defined gradient
    (pullback not implemented), so the model is benchmarked via `compile_logp`
    (no dlogp) alongside the discrete/simulator models.

Changes from original:
- Inlined helper functions and the custom Op class inside build_model()
- Removed sampling/plotting code
- Added ip capture + initval clearing boilerplate

Benchmark results:
- Original:  logp = -15.2542, 10.2 us/call (100000 evals)
- Frozen:    logp = -15.2542, 10.2 us/call (100000 evals)
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from pytensor.graph import Apply, Op


def build_model():
    def my_model(m, c, x):
        return m * x + c

    def my_loglike(m, c, sigma, x, data):
        for param in (m, c, sigma, x, data):
            if not isinstance(param, (float, np.ndarray)):
                raise TypeError(f"Invalid input type to loglike: {type(param)}")
        model = my_model(m, c, x)
        return -0.5 * ((data - model) / sigma) ** 2 - np.log(np.sqrt(2 * np.pi)) - np.log(sigma)

    class LogLike(Op):
        def make_node(self, m, c, sigma, x, data) -> Apply:
            m = pt.as_tensor(m)
            c = pt.as_tensor(c)
            sigma = pt.as_tensor(sigma)
            x = pt.as_tensor(x)
            data = pt.as_tensor(data)

            inputs = [m, c, sigma, x, data]
            outputs = [data.type()]
            return Apply(self, inputs, outputs)

        def perform(self, node, inputs, outputs):
            m, c, sigma, x, data = inputs
            loglike_eval = my_loglike(m, c, sigma, x, data)
            outputs[0][0] = np.asarray(loglike_eval)

    loglike_op = LogLike()

    # set up our data
    N = 10  # number of data points
    sigma = 1.0  # standard deviation of noise
    x = np.linspace(0.0, 9.0, N)

    mtrue = 0.4  # true gradient
    ctrue = 3.0  # true y-intercept

    truemodel = my_model(mtrue, ctrue, x)

    rng = np.random.default_rng(716743)
    data = sigma * rng.normal(size=N) + truemodel

    def custom_dist_loglike(data, m, c, sigma, x):
        # data, or observed, is always passed as the first input of CustomDist
        return loglike_op(m, c, sigma, x, data)

    with pm.Model() as model:
        m = pm.Uniform("m", lower=-10.0, upper=10.0, initval=mtrue)
        c = pm.Uniform("c", lower=-10.0, upper=10.0, initval=ctrue)

        pm.CustomDist(
            "likelihood", m, c, sigma, x, observed=data, logp=custom_dist_loglike
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model, discrete=True)
