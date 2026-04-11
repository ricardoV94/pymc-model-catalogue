"""
Model: Blackbox external likelihood via pm.Potential (with finite-difference gradient)
Source: pymc-examples/examples/howto/blackbox_external_likelihood_numpy.ipynb, Section: "Using a Potential instead of CustomDist"
Authors: Matt Pitkin, Jørgen Midtbø, Oriol Abril, Ricardo Vieira
Description: Same linear regression / blackbox log-likelihood as the CustomDist
    version, but uses `pm.Potential` instead of `pm.CustomDist` to add the
    blackbox log-likelihood term. Relies on `LogLikeWithGrad` (finite-difference
    gradient via `LogLikeGrad`) so NUTS is usable.

Changes from original:
- Inlined helper functions and the custom Op classes inside build_model()
- Removed sampling/plotting code
- Added ip capture + initval clearing boilerplate

Benchmark results:
- Original:  logp = -146.3773, grad norm = 1310.7227, 62.6 us/call (100000 evals)
- Frozen:    logp = -146.3773, grad norm = 1310.7227, 63.3 us/call (71226 evals)
"""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from pytensor.graph import Apply, Op
from scipy.optimize import approx_fprime


def build_model():
    def my_model(m, c, x):
        return m * x + c

    def my_loglike(m, c, sigma, x, data):
        for param in (m, c, sigma, x, data):
            if not isinstance(param, (float, np.ndarray)):
                raise TypeError(f"Invalid input type to loglike: {type(param)}")
        model = my_model(m, c, x)
        return -0.5 * ((data - model) / sigma) ** 2 - np.log(np.sqrt(2 * np.pi)) - np.log(sigma)

    def finite_differences_loglike(m, c, sigma, x, data, eps=1e-7):
        def inner_func(mc, sigma, x, data):
            return my_loglike(*mc, sigma, x, data)

        grad_wrt_mc = approx_fprime([m, c], inner_func, [eps, eps], sigma, x, data)
        return grad_wrt_mc[:, 0], grad_wrt_mc[:, 1]

    class LogLikeWithGrad(Op):
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

        def grad(self, inputs, g):
            m, c, sigma, x, data = inputs

            if m.type.ndim != 0 or c.type.ndim != 0:
                raise NotImplementedError("Gradient only implemented for scalar m and c")

            grad_wrt_m, grad_wrt_c = loglikegrad_op(m, c, sigma, x, data)

            [out_grad] = g
            return [
                pt.sum(out_grad * grad_wrt_m),
                pt.sum(out_grad * grad_wrt_c),
                pytensor.gradient.grad_not_implemented(self, 2, sigma),
                pytensor.gradient.grad_not_implemented(self, 3, x),
                pytensor.gradient.grad_not_implemented(self, 4, data),
            ]

    class LogLikeGrad(Op):
        def make_node(self, m, c, sigma, x, data) -> Apply:
            m = pt.as_tensor(m)
            c = pt.as_tensor(c)
            sigma = pt.as_tensor(sigma)
            x = pt.as_tensor(x)
            data = pt.as_tensor(data)

            inputs = [m, c, sigma, x, data]
            outputs = [data.type(), data.type()]
            return Apply(self, inputs, outputs)

        def perform(self, node, inputs, outputs):
            m, c, sigma, x, data = inputs
            grad_wrt_m, grad_wrt_c = finite_differences_loglike(m, c, sigma, x, data)
            outputs[0][0] = grad_wrt_m
            outputs[1][0] = grad_wrt_c

    loglikewithgrad_op = LogLikeWithGrad()
    loglikegrad_op = LogLikeGrad()

    # set up our data
    N = 10
    sigma = 1.0
    x = np.linspace(0.0, 9.0, N)

    mtrue = 0.4
    ctrue = 3.0

    truemodel = my_model(mtrue, ctrue, x)

    rng = np.random.default_rng(716743)
    data = sigma * rng.normal(size=N) + truemodel

    with pm.Model() as model:
        m = pm.Uniform("m", lower=-10.0, upper=10.0)
        c = pm.Uniform("c", lower=-10.0, upper=10.0)

        pm.Potential("likelihood", loglikewithgrad_op(m, c, sigma, x, data))

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
