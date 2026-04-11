"""
Model: Blackbox external likelihood wrapped as a custom PyTensor Op (with finite-difference gradient)
Source: pymc-examples/examples/howto/blackbox_external_likelihood_numpy.ipynb, Section: "PyTensor Op with gradients"
Authors: Matt Pitkin, Jørgen Midtbø, Oriol Abril, Ricardo Vieira
Description: Simple linear regression whose log-likelihood is implemented as a
    numpy function and wrapped with a custom PyTensor `Op` (`LogLikeWithGrad`)
    that defines a gradient via scipy's `approx_fprime` (finite differences)
    delegated to a second `LogLikeGrad` Op. Used via `pm.CustomDist`.

Changes from original:
- Inlined helper functions and the custom Op classes inside build_model()
- Relaxed the `grad()` scalar-only check: current PyMC's CustomDist may pass
  m/c in with extra batch dims, so we squeeze to scalars inside grad before
  calling the finite-difference Op.
- Removed sampling/plotting code
- Added ip capture + initval clearing boilerplate

Benchmark results:
- Original:  logp = -146.3773, grad norm = 1310.7227, 68.2 us/call (95307 evals)
- Frozen:    logp = -146.3773, grad norm = 1310.7227, 67.8 us/call (100000 evals)
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

            # Current PyMC's CustomDist may broadcast m/c to shape (1,) instead of ()
            # before passing them in; squeeze back to scalars for the numpy Op call and
            # then shape the scalar result back to match the input ndim.
            m_scalar = m.squeeze() if m.type.ndim > 0 else m
            c_scalar = c.squeeze() if c.type.ndim > 0 else c

            grad_wrt_m, grad_wrt_c = loglikegrad_op(m_scalar, c_scalar, sigma, x, data)

            [out_grad] = g
            m_grad = pt.zeros_like(m) + pt.sum(out_grad * grad_wrt_m)
            c_grad = pt.zeros_like(c) + pt.sum(out_grad * grad_wrt_c)
            return [
                m_grad,
                c_grad,
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
    N = 10  # number of data points
    sigma = 1.0  # standard deviation of noise
    x = np.linspace(0.0, 9.0, N)

    mtrue = 0.4  # true gradient
    ctrue = 3.0  # true y-intercept

    truemodel = my_model(mtrue, ctrue, x)

    rng = np.random.default_rng(716743)
    data = sigma * rng.normal(size=N) + truemodel

    def custom_dist_loglike(data, m, c, sigma, x):
        return loglikewithgrad_op(m, c, sigma, x, data)

    with pm.Model() as model:
        m = pm.Uniform("m", lower=-10.0, upper=10.0)
        c = pm.Uniform("c", lower=-10.0, upper=10.0)

        pm.CustomDist(
            "likelihood", m, c, sigma, x, observed=data, logp=custom_dist_loglike
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
