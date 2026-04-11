"""
Model: Excess Deaths due to COVID-19
Source: pymc-examples/examples/causal_inference/excess_deaths.ipynb, Section: "Modelling"
Authors: Benjamin T. Vincent
Description: Linear regression model for monthly deaths in England and Wales (pre-COVID),
    using month seasonality (ZeroSumNormal), linear trend, and temperature as predictors.
    TruncatedNormal likelihood ensures non-negative death counts.

Changes from original:
- pm.MutableData -> pm.Data (API update)
- Data variable "month" renamed to "month_idx" to avoid conflict with coord name "month"
- Data loaded from .npz instead of CSV (pre-COVID subset only)
- Removed sampling, plotting, counterfactual inference, and helper functions
- ZeroSumNormal helper function inlined
- Removed mutable=False from add_coord (no longer accepted)

Benchmark results:
- Original:  logp = -2168.9651, grad norm = 1068.5490, 17.7 us/call (100000 evals)
- Frozen:    logp = -2168.9651, grad norm = 1068.5490, 15.6 us/call (100000 evals)
"""

import calendar
from pathlib import Path

import numpy as np
import pymc as pm
import pytensor.tensor as pt

def build_model():
    # ZeroSumNormal helper (from original notebook)
    def ZeroSumNormal(name, *, sigma=None, active_dims=None, dims, model=None):
        _model = pm.modelcontext(model=model)

        if isinstance(dims, str):
            dims = [dims]

        if isinstance(active_dims, str):
            active_dims = [active_dims]

        if active_dims is None:
            active_dims = dims[-1]

        def extend_axis(value, axis):
            n_out = value.shape[axis] + 1
            sum_vals = value.sum(axis, keepdims=True)
            norm = sum_vals / (pt.sqrt(n_out) + n_out)
            fill_val = norm - sum_vals / pt.sqrt(n_out)
            out = pt.concatenate([value, fill_val], axis=axis)
            return out - norm

        dims_reduced = []
        active_axes = []
        for i, dim in enumerate(dims):
            if dim in active_dims:
                active_axes.append(i)
                dim_name = f"{dim}_reduced"
                if name not in _model.coords:
                    _model.add_coord(dim_name, length=len(_model.coords[dim]) - 1)
                dims_reduced.append(dim_name)
            else:
                dims_reduced.append(dim)

        raw = pm.Normal(f"{name}_raw", sigma=sigma, dims=dims_reduced)
        for axis in active_axes:
            raw = extend_axis(raw, axis)
        return pm.Deterministic(name, raw, dims=dims)

    # Load pre-COVID data
    data = np.load(Path(__file__).parent / "data" / "deaths_and_temps_england_wales.npz")
    month_data = data["month"]
    time_data = data["t"]
    temp_data = data["temp"]
    deaths_data = data["deaths"]

    month_strings = calendar.month_name[1:]

    with pm.Model(coords={"month": month_strings}) as model:
        # observed predictors and outcome
        month = pm.Data("month_idx", month_data, dims="t")
        time = pm.Data("time", time_data, dims="t")
        temp = pm.Data("temp", temp_data, dims="t")
        deaths = pm.Data("deaths", deaths_data, dims="t")

        # priors
        intercept = pm.Normal("intercept", 40_000, 10_000)
        month_mu = ZeroSumNormal("month mu", sigma=3000, dims="month")
        linear_trend = pm.TruncatedNormal("linear trend", 0, 50, lower=0)
        temp_coeff = pm.Normal("temp coeff", 0, 200)

        # the actual linear model
        mu = pm.Deterministic(
            "mu",
            intercept + (linear_trend * time) + month_mu[month - 1] + (temp_coeff * temp),
            dims="t",
        )
        sigma = pm.HalfNormal("sigma", 2_000)
        # likelihood
        pm.TruncatedNormal("obs", mu=mu, sigma=sigma, lower=0, observed=deaths, dims="t")

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
