"""
Model: Baby Births HSGP (Model 3 - trend + seasonal + day-of-week)
Source: pymc-examples/examples/gaussian_processes/GP-Births.ipynb, Section: "Baby Births Modelling with HSGPs"
Authors: Juan Orduz
Description: HSGP model for relative number of births per day in the USA 1969-1988.
    Includes a slow trend GP (HSGP), yearly seasonal periodic GP (HSGPPeriodic),
    and day-of-week fixed effects.

Changes from original:
- Saved preprocessed data to .npz instead of downloading CSV and using sklearn
- Removed sampling, plotting, posterior predictive, and diagnostics
- Added ip capture and initval clearing

Benchmark results:
- Original:  logp = -16323.4021, grad norm = 24343.3142, 6165.2 us/call (2015 evals)
- Frozen:    logp = -16323.4021, grad norm = 24343.3142, 260.9 us/call (62323 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    data = np.load(
        Path(__file__).parent / "data" / "births_usa_1969.npz", allow_pickle=True
    )
    normalized_time = data["normalized_time"]
    day_of_week_idx = data["day_of_week_idx"]
    normalized_log_births_relative100 = data["normalized_log_births_relative100"]
    time_std = data["time_std"].item()
    day_of_week = data["day_of_week"]
    day_of_week_no_monday = data["day_of_week_no_monday"]
    day_of_year2 = data["day_of_year2"]
    time = np.arange(len(normalized_time))

    coords = {
        "time": time,
        "day_of_week_no_monday": day_of_week_no_monday,
        "day_of_week": day_of_week,
        "day_of_year2": day_of_year2,
    }

    with pm.Model(coords=coords) as model:
        # --- Data Containers ---
        normalized_time_data = pm.Data(
            name="normalized_time_data", value=normalized_time, dims="time"
        )

        day_of_week_idx_data = pm.Data(
            name="day_of_week_idx_data", value=day_of_week_idx, dims="time"
        )
        normalized_log_births_relative100_data = pm.Data(
            name="log_births_relative100",
            value=normalized_log_births_relative100,
            dims="time",
        )

        # --- Priors ---

        # global trend
        amplitude_trend = pm.HalfNormal(name="amplitude_trend", sigma=1.0)
        ls_trend = pm.LogNormal(name="ls_trend", mu=np.log(700 / time_std), sigma=1)
        cov_trend = amplitude_trend * pm.gp.cov.ExpQuad(input_dim=1, ls=ls_trend)
        gp_trend = pm.gp.HSGP(m=[20], c=1.5, cov_func=cov_trend)
        f_trend = gp_trend.prior(name="f_trend", X=normalized_time_data[:, None], dims="time")

        ## year periodic
        amplitude_year_periodic = pm.HalfNormal(name="amplitude_year_periodic", sigma=1)
        ls_year_periodic = pm.LogNormal(
            name="ls_year_periodic", mu=np.log(7_000 / time_std), sigma=1
        )
        gp_year_periodic = pm.gp.HSGPPeriodic(
            m=20,
            scale=amplitude_year_periodic,
            cov_func=pm.gp.cov.Periodic(
                input_dim=1, period=365.25 / time_std, ls=ls_year_periodic
            ),
        )
        f_year_periodic = gp_year_periodic.prior(
            name="f_year_periodic", X=normalized_time_data[:, None], dims="time"
        )

        ## day of week
        b_day_of_week_no_monday = pm.Normal(
            name="b_day_of_week_no_monday", sigma=1, dims="day_of_week_no_monday"
        )

        b_day_of_week = pm.Deterministic(
            name="b_day_of_week",
            var=pt.concatenate(([0], b_day_of_week_no_monday)),
            dims="day_of_week",
        )

        # global noise
        sigma = pm.HalfNormal(name="sigma", sigma=0.5)

        # --- Parametrization ---
        mu = pm.Deterministic(
            name="mu",
            var=f_trend
            + f_year_periodic
            + b_day_of_week[day_of_week_idx_data] * (day_of_week_idx_data > 0),
            dims="time",
        )

        # --- Likelihood ---
        pm.Normal(
            name="likelihood",
            mu=mu,
            sigma=sigma,
            observed=normalized_log_births_relative100_data,
            dims="time",
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
