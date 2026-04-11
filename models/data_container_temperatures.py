"""
Model: Hierarchical city temperatures with named dimensions
Source: pymc-examples/examples/fundamentals/data_container.ipynb, Section: "Named dimensions with data containers"
Authors: Juan Martin Loyola, Kavya Jaiswal, Oriol Abril, Jesse Grabowski
Description: Hierarchical model of daily temperatures in 3 European cities over 62 days.
    Each city's expected temperature equals a continent-wide mean plus a city-specific offset,
    observed via a Normal likelihood. Demonstrates pm.Data with (date, city) dims and coords.

Changes from original:
- Inlined synthetic temperature data (3 cities x 62 dates; seed from "Data Containers in PyMC")
- Removed sampling and plotting code

Benchmark results:
- Original:  logp = -582.6084, grad norm = 689.0065, 2.9 us/call (100000 evals)
- Frozen:    logp = -582.6084, grad norm = 689.0065, 2.6 us/call (100000 evals)
"""

import numpy as np
import pandas as pd
import pymc as pm


def build_model():
    RANDOM_SEED = sum(map(ord, "Data Containers in PyMC"))
    rng = np.random.default_rng(RANDOM_SEED)

    dates = pd.date_range(start="2020-05-01", end="2020-07-01")
    df_data = pd.DataFrame(columns=["date"]).set_index("date")
    for city, mu in {"Berlin": 15, "San Marino": 18, "Paris": 16}.items():
        df_data[city] = rng.normal(loc=mu, size=len(dates))
    df_data.index = dates
    df_data.index.name = "date"

    coords = {"date": df_data.index, "city": df_data.columns}

    with pm.Model(coords=coords) as model:
        data = pm.Data("observed_temp", df_data, dims=("date", "city"))

        europe_mean = pm.Normal("europe_mean_temp", mu=15.0, sigma=3.0)
        city_offset = pm.Normal("city_offset", mu=0.0, sigma=3.0, dims="city")
        city_temperature = pm.Deterministic(
            "expected_city_temp", europe_mean + city_offset, dims="city"
        )

        sigma = pm.Exponential("sigma", 1)
        pm.Normal(
            "temperature",
            mu=city_temperature,
            sigma=sigma,
            observed=data,
            dims=("date", "city"),
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
