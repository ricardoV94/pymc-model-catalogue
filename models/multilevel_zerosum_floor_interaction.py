"""
Model: Multilevel Modeling - ZeroSumNormal Intercept and Floor Interaction
Source: nutpie README (https://github.com/pymc-devs/nutpie), "Usage with PyMC"
Authors: Adrian Seyboldt (nutpie)
Description: Hierarchical radon model parameterized with ZeroSumNormal county deviations.
    A global intercept and global floor effect are each augmented with county-level
    deviations expressed as a ZeroSumNormal auxiliary scaled by a HalfNormal standard
    deviation (a non-centered parameterization). Includes a county:floor interaction
    term, so the floor slope varies by county around the global floor effect.

Changes from original:
- Loaded preprocessed Minnesota radon data from .npz file instead of reading the
  bundled radon.csv (the catalogue's radon_mn.npz provides the same county indices,
  floor measure, and log_radon used by the other multilevel radon models).
- Removed sampling code.

Benchmark results:
- Original:  logp = -1842.1609, grad norm = 526.3650, 7.4 us/call (100000 evals)
- Frozen:    logp = -1842.1609, grad norm = 526.3650, 8.2 us/call (100000 evals)
"""

import numpy as np
import pymc as pm
from pathlib import Path
def build_model():
    data = np.load(Path(__file__).parent / "data" / "radon_mn.npz", allow_pickle=True)
    county_idx = data["county"]
    mn_counties = data["mn_counties"]
    floor_measure = data["floor_measure"].astype(np.float64)
    log_radon = data["log_radon"]
    coords = {"county": mn_counties, "obs_id": np.arange(len(county_idx))}

    with pm.Model(coords=coords, check_bounds=False) as zerosum_floor_interaction:
        intercept = pm.Normal("intercept", sigma=10)

        # County effects
        raw = pm.ZeroSumNormal("county_raw", dims="county")
        sd = pm.HalfNormal("county_sd")
        county_effect = pm.Deterministic("county_effect", raw * sd, dims="county")

        # Global floor effect
        floor_effect = pm.Normal("floor_effect", sigma=2)

        # County:floor interaction
        raw = pm.ZeroSumNormal("county_floor_raw", dims="county")
        sd = pm.HalfNormal("county_floor_sd")
        county_floor_effect = pm.Deterministic(
            "county_floor_effect", raw * sd, dims="county"
        )

        mu = (
            intercept
            + county_effect[county_idx]
            + floor_effect * floor_measure
            + county_floor_effect[county_idx] * floor_measure
        )

        sigma = pm.HalfNormal("sigma", sigma=1.5)
        pm.Normal(
            "log_radon", mu=mu, sigma=sigma, observed=log_radon, dims="obs_id"
        )

    ip = zerosum_floor_interaction.initial_point()
    zerosum_floor_interaction.rvs_to_initial_values = {rv: None for rv in zerosum_floor_interaction.free_RVs}
    return zerosum_floor_interaction, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
