"""
Model: Cubic B-spline regression of cherry blossoms with pm.Data containers
Source: pymc-examples/examples/howto/spline.ipynb, Section: "Predicting on new data"
Authors: Joshua Cook
Description: Same cubic B-spline regression as spline_cherry_blossoms.py but wraps
    year, doy, and the spline design matrix in pm.Data containers with named
    ("obs", "spline") dims so the model supports set_data-based prediction. Uses
    include_intercept=False, giving 18 basis columns.

Changes from original:
- Saved cherry_blossoms.csv (827 rows after dropna) to .npz; load inside build_model()
- Replaced patsy.dmatrix construction of the B-spline basis with an equivalent scipy
  BSpline construction (same knot placement: 15 interior knots at equally-spaced
  year quantiles; cubic degree; include_intercept=False -> 18 basis columns).
  Numerically verified to produce identical logp/grad as patsy. The precomputed
  basis is still passed to pm.Data as in the original, preserving the data
  container dims.
- Dropped the attachment of knots / design_info to spline_model (only used for
  out-of-sample prediction in the notebook, not for logp)
- Removed sampling and posterior predictive code
- Added ip capture and initval clearing

Benchmark results:
- Original:  logp = -26278.3065, grad norm = 50281.1302, 16.1 us/call (100000 evals)
- Frozen:    logp = -26278.3065, grad norm = 50281.1302, 8.9 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data = np.load(Path(__file__).parent / "data" / "spline_cherry_blossoms.npz")
    year = data["year"].astype(np.float64)
    doy_arr = data["doy"].astype(np.float64)

    # Build cubic B-spline basis equivalent to:
    #   patsy.dmatrix("bs(x, knots=knots, degree=3, include_intercept=False) - 1",
    #                 {"x": year})
    from scipy.interpolate import BSpline

    num_knots = 15
    knot_list = np.percentile(year, np.linspace(0, 100, num_knots + 2))[1:-1]
    degree = 3
    lb, ub = year.min(), year.max()
    # include_intercept=True knot vector
    t_full = np.concatenate(
        [np.repeat(lb, degree + 1), knot_list, np.repeat(ub, degree + 1)]
    )
    n_basis_full = len(t_full) - degree - 1  # 19
    year_eval = year.copy()
    year_eval[year_eval == ub] = ub - 1e-9
    B_full = np.zeros((len(year_eval), n_basis_full))
    for i in range(n_basis_full):
        c = np.zeros(n_basis_full)
        c[i] = 1.0
        B_full[:, i] = BSpline(t_full, c, degree, extrapolate=False)(year_eval)
    # Drop first column to match patsy include_intercept=False
    dm = B_full[:, 1:]  # shape (827, 18)

    COORDS = {"obs": np.arange(len(year))}
    with pm.Model(coords=COORDS) as spline_model:
        year_data = pm.Data("year", year)
        doy = pm.Data("doy", doy_arr)

        # intercept
        a = pm.Normal("a", 100, 5)

        # Create spline bases & coefficients
        spline_model.add_coords({"spline": np.arange(dm.shape[1])})
        splines_basis = pm.Data(
            "splines_basis", np.asarray(dm), dims=("obs", "spline")
        )
        w = pm.Normal("w", mu=0, sigma=3, dims="spline")

        mu = pm.Deterministic(
            "mu",
            a + pm.math.dot(splines_basis, w),
        )
        sigma = pm.Exponential("sigma", 1)

        D = pm.Normal("D", mu=mu, sigma=sigma, observed=doy)

    ip = spline_model.initial_point()
    spline_model.rvs_to_initial_values = {rv: None for rv in spline_model.free_RVs}
    return spline_model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
