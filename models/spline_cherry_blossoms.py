"""
Model: Cubic B-spline regression of cherry blossom bloom day on year
Source: pymc-examples/examples/howto/spline.ipynb, Section: "Fit the model"
Authors: Joshua Cook
Description: Bayesian cubic B-spline regression (15 interior knots, include_intercept=True
    giving 19 basis functions) fitting day-of-year of cherry blossom first bloom as a
    Normal response with linear-in-basis mean and Exponential noise scale.

Changes from original:
- Saved cherry_blossoms.csv (827 rows after dropna) to .npz; load inside build_model()
- Replaced patsy.dmatrix construction of the B-spline basis with an equivalent scipy
  BSpline construction (same knot placement: 15 interior knots at equally-spaced
  year quantiles; cubic degree; include_intercept=True -> 19 basis columns; rows
  sum to 1). Numerically verified to produce identical logp/grad as patsy.
- Removed sampling, prior/posterior predictive, and plotting code
- Added ip capture and initval clearing

Benchmark results:
- Original:  logp = -26280.3240, grad norm = 50281.1311, 7.9 us/call (100000 evals)
- Frozen:    logp = -26280.3240, grad norm = 50281.1311, 7.8 us/call (100000 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data = np.load(Path(__file__).parent / "data" / "spline_cherry_blossoms.npz")
    year = data["year"].astype(np.float64)
    doy = data["doy"].astype(np.float64)

    # Build cubic B-spline basis equivalent to:
    #   patsy.dmatrix("bs(year, knots=knots, degree=3, include_intercept=True) - 1",
    #                 {"year": year, "knots": knot_list})
    from scipy.interpolate import BSpline

    num_knots = 15
    knot_list = np.percentile(year, np.linspace(0, 100, num_knots + 2))[1:-1]
    degree = 3
    lb, ub = year.min(), year.max()
    t = np.concatenate(
        [np.repeat(lb, degree + 1), knot_list, np.repeat(ub, degree + 1)]
    )
    n_basis = len(t) - degree - 1  # 19 basis columns
    year_eval = year.copy()
    year_eval[year_eval == ub] = ub - 1e-9  # include right endpoint
    B = np.zeros((len(year_eval), n_basis))
    for i in range(n_basis):
        c = np.zeros(n_basis)
        c[i] = 1.0
        B[:, i] = BSpline(t, c, degree, extrapolate=False)(year_eval)

    COORDS = {"splines": np.arange(B.shape[1])}
    with pm.Model(coords=COORDS) as spline_model:
        a = pm.Normal("a", 100, 5)
        w = pm.Normal("w", mu=0, sigma=3, size=B.shape[1], dims="splines")

        mu = pm.Deterministic(
            "mu",
            a + pm.math.dot(np.asarray(B, order="F"), w.T),
        )
        sigma = pm.Exponential("sigma", 1)

        D = pm.Normal("D", mu=mu, sigma=sigma, observed=doy)

    ip = spline_model.initial_point()
    spline_model.rvs_to_initial_values = {rv: None for rv in spline_model.free_RVs}
    return spline_model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
