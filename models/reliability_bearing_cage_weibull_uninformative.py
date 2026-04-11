"""
Model: Weibull Reliability Model for Bearing Cage Data (Uninformative Prior)
Source: pymc-examples/examples/case_studies/reliability_and_calibrated_prediction.ipynb, Section: "Direct PYMC implementation of Weibull Survival"
Authors: Nathaniel Forde
Description: Bayesian Weibull reliability fit for the Bearing Cage dataset with
    vague priors (Uniform beta, TruncatedNormal alpha). Censored observations are
    handled via a Potential using the Weibull log complementary CDF.

Changes from original:
- Inlined bearing-cage aggregated counts; censored array reconstructed via np.repeat.
- Removed sampling/plotting code.

Benchmark results:
- Original:  logp = -92.7707, grad norm = 28.3562, 35.7 us/call (100000 evals)
- Frozen:    logp = -92.7707, grad norm = 28.3562, 34.9 us/call (100000 evals)
"""

import numpy as np
import pymc as pm


def build_model():
    # Bearing cage data (Meeker & Escobar). Aggregated period-format counts.
    # Censored (hours, count) pairs
    censored_hours = np.array(
        [50, 150, 250, 350, 450, 550, 650, 750, 850, 950,
         1050, 1150, 1250, 1350, 1450, 1550, 1650, 1850, 2050],
        dtype=float,
    )
    censored_counts = np.array(
        [288, 148, 124, 111, 106, 99, 110, 114, 119, 127,
         123, 93, 47, 41, 27, 11, 6, 1, 2],
        dtype=int,
    )
    # Expand to item-level via repeat. Each id's max time equals its censor hour.
    y_cens = np.repeat(censored_hours, censored_counts)

    # Failure times (6 failures)
    y_obs_data = np.array([230.0, 334.0, 423.0, 990.0, 1009.0, 1510.0])

    def weibull_lccdf(y, alpha, beta):
        """Log complementary cdf of Weibull distribution."""
        return -((y / beta) ** alpha)

    priors = {"beta": [100, 15_000], "alpha": [4, 1, 0.02, 8]}

    with pm.Model() as model:
        beta = pm.Uniform("beta", priors["beta"][0], priors["beta"][1])
        alpha = pm.TruncatedNormal(
            "alpha",
            priors["alpha"][0],
            priors["alpha"][1],
            lower=priors["alpha"][2],
            upper=priors["alpha"][3],
        )

        y_obs = pm.Weibull("y_obs", alpha=alpha, beta=beta, observed=y_obs_data)
        y_cens_pot = pm.Potential("y_cens", weibull_lccdf(y_cens, alpha, beta))

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
