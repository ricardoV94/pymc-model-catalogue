"""
Model: Weibull Reliability Model for Bearing Cage Data (Informative Prior)
Source: pymc-examples/examples/case_studies/reliability_and_calibrated_prediction.ipynb, Section: "Direct PYMC implementation of Weibull Survival"
Authors: Nathaniel Forde
Description: Bayesian Weibull reliability fit for the Bearing Cage dataset with
    priors closer to the MLE fits (Normal beta, tighter TruncatedNormal alpha).
    Censored observations are handled via a Potential using the Weibull log
    complementary CDF.

Changes from original:
- Inlined bearing-cage aggregated counts; censored array reconstructed via np.repeat.
- Removed sampling/plotting code.

Benchmark results:
- Original:  logp = -98.3856, grad norm = 43.5559, 35.5 us/call (100000 evals)
- Frozen:    logp = -98.3856, grad norm = 43.5559, 36.2 us/call (100000 evals)
"""

import numpy as np
import pymc as pm


def build_model():
    # Bearing cage data (Meeker & Escobar). Aggregated period-format counts.
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
    y_cens = np.repeat(censored_hours, censored_counts)

    y_obs_data = np.array([230.0, 334.0, 423.0, 990.0, 1009.0, 1510.0])

    def weibull_lccdf(y, alpha, beta):
        """Log complementary cdf of Weibull distribution."""
        return -((y / beta) ** alpha)

    priors_informative = {"beta": [10_000, 500], "alpha": [2, 0.5, 0.02, 3]}

    with pm.Model() as model:
        beta = pm.Normal("beta", priors_informative["beta"][0], priors_informative["beta"][1])
        alpha = pm.TruncatedNormal(
            "alpha",
            priors_informative["alpha"][0],
            priors_informative["alpha"][1],
            lower=priors_informative["alpha"][2],
            upper=priors_informative["alpha"][3],
        )

        y_obs = pm.Weibull("y_obs", alpha=alpha, beta=beta, observed=y_obs_data)
        y_cens_pot = pm.Potential("y_cens", weibull_lccdf(y_cens, alpha, beta))

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
