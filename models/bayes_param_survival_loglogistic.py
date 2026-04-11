"""
Model: Log-Logistic Survival Regression
Source: pymc-examples/examples/survival_analysis/bayes_param_survival.ipynb, Section: "Log-logistic survival regression"
Authors: Austin Rochford, George Ho, Chris Fonnesbeck, Osvaldo Martin
Description: Log-logistic accelerated failure time survival regression model for mastectomy data.
    Models log survival times with logistic-distributed errors, incorporating metastization status
    as a covariate. Censored observations are handled via a logistic survival function potential.

Changes from original:
- Inlined mastectomy data as numpy arrays (44 observations).
- Removed sampling, posterior predictive, and plotting code.

Benchmark results:
- Original:  logp = -75.6729, grad norm = 24.7171, 5.2 us/call (100000 evals)
- Frozen:    logp = -75.6729, grad norm = 24.7171, 4.3 us/call (100000 evals)
"""

import numpy as np
import pymc as pm

def build_model():
    # Mastectomy data from HSAUR R package
    time = np.array(
        [23, 47, 69, 70, 100, 101, 148, 181, 198, 208, 212, 224,
         5, 8, 10, 13, 18, 24, 26, 26, 31, 35, 40, 41, 48, 50,
         59, 61, 68, 71, 76, 105, 107, 109, 113, 116, 118, 143,
         145, 162, 188, 212, 217, 225], dtype=float)
    event = np.array(
        [1., 1., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0.,
         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 1., 0.,
         1., 1., 0., 0., 0., 0., 0., 0.])
    metastized = np.array(
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 1., 1., 1., 1., 1., 1., 1.])

    n_patient = len(time)
    X = np.empty((n_patient, 2))
    X[:, 0] = 1.0
    X[:, 1] = metastized

    y = np.log(time)
    y_std = (y - y.mean()) / y.std()

    def logistic_sf(y, mu, s):
        return 1.0 - pm.math.sigmoid((y - mu) / s)

    with pm.Model() as model:
        predictors = pm.Data("predictors", X)
        censored = pm.Data("censored", event == 0.0)
        y_obs = pm.Data("y_obs", y_std[event == 1.0])
        y_cens = pm.Data("y_cens", y_std[event == 0.0])

        beta = pm.Normal("beta", 0.0, 5.0, shape=2)
        eta = beta.dot(predictors.T)

        s = pm.HalfNormal("s", 5.0)

        events = pm.Logistic("events", eta[~censored], s, observed=y_obs)
        censored_like = pm.Potential("censored_like", logistic_sf(y_cens, eta[censored], s))

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip
if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
