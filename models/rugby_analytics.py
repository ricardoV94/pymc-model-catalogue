"""
Model: Hierarchical Poisson Model for Rugby Prediction
Source: pymc-examples/examples/case_studies/rugby_analytics.ipynb, Section: "Building of the model"
Authors: Peadar Coyle, Meenal Jhajharia, Oriol Abril-Pla
Description: Hierarchical Poisson model for Six Nations rugby scores with team-specific attack/defense
    effects, a home advantage parameter, and a sum-to-zero constraint on team effects.

Changes from original:
- Inlined rugby.csv data (60 matches) as numpy arrays
- Used pd.factorize on inlined data to compute home_idx/away_idx
- Removed sampling, plotting, and posterior predictive code

Benchmark results:
- Original:  logp = -787.0960, grad norm = 570.5859, 6.6 us/call (100000 evals)
- Frozen:    logp = -787.0960, grad norm = 570.5859, 6.7 us/call (100000 evals)
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

def build_model():
    # Six Nations rugby data 2014-2017 (60 matches)
    # fmt: off
    home_teams = [
        "Wales", "France", "Ireland", "Ireland", "Scotland", "France",
        "Wales", "Italy", "England", "Ireland", "Scotland", "England",
        "Italy", "Wales", "France", "Wales", "Italy", "France",
        "England", "Ireland", "Scotland", "Scotland", "France", "Ireland",
        "Wales", "England", "Italy", "Italy", "Scotland", "England",
        "France", "Scotland", "Ireland", "France", "Wales", "Italy",
        "Wales", "Italy", "England", "Ireland", "England", "Scotland",
        "Wales", "Ireland", "France", "Scotland", "England", "Italy",
        "Italy", "Wales", "France", "Scotland", "Ireland", "England",
        "Wales", "Italy", "England", "Scotland", "France", "Ireland",
    ]
    away_teams = [
        "Italy", "England", "Scotland", "Wales", "England", "Italy",
        "France", "Scotland", "Ireland", "Italy", "France", "Wales",
        "England", "Scotland", "Ireland", "England", "Ireland", "Scotland",
        "Italy", "France", "Wales", "Italy", "Wales", "England",
        "Ireland", "Scotland", "France", "Wales", "Ireland", "France",
        "Italy", "England", "Wales", "Ireland", "Scotland", "England",
        "France", "Scotland", "Ireland", "Italy", "Wales", "France",
        "Italy", "Scotland", "England", "Ireland", "France", "Wales",
        "Ireland", "England", "Scotland", "Wales", "France", "Italy",
        "Ireland", "France", "Scotland", "Italy", "Wales", "England",
    ]
    home_score = np.array([
        23, 26, 28, 26,  0, 30, 27, 20, 13, 46, 17, 29, 11, 51, 20,
        16,  3, 15, 47, 18, 23, 19, 13, 19, 23, 25,  0, 20, 10, 55,
        23,  9, 16, 10, 27,  9, 19, 20, 21, 58, 25, 29, 67, 35, 21,
        27, 19,  7, 10, 16, 22, 29, 19, 36, 22, 18, 61, 29, 20, 13,
    ])
    away_score = np.array([
        15, 24,  6,  3, 20, 10,  6, 21, 10,  7, 19, 18, 52,  3, 22,
        21, 26,  8, 17, 11, 26, 22, 20,  9, 16, 13, 29, 61, 40, 35,
        21, 15, 16,  9, 23, 40, 10, 36, 10, 15, 21, 18, 14, 25, 31,
        22, 16, 33, 63, 21, 16, 13,  9, 15,  9, 40, 21,  0, 18,  9,
    ])
    # fmt: on

    df_all = pd.DataFrame({
        "home_team": home_teams,
        "away_team": away_teams,
        "home_score": home_score,
        "away_score": away_score,
    })

    home_idx, teams = pd.factorize(df_all["home_team"], sort=True)
    away_idx, _ = pd.factorize(df_all["away_team"], sort=True)
    coords = {"team": teams}

    with pm.Model(coords=coords) as model:
        # constant data
        home_team = pm.Data("home_team", home_idx, dims="match")
        away_team = pm.Data("away_team", away_idx, dims="match")

        # global model parameters
        home = pm.Normal("home", mu=0, sigma=1)
        sd_att = pm.HalfNormal("sd_att", sigma=2)
        sd_def = pm.HalfNormal("sd_def", sigma=2)
        intercept = pm.Normal("intercept", mu=3, sigma=1)

        # team-specific model parameters
        atts_star = pm.Normal("atts_star", mu=0, sigma=sd_att, dims="team")
        defs_star = pm.Normal("defs_star", mu=0, sigma=sd_def, dims="team")

        atts = pm.Deterministic("atts", atts_star - pt.mean(atts_star), dims="team")
        defs = pm.Deterministic("defs", defs_star - pt.mean(defs_star), dims="team")
        home_theta = pt.exp(intercept + home + atts[home_idx] + defs[away_idx])
        away_theta = pt.exp(intercept + atts[away_idx] + defs[home_idx])

        # likelihood of observed data
        home_points = pm.Poisson(
            "home_points",
            mu=home_theta,
            observed=df_all["home_score"].values,
            dims=("match"),
        )
        away_points = pm.Poisson(
            "away_points",
            mu=away_theta,
            observed=df_all["away_score"].values,
            dims=("match"),
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip

if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
