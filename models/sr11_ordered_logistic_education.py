"""
Model: Ordered Logistic Regression with Education, Gender, and Age Effects
Source: pymc-examples/examples/statistical_rethinking_lectures/11-Ordered_Categories.ipynb, Section: "Direct Effect of Education"
Authors: Dustin Stansbury
Description: Ordered logistic model of trolley problem moral judgments. Includes gender-stratified
    effects for action, intention, contact, education, and age. Education levels are modeled
    via cumulative Dirichlet-distributed deltas to respect ordinal structure.

Changes from original:
- univariate_ordered -> ordered (API update)
- Loaded data from npz instead of using utils.load_data
- Created education level mapping inline
- Standardized age inline
- Removed sampling and plotting code

Benchmark results:
- Original:  logp = -19711.8059, grad norm = 2910.6450, 1062.9 us/call (13846 evals)
- Frozen:    logp = -19711.8059, grad norm = 2910.6450, 1087.9 us/call (14233 evals)
"""

from pathlib import Path

import numpy as np
import pymc as pm


def build_model():
    data = np.load(
        Path(__file__).parent / "data" / "sr_trolley.npz", allow_pickle=True
    )
    response = data["response"].astype(int)
    action = data["action"].astype(float)
    intention = data["intention"].astype(float)
    contact = data["contact"].astype(float)
    male = data["male"]
    age_raw = data["age"].astype(float)
    edu = data["edu"]

    # Factorize response
    unique_response = np.sort(np.unique(response))
    response_id = np.searchsorted(unique_response, response)
    N_RESPONSE_CLASSES = len(unique_response)

    # Factorize gender
    gender_labels = np.array(["M" if m else "F" for m in male])
    unique_gender = []
    gender_id = np.empty(len(gender_labels), dtype=int)
    for i, g in enumerate(gender_labels):
        if g not in unique_gender:
            unique_gender.append(g)
        gender_id[i] = unique_gender.index(g)
    GENDER = unique_gender

    # Education level mapping
    EDUCATION_ORDERED_LIST = [
        "Elementary School",
        "Middle School",
        "Some High School",
        "High School Graduate",
        "Some College",
        "Bachelor's Degree",
        "Master's Degree",
        "Graduate Degree",
    ]
    EDUCUATION_MAP = {e: ii + 1 for ii, e in enumerate(EDUCATION_ORDERED_LIST)}
    EDUCUATION_MAP_R = {v: k for k, v in EDUCUATION_MAP.items()}

    # Create education_level column
    education_level_vals = np.array([EDUCUATION_MAP.get(str(e), 0) for e in edu])
    unique_edu_ids = np.sort(np.unique(education_level_vals))
    edu_id = np.searchsorted(unique_edu_ids, education_level_vals)
    EDUCATION_LEVEL = [EDUCUATION_MAP_R.get(e, str(e)) for e in unique_edu_ids]
    N_EDUCATION_LEVELS = len(EDUCATION_LEVEL)

    # Standardize age
    AGE = (age_raw - np.mean(age_raw)) / np.std(age_raw)

    DIRICHLET_PRIOR_WEIGHT = 2.0
    CUTPOINTS = np.arange(1, N_RESPONSE_CLASSES).astype(int)

    coords = {
        "GENDER": GENDER,
        "EDUCATION_LEVEL": EDUCATION_LEVEL,
        "CUTPOINTS": CUTPOINTS,
    }

    with pm.Model(coords=coords) as model:
        action_ = pm.Data("action", action)
        intent_ = pm.Data("intent", intention)
        contact_ = pm.Data("contact", contact)
        gender_ = pm.Data("gender", gender_id)
        age_ = pm.Data("age", AGE)
        edu_ = pm.Data("education_level", edu_id)

        # Priors (all gender-level)
        beta_action = pm.Normal("beta_action", 0, 0.5, dims="GENDER")
        beta_intent = pm.Normal("beta_intent", 0, 0.5, dims="GENDER")
        beta_contact = pm.Normal("beta_contact", 0, 0.5, dims="GENDER")
        beta_education = pm.Normal("beta_education", 0, 0.5, dims="GENDER")
        beta_age = pm.Normal("beta_age", 0, 0.5, dims="GENDER")

        # Education deltas
        delta_dirichlet = pm.Dirichlet(
            "delta", np.ones(N_EDUCATION_LEVELS - 1) * DIRICHLET_PRIOR_WEIGHT
        )

        # Insert delta_0 = 0.0
        delta_0 = [0.0]
        delta_education = pm.Deterministic(
            "delta_education",
            pm.math.concatenate([delta_0, delta_dirichlet]),
            dims="EDUCATION_LEVEL",
        )

        # Cumulative delta
        cumulative_delta_education = delta_education.cumsum()

        # Response cut points
        cutpoints = pm.Normal(
            "alpha",
            mu=0,
            sigma=1,
            transform=pm.distributions.transforms.ordered,
            shape=N_RESPONSE_CLASSES - 1,
            initval=np.arange(N_RESPONSE_CLASSES - 1) - 2.5,
            dims="CUTPOINTS",
        )

        # Likelihood
        phi = (
            beta_education[gender_] * cumulative_delta_education[edu_]
            + beta_age[gender_] * age_
            + beta_action[gender_] * action_
            + beta_intent[gender_] * intent_
            + beta_contact[gender_] * contact_
        )

        pm.OrderedLogistic(
            "response", cutpoints=cutpoints, eta=phi, observed=response_id
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
