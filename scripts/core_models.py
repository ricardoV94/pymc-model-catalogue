"""Canonical list of BENCHMARK_CORE model module paths + helpers.

Keep this in sync with BENCHMARK_CORE.md. CI workflows and anything else
that needs to filter asv benchmarks to the core set should import from
here rather than hardcoding a duplicate regex.
"""

from __future__ import annotations

CORE_MODELS: tuple[str, ...] = (
    "models.eight_schools_noncentered",
    "models.BEST",
    "models.GLM_poisson_regression",
    "models.GLM_negative_binomial_regression",
    "models.multilevel_varying_intercept_slope_noncentered",
    "models.GLM_hierarchical_binomial_rat_tumor",
    "models.rugby_analytics",
    "models.lkj_cholesky_cov_mvnormal",
    "models.probabilistic_matrix_factorization",
    "models.bayesian_sem_workflow",
    "models.stochastic_volatility",
    "models.ar2",
    "models.time_series_generative_graph_ar2",
    "models.bayesian_var_ireland",
    "models.euler_maruyama_linear_sde",
    "models.gp_marginal_matern52",
    "models.gp_births_hsgp",
    "models.malaria_hsgp",
    "models.marginalized_gaussian_mixture_model",
    "models.dirichlet_mixture_of_multinomials",
    "models.frailty_coxph",
    "models.ode_lotka_volterra_pytensor_scan",
    "models.nyc_bym_traffic",
    "models.excess_deaths",
    "models_discrete.occupancy_crossbill",
)


def asv_bench_regex(models: tuple[str, ...] | list[str] = CORE_MODELS) -> str:
    """Return an asv --bench regex matching the given model set.

    The regex is intentionally loose — it matches anywhere the fully-qualified
    model module path appears inside an asv benchmark id. Tight matching on
    asv's ``('models.foo')`` param stringification is fragile: if asv ever
    changes quoting or param formatting the filter would silently select zero
    benchmarks. The model paths are distinctive enough that loose matching
    against the ``ModelBench`` class yields no false positives.
    """
    import re
    alts = "|".join(re.escape(m) for m in models)
    return rf"ModelBench.*(?:{alts})"


def regex_for_yaml(yaml_path: str) -> str:
    """Return the asv --bench regex for an experiment YAML.

    If the YAML has a top-level ``models`` field, use that. Otherwise use
    the default BENCHMARK_CORE set.
    """
    import yaml
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    models = cfg.get("models") or CORE_MODELS
    return asv_bench_regex(tuple(models))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(regex_for_yaml(sys.argv[1]))
    else:
        print(asv_bench_regex())
