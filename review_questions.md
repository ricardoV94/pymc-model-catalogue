# Review Questions

## freeze_dims_and_data fails on models with initval
- `gaussian_mixture_model.py` and `marginalized_gaussian_mixture_model.py` use `initval=` on ordered transforms
- `freeze_dims_and_data` -> `fgraph_from_model` raises `NotImplementedError: Cannot convert models with non-default initial_values`
- **Question**: Should we skip the frozen benchmark for these models, or is this a bug that should be fixed in pymc?

## multilevel_modeling.ipynb - skipped models
- GLM agent extracted 8 models but skipped the centered varying_intercept_slope and centered covariation because they produce divergences
- Also skipped the `new_county_house` prediction-only model
- **Question**: These are valid models users would write (the divergences are expected). Should we include them?

## Skipped models — missing dependencies
- `GEV_port_pirie.py` and `ssm_hurricane_newtonian.py` require `pymc_extras` (not installed)
- `hsgp_basic_1d.py`, `hsgp_basic_2d_linearized.py`, `hsgp_hierarchical.py`, `hsgp_kronecker.py` require `preliz`
- `ab_testing_bernoulli.py` — Beta(100, 100) produces `inf` logp (PyMC numerical bug)
- `gp_tprocess_poisson.py` — internal PyMC error in MvStudentT rv_op
- `sr19_child_strategy.py` — broadcast shape mismatch in CustomDist
- **Question**: Should we install `pymc_extras` and `preliz` to enable these? Or keep them as .skip files?

## Skipped notebooks — no extractable model
- `GP-Circular.ipynb`, `GP-MaunaLoa2.ipynb`, `GP-SparseApprox.ipynb` — PyMC3/theano-only, not ported to v5
- `GP-MeansAndCovs.ipynb` — reference tutorial, no complete inference model
- `GLM-ordinal-features.ipynb` — requires R data file not available locally
- `bayesian_nonparametric_causal.ipynb` — all models use `pymc_bart`
- `interventional_distribution.ipynb` — generative-only (no observed data)
- StatRethinking 02, 06, 20 — no `pm.Model` or discussion-only

## bayesian_workflow_logistic.py uses synthetic data
- Original loads COVID-19 data dynamically from GitHub via polars
- We used synthetic data approximation since the model structure is what matters
- **Question**: Should we try to download and save the real data instead?

## Spatial batch notes
- `car_scotland_lipcancer.py` — extracted only the final (random-alpha CAR) model per PLAN. The two intermediate models (independent-only, fixed-alpha CAR) were skipped as building blocks but are valid complete models with different logp shapes. **Question**: include them as separate benchmarks?
- `malaria_hsgp.py` — elevation/lonlat data was reconstructed via `rasterio`/`pyproj`/`cKDTree` in an external one-shot (geopandas not installed in venv). Arrays look plausible but not numerically cross-checked against a notebook run. **Question**: is proxy reproduction acceptable or should we match a notebook run exactly?
- `nyc_bym_traffic.py` — the BYM2 scaling factor (sparse pseudo-inverse of the 1921x1921 Laplacian) is precomputed offline and stored in the .npz rather than inlined into `build_model()`. It depends only on the adjacency matrix and isn't part of the model graph. **Question**: does this violate the "inline helpers" rule, or is treating it as derived data acceptable?

## How-To batch notes
- `lkj_cholesky_cov_mvnormal.py` — notebook has no `:::{post}` directive, so no `:author:` was available (marked "Unknown"). Data is generated inline via seeded RNG (N=10000). **Question**: fall back to top git contributors for Authors? Snapshot data to .npz for RNG stability?
- `missing_data_chained_normal.py` — the original notebook appears to swap `lmx_mean/lmx_sd` and `climate_mean/climate_sd` when building the prior dict (preserved verbatim). **Question**: notebook bug or intentional?
- `updating_priors_linear.py` — only the initial base model was extracted. The `pm.Interpolated` variant (the actual point of the notebook) was skipped per PLAN's concerns about numba compilation. **Question**: attempt to benchmark an Interpolated variant with a precomputed x/pdf grid anyway?

## Large models may be slow to benchmark
- `probabilistic_matrix_factorization.py` — 26,250 free RVs (943 users x 1682 movies x 10 latent factors)
- `item_response_nba.py` — 1,559 latent params
- `GLM_discrete_choice_mixed_logit.py` — 544 individual taste parameters
- `GLM_rolling_regression.py` — 2,272 random walk parameters
- These all pass but may dominate benchmark time
