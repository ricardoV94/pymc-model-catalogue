# Progress Tracker

Tracks extraction progress across PLAN.md's notebook inventory. One notebook can
produce multiple model files, so the primary metric is **model files**, not notebooks.

**Do not re-attempt any notebook listed under "Skipped"** — reasons are listed per
item and won't change unless a dependency or upstream bug is fixed.

## Summary

| Metric | Count |
|---|---|
| Model files done (continuous) | 0 |
| Model files done (discrete) | 0 |
| **Total model files done** | **0** |
| Skipped model files (`.skip`) | 0 |
| Notebooks fully addressed | 0 / 139 |
| Notebooks remaining | 139 |
| Estimated final total | ~155–170 model files |

## Per-category breakdown

| Category | Notebooks | Notebooks done | Model files | Skipped files | Notebooks remaining |
|---|---|---|---|---|---|
| BART | 4 | 0 | 0 | 0 | 4 |
| Case Studies | 16 | 0 | 0 | 0 | 16 |
| Causal Inference | 11 | 0 | 0 | 0 | 11 |
| Diagnostics | 4 | 0 | 0 | 0 | 4 |
| Fundamentals | 1 | 0 | 0 | 0 | 1 |
| Gaussian Processes | 17 | 0 | 0 | 0 | 17 |
| GLMs | 15 | 0 | 0 | 0 | 15 |
| How-To | 12 | 0 | 0 | 0 | 12 |
| Introductory | 1 | 0 | 0 | 0 | 1 |
| Mixture Models | 5 | 0 | 0 | 0 | 5 |
| ODE Models | 4 | 0 | 0 | 0 | 4 |
| Samplers | 8 | 0 | 0 | 0 | 8 |
| Spatial | 3 | 0 | 0 | 0 | 3 |
| Statistical Rethinking | 19 | 0 | 0 | 0 | 19 |
| Survival Analysis | 5 | 0 | 0 | 0 | 5 |
| Time Series | 9 | 0 | 0 | 0 | 9 |
| Variational Inference | 5 | 0 | 0 | 0 | 5 |
| **Total** | **139** | **0** | **0** | **0** | **139** |

## Skipped — permanent

_To be filled in as notebooks are evaluated. Group entries under one of three buckets:_

### Missing PyMC model (no inference graph)

### Missing dependencies

### PyMC / PyTensor bugs

## Remaining

_Active working list of notebooks still to extract, grouped by category._
