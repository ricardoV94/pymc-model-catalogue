# pymc-model-catalogue

ASV benchmark suite for [PyMC](https://github.com/pymc-devs/pymc) and
[PyTensor](https://github.com/pymc-devs/pytensor), built around a curated
set of ~200 models extracted from pymc-examples. Each model is a standalone
`build_model() -> (Model, initial_point)` function; the benchmark wraps
them in an asv `ModelBench` class and captures four metrics per model.

**Dashboard:** https://ricardov94.github.io/pymc-model-catalogue/dashboard.html
**Experiments:** https://ricardov94.github.io/pymc-model-catalogue/experiments.html

## Metrics

- **`rewrite_time`** — wall-clock from calling `logp_dlogp_function` (or
  `compile_logp` for discrete models) to the moment it returns. Covers
  graph construction and the full rewriter pipeline.
- **`compile_time`** — wall-clock from the end of the rewrite phase to
  the end of the first `f(x)` call. Captures the NUMBA JIT cost, which
  is triggered on first call; the one eval's worth of arithmetic inside
  is microseconds, so the metric reflects compile cost in practice.
- **`n_rewrites`** — number of `rewriting: ...` lines emitted under
  `pytensor.config.optimizer_verbose = True`. A proxy for how many
  rewrites the graph went through; shifts when rewrite rules are added
  or removed.
- **`time_eval`** — steady-state per-call time, measured by asv's native
  timing machinery.

Each metric is tracked on a curated subset of 25 models listed in
[`BENCHMARK_CORE.md`](./BENCHMARK_CORE.md), chosen to give broad coverage
(hierarchical, GP, scan, linear algebra, survival, ODE, mixtures,
discrete).

## Branches

- **`main`** — source of truth: asv infra (`asv_bench/`), the model set
  (`models/`, `models_discrete/`), CI workflows, and the experiment
  scaffolding (`experiments/build.py`). Nothing here is benchmark-result
  state. `versions/tracked_versions.json` ships empty; the ledger is
  advanced on `timeline`.
- **`timeline`** — append-only. Each commit adds one pymc X.Y.0 release
  to the ledger and becomes one data point on the dashboard. Created
  and advanced by `backfill.yml` / `detect-releases.yml`; do not commit
  here by hand.
- **`experiments`** — holds `experiments/*.yaml` specs and their
  `experiments/patches/` files. Rebased onto `main` whenever the schema
  or infrastructure moves. Experiment results live under
  `gh-pages/experiments/<name>/<short>/`, keyed by the last git SHA that
  touched the YAML file.
- **`gh-pages`** — the published dashboard. Written only by CI.

## Running locally

```bash
# Smoke-test a single model directly (no asv, no timeline):
python models/stochastic_volatility.py

# Run asv against the current .venv for a curated subset:
asv run --launch-method=spawn --environment=existing:$(which python) \
  --set-commit-hash $(git rev-parse HEAD) \
  --bench "$(python scripts/core_models.py)"
```

## Workflows

| Workflow | Trigger | What it does |
|---|---|---|
| `backfill.yml` | manual dispatch | Walks the timeline a `step` at a time. First dispatch creates the branch; `reset=true` wipes it back to main. |
| `benchmark.yml` | push on `timeline`, weekly cron | Walks any new commits with `asv run --skip-existing-commits` and publishes after each — partial progress is durable across a single failing release. |
| `detect-releases.yml` | daily cron | Polls PyPI for new pymc minors, opens a PR against `timeline`. Merging the PR kicks off `benchmark.yml`. |
| `experiment.yml` | push on `experiments`, manual dispatch | Runs each changed YAML, keyed by the YAML file's last-touching git SHA for dedup. |

The timeline CI uses a custom install hook (`asv_bench/_provision.py`)
that installs `pymc==<pinned>` with `uv pip install --exclude-newer
<released_at + 1 day>`, so pytensor and every transitive dep resolve to
what was latest on each pymc release day — historical data points stay
reproducible.

## Adding an experiment

1. Check out the `experiments` branch (rebase on main first if stale).
2. Copy `base.yaml` to `<your_experiment>.yaml` and add `revert_commits`
   or `patches` on either the `pytensor` or `pymc` block.
3. Patches are `git format-patch`-style files — generate them from a
   manually-resolved clone and commit them under
   `experiments/patches/<your_experiment>/`.
4. Push the branch; `experiment.yml` picks it up via the path filter.

Experiments layer modifications on top of the `pytensor.ref` / `pymc.ref`
pins. `build.py` clones both packages locally and installs them editable
so pymc and pytensor can be patched symmetrically.

## Requirements

Local development uses current pymc/pytensor branches:

```bash
uv pip install "pymc @ git+https://github.com/pymc-devs/pymc.git@main" \
               "pytensor @ git+https://github.com/pymc-devs/pytensor.git@main" \
               numba asv
```
