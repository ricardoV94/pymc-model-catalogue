# PyMC Model Catalogue

Standalone PyMC models extracted from [pymc-examples](https://github.com/pymc-devs/pymc-examples) notebooks.
Used to benchmark PyTensor's NUMBA linker compilation and evaluation of `logp`/`dlogp` functions,
both before and after `freeze_dims_and_data`.

## Structure

```
models/              # Continuous models (logp + dlogp)
models/data/         # .npz data files for models with large datasets
models_discrete/     # Models with discrete free variables (logp only)
run_all.py           # Parallel runner (1 CPU per model, 600s timeout)
```

## Running

Single model:

```bash
python models/stochastic_volatility.py
```

All models in parallel:

```bash
python run_all.py
```

Each model prints logp, gradient norm (continuous only), and per-call timing
for both the original and frozen (after `freeze_dims_and_data`) variants.

## Requirements

Development versions of PyMC and PyTensor with NUMBA support:

```bash
pip install pymc pytensor numba
```
