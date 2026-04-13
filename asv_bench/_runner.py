"""ASV-side wrapper around ``models._benchmark.build_logp_fn``.

Splits the shared ``build_logp_fn`` call into measured phases so asv can
capture each as a metric. ``build_logp_fn`` itself is intentionally
API-version-agnostic (no ``logp_dlogp_function``, no ``ravel_inputs``),
so the historical timeline can reach pymc releases that predate those.

Metric definitions (intentional, don't "correct" these):

- ``rewrite_time`` — wall-clock from calling ``build_logp_fn`` (which
  internally runs ``rewrite_pregrad``, takes gradients, joins inputs,
  and calls ``pymc.compile``) to the moment it returns. Covers graph
  construction and the full rewriter pipeline. Pytensor's NUMBA
  backend does not JIT in this phase; code generation is deferred to
  the first call.
- ``compile_time`` — wall-clock from the end of ``rewrite_time`` to the
  end of the first ``f(x)`` call. This deliberately includes one eval's
  arithmetic because that is when numba JIT compilation happens. For
  practical models the JIT dominates (seconds) and the arithmetic is
  microseconds, so calling this ``compile_time`` reflects what it
  actually measures.
- ``eval_time`` — steady-state per-call timing via asv's native timing
  machinery, implemented in ``bench_models.py``.
- ``n_rewrites`` — count of ``rewriting: ...`` lines printed by pytensor
  under ``config.optimizer_verbose = True``.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from pytensor import config

# The shared build_logp_fn lives in models/_benchmark.py — add the repo
# root to sys.path so the import works both when asv runs this module
# directly and when test scripts run it cwd-relative.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models._benchmark import build_logp_fn  # noqa: E402

_REWRITE_PREFIX = "rewriting: "


def build_and_measure(model_path: str, *, mode: str = "NUMBA") -> dict:
    discrete = model_path.startswith("models_discrete.")
    module = importlib.import_module(model_path)
    model, ip = module.build_model()

    buf = io.StringIO()
    t0 = perf_counter()
    with config.change_flags(optimizer_verbose=True), contextlib.redirect_stdout(buf):
        fn, x = build_logp_fn(model, ip, mode=mode, with_grad=not discrete)
    rewrite_time = perf_counter() - t0
    n_rewrites = sum(
        1 for line in buf.getvalue().splitlines() if line.startswith(_REWRITE_PREFIX)
    )

    # "compile_time" here intentionally covers the first call — see the
    # module docstring. NUMBA JIT happens on first call.
    t0 = perf_counter()
    out = fn(x)
    compile_time = perf_counter() - t0

    if discrete:
        (logp,) = out
        assert np.isfinite(logp), f"logp is not finite: {logp}"
    else:
        logp, dlogp = out
        assert np.isfinite(logp), f"logp is not finite: {logp}"
        assert np.all(np.isfinite(dlogp)), "dlogp has non-finite values"

    return {
        "rewrite_time": rewrite_time,
        "compile_time": compile_time,
        "n_rewrites": n_rewrites,
        "call": lambda: fn(x),
    }
