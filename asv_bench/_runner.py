"""Shared build+measure helper for ASV benchmarks.

Extracted from ``models/_benchmark.py`` and split into phases so each metric
(rewrite, compile, eval) is captured separately. Returns a dict that also
exposes a zero-arg ``call`` closure for the eval loop.

Metric definitions (intentional, don't "correct" these):

- ``rewrite_time`` — wall-clock from calling ``logp_dlogp_function`` /
  ``compile_logp`` to the moment it returns. Covers graph construction
  and the full rewriter pipeline. Pytensor's NUMBA backend does not JIT
  in this phase; code generation is deferred to the first call.
- ``compile_time`` — wall-clock from the *end* of the rewrite phase to
  the *end* of the first ``f(x)`` call. This deliberately includes the
  first eval's arithmetic because that is when numba JIT compilation
  happens: for pytensor's NUMBA linker the JIT cost is triggered on
  first call, and we want that cost captured. For practical models the
  JIT dominates (seconds) and the one-shot arithmetic is microseconds,
  so calling this ``compile_time`` reflects what it actually measures
  in practice. Renaming to ``first_eval_time`` would be pedantically
  more accurate but obscures the intent.
- ``eval_time`` — steady-state per-call timing from asv's native
  timing machinery (implemented in ``bench_models.py``).
- ``n_rewrites`` — count of ``rewriting: ...`` lines printed by
  pytensor under ``config.optimizer_verbose=True``.
"""

import contextlib
import importlib
import io
from time import perf_counter

import numpy as np
from pymc.blocking import DictToArrayBijection
from pytensor import config

_REWRITE_PREFIX = "rewriting: "


def build_and_measure(model_path: str, *, mode: str = "NUMBA") -> dict:
    discrete = model_path.startswith("models_discrete.")
    module = importlib.import_module(model_path)
    model, ip = module.build_model()

    buf = io.StringIO()
    t0 = perf_counter()
    with config.change_flags(optimizer_verbose=True), contextlib.redirect_stdout(buf):
        if discrete:
            logp_fn = model.compile_logp(mode=mode)
            pt_fn = logp_fn.f
        else:
            logp_dlogp_fn = model.logp_dlogp_function(ravel_inputs=True, mode=mode)
            pt_fn = logp_dlogp_fn._pytensor_function
    rewrite_time = perf_counter() - t0
    n_rewrites = sum(
        1 for line in buf.getvalue().splitlines() if line.startswith(_REWRITE_PREFIX)
    )

    pt_fn.trust_input = True

    if discrete:
        args = [np.asarray(ip[v.name], dtype=v.dtype) for v in model.value_vars]
        call = lambda: pt_fn(*args)
    else:
        raveled = DictToArrayBijection.map(
            {v.name: ip[v.name] for v in model.value_vars}
        )
        x = raveled.data
        call = lambda: pt_fn(x)

    # "compile_time" here intentionally covers the first f(x) call — see
    # the module docstring. NUMBA JIT compilation happens on first call,
    # so this is the metric we care about even though one eval's worth of
    # arithmetic is included.
    t0 = perf_counter()
    out = call()
    compile_time = perf_counter() - t0

    if discrete:
        assert np.isfinite(out), f"logp is not finite: {out}"
    else:
        logp, dlogp = out
        assert np.isfinite(logp), f"logp is not finite: {logp}"
        assert np.all(np.isfinite(dlogp)), "dlogp has non-finite values"

    return {
        "rewrite_time": rewrite_time,
        "compile_time": compile_time,
        "n_rewrites": n_rewrites,
        "call": call,
    }
