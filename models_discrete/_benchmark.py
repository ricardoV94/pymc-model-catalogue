"""Benchmark helper for PyMC model catalogue."""

import timeit

import numpy as np
from pymc.blocking import DictToArrayBijection
from pymc.model.transform.optimization import freeze_dims_and_data


def run_benchmark(build_model, *, discrete=False, mode="NUMBA"):
    model, ip = build_model()
    frozen_model, _ = build_model()
    frozen_model = freeze_dims_and_data(frozen_model)

    for label, m in [("original", model), ("frozen", frozen_model)]:
        print(f"\n=== {label} ===")

        if discrete:
            logp_fn = m.compile_logp(mode=mode)
            f = logp_fn.f
            f.trust_input = True

            args = [np.asarray(ip[v.name], dtype=v.dtype) for v in m.value_vars]
            logp = f(*args)
            assert np.isfinite(logp), f"logp is not finite: {logp}"
            print(f"logp = {logp:.4f}")

            call = lambda: f(*args)
        else:
            logp_dlogp_fn = m.logp_dlogp_function(ravel_inputs=True, mode=mode)
            f = logp_dlogp_fn._pytensor_function
            f.trust_input = True

            raveled = DictToArrayBijection.map(
                {v.name: ip[v.name] for v in m.value_vars}
            )
            x = raveled.data

            logp, dlogp = f(x)
            assert np.isfinite(logp), f"logp is not finite: {logp}"
            assert np.all(np.isfinite(dlogp)), f"dlogp has non-finite values"
            print(f"logp = {logp:.4f}, grad norm = {np.linalg.norm(dlogp):.4f}")

            call = lambda: f(x)

        single = timeit.timeit(call, number=1)
        n_evals = max(1, min(100_000, int(15.0 / max(single, 1e-8))))
        total = timeit.timeit(call, number=n_evals)
        print(f"Time per call: {total / n_evals * 1e6:.1f} us ({n_evals} evals)")
