"""ASV benchmark module: per-model rewrite/compile/eval timings."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asv_bench._runner import build_and_measure  # noqa: E402


def _discover_models() -> list[str]:
    paths: list[str] = []
    for subdir in ("models", "models_discrete"):
        for py in sorted((REPO_ROOT / subdir).glob("*.py")):
            if py.name.startswith("_"):
                continue
            paths.append(f"{subdir}.{py.stem}")
    return paths


class ModelBench:
    params = [_discover_models()]
    param_names = ["model"]
    timeout = 1200.0

    def setup(self, model):
        self._result = build_and_measure(model)
        self._call = self._result["call"]

    def track_rewrite_time(self, model):
        return self._result["rewrite_time"]

    track_rewrite_time.unit = "seconds"

    def track_compile_time(self, model):
        return self._result["compile_time"]

    track_compile_time.unit = "seconds"

    def track_n_rewrites(self, model):
        return self._result["n_rewrites"]

    track_n_rewrites.unit = "count"

    def time_eval(self, model):
        self._call()
