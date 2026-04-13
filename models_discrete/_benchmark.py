"""Thin shim so ``python models_discrete/<foo>.py`` can use the shared helper.

The real implementation lives in ``models/_benchmark.py``. This file
only exists because example scripts in ``models_discrete/`` import
``_benchmark`` as a sibling module, so a same-directory entry point is
needed; we just re-export the shared functions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models._benchmark import build_logp_fn, run_benchmark  # noqa: E402,F401
