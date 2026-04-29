"""Custom ASV env provisioner.

ASV invokes this once per (commit, environment) to provision the env that
will run the benchmark suite. Instead of pip-installing this project (it's
not a package), we read the pinned pymc minor version and its release date
from ``versions/tracked_versions.json`` at the build_dir (the checked-out
copy of the repo at the commit under test) and install that exact release
with ``uv pip install --exclude-newer <released_at>``. The date constraint
forces pytensor and every other transitive dep to resolve to the version
that would have been picked at pymc's release time, making each historical
data point reproducible — without this, pymc's open-ended pytensor pin
(e.g. ``pytensor<2.21``) would resolve to whatever is latest today.

uv is required because pip has no equivalent of ``--exclude-newer``.

Invocation (from asv.conf.json install_command)::

    python {build_dir}/asv_bench/_provision.py {env_dir} {build_dir}
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

from packaging.version import Version


def main() -> None:
    env_dir = Path(sys.argv[1])
    build_dir = Path(sys.argv[2])

    ledger_path = build_dir / "versions" / "tracked_versions.json"
    ledger = json.loads(ledger_path.read_text())
    releases = ledger["minor_releases"]
    if not releases:
        raise RuntimeError(f"{ledger_path} has no minor_releases entries")
    # Pick the highest-versioned entry rather than trust list order — the
    # ledger should always be ordered but a bad hand-edit or re-upload
    # mustn't silently install the wrong release.
    pinned = max(releases, key=lambda e: Version(e["pymc"]))
    pymc_version = pinned["pymc"]
    released_at = pinned.get("released_at")
    if not released_at:
        raise RuntimeError(
            f"ledger entry for pymc {pymc_version} is missing 'released_at'; "
            "rerun scripts/bump_release.py to refresh dates"
        )
    # --exclude-newer with a date-only value has ambiguous semantics (uv's
    # docs don't say whether YYYY-MM-DD is start- or end-of-day). Add one
    # day to the release date so a release that went out at, say, 14:00 UTC
    # on its own ship date is still within the cutoff. Pymc minors are
    # weeks apart so the extra day cannot pick up a later release.
    cutoff = (date.fromisoformat(released_at) + timedelta(days=1)).isoformat()

    print(
        f"[provision] pinning pymc=={pymc_version}, "
        f"deps resolved as of {cutoff} (released_at + 1 day)",
        file=sys.stderr,
    )

    env_python = env_dir / "bin" / "python"
    subprocess.run(
        [
            "uv", "pip", "install",
            "--python", str(env_python),
            "--exclude-newer", cutoff,
            f"pymc=={pymc_version}",
            "numba",
            "asv",
        ],
        check=True,
    )

    _patch_pytensor_numba_cache(env_python)


# Backport of pymc-devs/pytensor#2079, appended to the installed pytensor's
# link/numba/cache.py. Replaces numba's sequential FunctionIdentity._unique_ids
# counter with a UUID iterator so sibling forks fresh-compiling same-qualname
# functions can't allocate colliding LLVM symbols. Without it, older pytensors
# that ship the numba disk-cache infrastructure but predate #2079 segfault
# under asv's two-phase Build->Eval cycle when the cache is hit across process
# boundaries.
_PR2079_MARKER = "FunctionIdentity._unique_ids = _RandomUidIter()"
_PR2079_PATCH = '''

# pymc-devs/pytensor#2079 backport
import uuid
from numba.core.bytecode import FunctionIdentity


class _RandomUidIter:
    def __iter__(self):
        return self

    def __next__(self):
        return uuid.uuid4().int


FunctionIdentity._unique_ids = _RandomUidIter()
'''


def _patch_pytensor_numba_cache(env_python: Path) -> None:
    """Backport pymc-devs/pytensor#2079 into the freshly-installed pytensor.

    Idempotent: skipped if cache.py is missing (pre-cache-infra, the bug
    can't manifest) or if the patch is already in place (post-#2079).
    """
    result = subprocess.run(
        [
            str(env_python),
            "-c",
            "import pytensor, pathlib; print(pathlib.Path(pytensor.__file__).parent)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    pytensor_dir = Path(result.stdout.strip())
    cache_py = pytensor_dir / "link" / "numba" / "cache.py"

    if not cache_py.exists():
        print(
            f"[provision] pytensor has no {cache_py.relative_to(pytensor_dir)} "
            "— skipping PR 2079 backport (pre-cache-infra)",
            file=sys.stderr,
        )
        return

    src = cache_py.read_text()
    if _PR2079_MARKER in src:
        print("[provision] pytensor PR 2079 already applied — skipping", file=sys.stderr)
        return

    cache_py.write_text(src + _PR2079_PATCH)
    print(f"[provision] applied pytensor PR 2079 backport to {cache_py}", file=sys.stderr)


if __name__ == "__main__":
    main()
