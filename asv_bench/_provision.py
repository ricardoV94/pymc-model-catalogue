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


# Pre-patch block as it appears in pytensor on commits that have the numba
# disk-cache infrastructure but predate pymc-devs/pytensor#1992. Matched as
# exact literal text so any whitespace drift fails loudly rather than silently
# leaving an unpatched pytensor in the env.
_PR1992_OLD = """        op_name = jitable_func.__name__
        cached_func = compile_numba_function_src(
            src=f"def {op_name}(*args): return jitable_func(*args)",
            function_name=op_name,
            global_env=globals() | {"jitable_func": jitable_func},
            cache_key=f"{cache_key}_fastmath{int(config.numba__fastmath)}",
        )
"""

_PR1992_NEW = """        full_cache_key = f"{cache_key}_fastmath{int(config.numba__fastmath)}"
        safe_key = re.sub(r"[^a-zA-Z0-9_]", "_", full_cache_key)
        op_name = f"{jitable_func.__name__}_{safe_key}"
        cached_func = compile_numba_function_src(
            src=f"def {op_name}(*args): return jitable_func(*args)",
            function_name=op_name,
            global_env=globals() | {"jitable_func": jitable_func},
            cache_key=full_cache_key,
        )
"""

# Sentinel that only appears in the post-1992 version of the file.
_PR1992_APPLIED_MARKER = 'full_cache_key = f"{cache_key}_fastmath'


def _patch_pytensor_numba_cache(env_python: Path) -> None:
    """Backport pymc-devs/pytensor#1992 into the freshly-installed pytensor.

    Without this, historical pytensor versions that already ship the numba
    disk-cache infrastructure but predate #1992 crash on LLVM symbol-mangling
    collisions when asv's two-phase Build→Eval benchmark hits the numba cache
    across process boundaries. The patch is tiny (one function in one file)
    and idempotent — we detect already-patched / pre-cache-infra / patchable
    cases and only rewrite in the last one.
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
    basic_py = pytensor_dir / "link" / "numba" / "dispatch" / "basic.py"

    if not basic_py.exists():
        print(
            f"[provision] pytensor has no {basic_py.relative_to(pytensor_dir)} "
            "— skipping PR 1992 backport",
            file=sys.stderr,
        )
        return

    src = basic_py.read_text()

    if _PR1992_APPLIED_MARKER in src:
        print("[provision] pytensor PR 1992 already applied — skipping", file=sys.stderr)
        return

    if _PR1992_OLD not in src:
        print(
            "[provision] pytensor basic.py does not match PR 1992's pre-patch "
            "shape — skipping (likely pre-cache-infra)",
            file=sys.stderr,
        )
        return

    new_src = src.replace(_PR1992_OLD, _PR1992_NEW, 1)
    if new_src.count(_PR1992_NEW) != 1:
        raise RuntimeError(
            f"PR 1992 backport replacement in {basic_py} did not produce "
            "exactly one match; aborting to avoid running an unpatched pytensor"
        )

    if "\nimport re\n" not in new_src and not new_src.startswith("import re\n"):
        new_src = new_src.replace("import warnings\n", "import re\nimport warnings\n", 1)

    basic_py.write_text(new_src)
    print(f"[provision] applied pytensor PR 1992 backport to {basic_py}", file=sys.stderr)


if __name__ == "__main__":
    main()
