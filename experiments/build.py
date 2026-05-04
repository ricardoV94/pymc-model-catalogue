"""Build a venv for a benchmark experiment described by a YAML file.

An experiment pins a pytensor ref and a pymc ref, then optionally layers
modifications on top of either source. Modifications are symmetric
between the two packages:

- ``revert_commits`` — SHAs to ``git revert``. Must apply cleanly.
- ``patches`` — ordered list of ``git format-patch``-style .patch files
  (paths relative to the repo root). Applied with ``git am``. Use this
  when a modification needs manual conflict resolution — generate the
  patches once from a resolved clone and commit them into the repo so
  CI can reproduce the state.

Both clones are installed editable into the venv; the venv's python path
is printed so callers can feed it to ``asv run``.

YAML schema::

    name: <str>
    description: <str>
    pytensor:
      repo: pymc-devs/pytensor           # optional, default
      ref: main                          # any git ref
      revert_commits: [<sha>, ...]       # optional
      patches: [<path>, ...]             # optional
    pymc:
      repo: pymc-devs/pymc               # optional, default
      ref: v6                            # any git ref
      revert_commits: [<sha>, ...]       # optional
      patches: [<path>, ...]             # optional
    models: [<module_path>, ...]         # optional; override default CORE
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDS_DIR = REPO_ROOT / "experiments" / ".builds"

DEFAULT_REPOS = {
    "pytensor": "pymc-devs/pytensor",
    "pymc": "pymc-devs/pymc",
}


def run(cmd: list[str | Path], cwd: Path | None = None) -> None:
    printable = " ".join(str(c) for c in cmd)
    print(f"  $ {printable}", file=sys.stderr)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def clone_at(repo: str, ref: str, dest: Path) -> None:
    url = f"https://github.com/{repo}.git"
    if not dest.exists():
        run(["git", "clone", "--quiet", url, str(dest)])
    else:
        # Ensure origin points at the right repo (may differ between experiments)
        cur = subprocess.check_output(
            ["git", "remote", "get-url", "origin"], cwd=dest, text=True
        ).strip()
        if cur != url:
            run(["git", "remote", "set-url", "origin", url], cwd=dest)
    for subcmd in (["revert", "--abort"], ["merge", "--abort"],
                   ["cherry-pick", "--abort"], ["am", "--abort"]):
        subprocess.run(["git", *subcmd], cwd=dest, check=False, capture_output=True)
    run(["git", "reset", "--quiet", "--hard"], cwd=dest)
    run(["git", "fetch", "--quiet", "origin", ref], cwd=dest)
    run(["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"], cwd=dest)
    run(["git", "clean", "-qfdx"], cwd=dest)


def apply_reverts(clone_dir: Path, shas: list[str]) -> None:
    # Newest-first so each revert applies against a tree closest to where
    # the commit was authored, minimising conflicts with subsequent changes.
    timestamped = []
    for sha in shas:
        ts = int(
            subprocess.check_output(
                ["git", "log", "-1", "--format=%at", sha], cwd=clone_dir, text=True
            )
        )
        timestamped.append((ts, sha))
    timestamped.sort(reverse=True)
    for _, sha in timestamped:
        print(f"  reverting {sha[:10]}", file=sys.stderr)
        parent_count = (
            subprocess.check_output(
                ["git", "cat-file", "-p", sha], cwd=clone_dir, text=True
            ).count("\nparent ")
        )
        cmd = ["git", "revert", "--no-edit"]
        if parent_count > 1:
            cmd += ["-m", "1"]
        cmd.append(sha)
        run(cmd, cwd=clone_dir)


def apply_patches(clone_dir: Path, patch_files: list[Path]) -> None:
    for p in patch_files:
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        print(f"  applying {p.relative_to(REPO_ROOT)}", file=sys.stderr)
        run(
            [
                "git",
                "-c", "user.email=ci@example.com",
                "-c", "user.name=ci",
                "am", "--quiet", str(p),
            ],
            cwd=clone_dir,
        )


def prepare_clone(pkg: str, cfg: dict, clone_dir: Path) -> None:
    repo = cfg.get("repo", DEFAULT_REPOS[pkg])
    ref = cfg["ref"]
    reverts = cfg.get("revert_commits") or []
    patches = [Path(p) for p in (cfg.get("patches") or [])]
    print(f"[prepare:{pkg}] {repo}@{ref}", file=sys.stderr)
    clone_at(repo, ref, clone_dir)
    if reverts:
        print(f"[prepare:{pkg}] reverting {len(reverts)} commit(s)", file=sys.stderr)
        apply_reverts(clone_dir, reverts)
    if patches:
        print(f"[prepare:{pkg}] applying {len(patches)} patch(es)", file=sys.stderr)
        apply_patches(clone_dir, patches)
    if pkg == "pytensor":
        patch_pytensor_2079(clone_dir)
        patch_pytensor_mpm_cheap(clone_dir)


# Backport of pymc-devs/pytensor#2079, appended to pytensor's
# link/numba/cache.py. Replaces numba's sequential FunctionIdentity._unique_ids
# counter with a UUID iterator so sibling forks fresh-compiling same-qualname
# functions can't allocate colliding LLVM symbols. Without it, pytensor refs
# that ship the numba disk-cache infrastructure but predate #2079 segfault
# under asv's forkserver when sibling benchmarks hit the cache.
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


def patch_pytensor_2079(clone_dir: Path) -> None:
    """Idempotent: skipped if cache.py is missing (pre-cache-infra) or the
    patch is already in place (post-#2079)."""
    cache_py = clone_dir / "pytensor" / "link" / "numba" / "cache.py"
    if not cache_py.exists():
        print(
            f"[prepare:pytensor] no {cache_py.relative_to(clone_dir)} "
            "— skipping PR 2079 backport (pre-cache-infra)",
            file=sys.stderr,
        )
        return
    src = cache_py.read_text()
    if _PR2079_MARKER in src:
        print("[prepare:pytensor] PR 2079 already present — skipping", file=sys.stderr)
        return
    cache_py.write_text(src + _PR2079_PATCH)
    print(f"[prepare:pytensor] applied PR 2079 backport to {cache_py}", file=sys.stderr)


_MPM_CHEAP_MARKER = "# _mpm_cheap no-op override"
_MPM_CHEAP_NOOP = '''

# _mpm_cheap no-op override
import contextlib as _contextlib

@_contextlib.contextmanager
def use_optimized_cheap_pass():
    yield
'''


def patch_pytensor_mpm_cheap(clone_dir: Path) -> None:
    """Idempotent: skipped if basic.py is missing or _mpm_cheap isn't referenced."""
    basic_py = clone_dir / "pytensor" / "link" / "numba" / "dispatch" / "basic.py"
    if not basic_py.exists():
        return
    src = basic_py.read_text()
    if _MPM_CHEAP_MARKER in src:
        print("[prepare:pytensor] _mpm_cheap no-op already present — skipping", file=sys.stderr)
        return
    if "_mpm_cheap" not in src:
        return
    basic_py.write_text(src + _MPM_CHEAP_NOOP)
    print(f"[prepare:pytensor] applied _mpm_cheap no-op to {basic_py}", file=sys.stderr)


def create_venv(venv_dir: Path, pymc_dir: Path, pytensor_dir: Path) -> None:
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    run(["uv", "venv", "--quiet", str(venv_dir)])
    py = venv_dir / "bin" / "python"
    # Install pymc editable — its deps pull a wheel pytensor.
    run([
        "uv", "pip", "install", "--quiet", "--python", str(py),
        "-e", str(pymc_dir), "numba", "asv",
    ])
    # Replace the wheel pytensor with our editable clone.
    run([
        "uv", "pip", "install", "--quiet", "--python", str(py),
        "--reinstall-package", "pytensor",
        "-e", str(pytensor_dir),
    ])


def build(cfg: dict) -> Path:
    name = cfg["name"]
    build_dir = BUILDS_DIR / name
    pytensor_dir = build_dir / "pytensor"
    pymc_dir = build_dir / "pymc"
    venv_dir = build_dir / "venv"
    build_dir.mkdir(parents=True, exist_ok=True)

    prepare_clone("pytensor", cfg["pytensor"], pytensor_dir)
    prepare_clone("pymc", cfg["pymc"], pymc_dir)

    print(f"[build:{name}] creating venv", file=sys.stderr)
    create_venv(venv_dir, pymc_dir, pytensor_dir)
    print(f"[build:{name}] done", file=sys.stderr)
    return venv_dir / "bin" / "python"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml", type=Path, help="path to experiment YAML file")
    args = ap.parse_args()
    with args.yaml.open() as f:
        cfg = yaml.safe_load(f)
    py = build(cfg)
    print(py)


if __name__ == "__main__":
    main()
