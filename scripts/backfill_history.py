"""Append synthetic commits to the pymc-version ledger.

Idempotent by design: each invocation reads the existing ledger, fetches
PyPI, picks up where the ledger left off, and appends strictly-newer minor
releases (up to ``--limit`` new commits). Safe to run repeatedly — each
call advances the timeline by at most N commits, so a bootstrap can be
paced in small steps and failures stop at the first broken commit.

Drift guard: refuses to run if any entry already in the ledger disagrees
with PyPI's record of that version (missing, different ``released_at``).
This prevents silent corruption if someone hand-edits the ledger on the
timeline branch.

Usage::

    git checkout timeline
    python scripts/backfill_history.py                 # append everything new
    python scripts/backfill_history.py --limit 3       # append at most 3
    python scripts/backfill_history.py --from 5.13.0   # floor (first run)
    python scripts/backfill_history.py --dry-run

5.13.0 is the earliest minor release that ships ``freeze_dims_and_data`` —
models predating it will not load under the benchmark's current setup.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = REPO_ROOT / "versions" / "tracked_versions.json"
PYPI_URL = "https://pypi.org/pypi/pymc/json"


def fetch_minor_releases() -> dict[Version, str]:
    """Return {Version: released_at} for every pymc X.Y.0 on PyPI."""
    with urllib.request.urlopen(PYPI_URL) as resp:  # noqa: S310
        data = json.load(resp)
    out: dict[Version, str] = {}
    for ver_str, files in data["releases"].items():
        if not files:
            continue
        try:
            v = Version(ver_str)
        except Exception:
            continue
        if v.is_prerelease or v.is_devrelease or v.micro != 0:
            continue
        out[v] = files[0].get("upload_time", "")[:10]
    return out


def run_git(args: list[str]) -> None:
    print(f"  $ git {' '.join(args)}", file=sys.stderr)
    subprocess.run(["git", *args], cwd=REPO_ROOT, check=True)


def load_ledger() -> dict:
    return json.loads(LEDGER_PATH.read_text())


def write_ledger(ledger: dict) -> None:
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2) + "\n")


def drift_guard(existing: list[dict], pypi: dict[Version, str]) -> int:
    """Abort if any tracked entry no longer matches PyPI."""
    for entry in existing:
        v = Version(entry["pymc"])
        expected = pypi.get(v)
        if expected is None:
            print(
                f"ERROR: ledger entry pymc {v} is not on PyPI; refusing to run",
                file=sys.stderr,
            )
            return 1
        if entry.get("released_at") != expected:
            print(
                f"ERROR: ledger entry pymc {v} has released_at "
                f"{entry.get('released_at')!r} but PyPI says {expected!r}; "
                "refusing to run",
                file=sys.stderr,
            )
            return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_version", default="5.13.0")
    ap.add_argument("--to", dest="to_version", default=None)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap number of new commits to append (unbounded if omitted)",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    floor = Version(args.from_version)
    ceiling = Version(args.to_version) if args.to_version else None

    ledger = load_ledger()
    existing: list[dict] = ledger.get("minor_releases") or []
    pypi = fetch_minor_releases()

    rc = drift_guard(existing, pypi)
    if rc:
        return rc

    tracked = {Version(e["pymc"]) for e in existing}
    highest = max(tracked) if tracked else None

    new: list[dict] = []
    for v in sorted(pypi):
        if v < floor:
            continue
        if ceiling and v > ceiling:
            continue
        if highest is not None and v <= highest:
            continue
        new.append({"pymc": str(v), "released_at": pypi[v]})

    if args.limit is not None:
        new = new[: args.limit]

    if not new:
        print("no new minor releases to append", file=sys.stderr)
        return 0

    print(f"Planning {len(new)} commit(s):", file=sys.stderr)
    for entry in new:
        print(f"  pymc {entry['pymc']} ({entry['released_at']})", file=sys.stderr)
    if args.dry_run:
        return 0

    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT, text=True,
    ).strip()
    if branch != "timeline":
        print(
            f"ERROR: refusing to append onto branch '{branch}' — "
            "run 'git checkout timeline' first",
            file=sys.stderr,
        )
        return 2
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True,
    ).strip():
        print("ERROR: working tree not clean", file=sys.stderr)
        return 3

    cumulative = list(existing)
    for entry in new:
        cumulative.append(entry)
        ledger["minor_releases"] = cumulative
        write_ledger(ledger)
        run_git(["add", str(LEDGER_PATH.relative_to(REPO_ROOT))])
        run_git(["commit", "-m", f"Track pymc {entry['pymc']}"])

    print(f"Appended {len(new)} commit(s) to branch 'timeline'", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
