"""Poll PyPI for new pymc minor releases and update the version ledger.

Each run reads ``versions/tracked_versions.json``, fetches the pymc release
list from PyPI, filters to X.Y.0 releases newer than anything already
tracked, and appends them to the ledger. Patch releases (X.Y.N with N > 0)
are intentionally skipped because they do not meaningfully move the
pytensor pin. Major bumps go through the same filter — nothing in the
logic hardcodes a specific major.

Usage::

    python scripts/bump_release.py              # writes the file in-place
    python scripts/bump_release.py --dry-run    # print what would change

CI (see .github/workflows/detect-releases.yml) calls this daily and opens a
PR when the ledger changes. Each merged PR is one synthetic commit on the
benchmark timeline.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = REPO_ROOT / "versions" / "tracked_versions.json"
PYPI_URL = "https://pypi.org/pypi/pymc/json"


def fetch_pymc_releases() -> list[tuple[Version, str]]:
    with urllib.request.urlopen(PYPI_URL) as resp:  # noqa: S310
        data = json.load(resp)
    out: list[tuple[Version, str]] = []
    for ver_str, files in data["releases"].items():
        if not files:
            continue
        try:
            v = Version(ver_str)
        except Exception:
            continue
        if v.is_prerelease or v.is_devrelease:
            continue
        # Only .0 minors — skip patch releases entirely.
        if v.micro != 0:
            continue
        released_at = files[0].get("upload_time", "")[:10]
        out.append((v, released_at))
    out.sort()
    return out


def load_ledger() -> dict:
    return json.loads(LEDGER_PATH.read_text())


def dump_ledger(ledger: dict) -> None:
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2) + "\n")


def drift_guard(existing: list[dict], candidates: list[tuple[Version, str]]) -> int:
    """Abort if any tracked entry no longer matches PyPI (shared with backfill_history)."""
    by_version = {v: d for v, d in candidates}
    for entry in existing:
        v = Version(entry["pymc"])
        expected = by_version.get(v)
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


def find_new_minors(ledger: dict, candidates: list[tuple[Version, str]]) -> list[dict]:
    tracked = {Version(entry["pymc"]) for entry in ledger["minor_releases"]}
    last = max(tracked) if tracked else None
    new: list[dict] = []
    for v, date in candidates:
        if last is not None and v <= last:
            continue
        new.append({"pymc": str(v), "released_at": date})
    return new


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ledger = load_ledger()
    candidates = fetch_pymc_releases()
    rc = drift_guard(ledger.get("minor_releases") or [], candidates)
    if rc:
        return rc
    new = find_new_minors(ledger, candidates)

    if not new:
        print("no new minor releases", file=sys.stderr)
        return 0

    for entry in new:
        print(f"new: pymc {entry['pymc']} ({entry['released_at']})")

    if args.dry_run:
        return 0

    ledger["minor_releases"].extend(new)
    dump_ledger(ledger)
    print(f"wrote {len(new)} entries to {LEDGER_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
