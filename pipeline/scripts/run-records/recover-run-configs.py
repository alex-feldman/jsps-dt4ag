#!/usr/bin/env python3
"""Archive the REAL config of every already-executed run whose file still matches.

Runs before 2026-08-21 froze no config, because nothing did. But a config file
is only lost if it was EDITED since, and many were not: the six arabidopsis
configs, for instance, have not been touched since the runs that used them. For
those, the real config is still sitting on disk and can simply be archived.

This copies each such file VERBATIM to ``<data_root>/configs/runs/<run-id>.ini``,
the same place and name the pipeline now writes for every new run.

**It only claims a config it can PROVE still matches.** For each run it compares
the file on disk against what ``run-log.csv`` recorded at run time: the images
subpath, the iteration count, the downscale factor, the masking route and the
training method. Every checkable field must agree. A single mismatch means the
file has been edited since, so it is no longer that run's config, and the run is
reported as unrecoverable rather than archived.

That check is necessarily partial: it can only compare fields the run log
carries, so a change confined to, say, a COLMAP flag would pass unnoticed. The
archived record says which fields were verified, so the strength of the claim is
visible rather than implied. This is why the check is worth having anyway: every
config edit observed on this project so far has moved one of these five fields,
because they are the ones anybody has reason to change.

Runs whose config HAS changed are not guessed at. They are listed, with the
field that differs, so you can recover them by hand if the old value is known.

Idempotent, never overwrites an existing ``<run-id>.ini``, and read-only against
everything except ``configs/runs/``.

Usage::

    python pipeline/scripts/run-records/recover-run-configs.py --data-root <path> [--dry-run]
"""

from __future__ import annotations

import argparse
import configparser
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]

FOOTER = """\

# ----------------------------------------------------------------
# ARCHIVED {stamp}, after the fact. Everything above is the real config
# file this run used, copied verbatim: it had not been edited between the
# run and the archiving, which was verified by comparing it against what
# run-log.csv recorded at run time.
#
# Verified fields: {verified}
# Only those could be checked, because they are what the run log carries.
#
# This IS a runnable config. Re-running derives a new run id unless [run]
# date and run_count are pinned to the values below.
# ----------------------------------------------------------------
[run-record]
run_id          = {run_id}
record_type     = recovered-verified
archived        = {stamp}
config_source   = {source}
logged_at       = {logged_at}
verified_fields = {verified}
training_config = see outputs/**/{run_id}/splatfacto/*/config.yml
"""


def read_run_log(data_root: Path) -> Dict[str, dict]:
    """Latest run-log row per run id."""
    log = data_root / "run-log.csv"
    if not log.is_file():
        return {}
    rows: Dict[str, dict] = {}
    with log.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            run_id = (row.get("run_id") or "").strip()
            if run_id:
                rows[run_id] = row
    return rows


def resolve_config(raw: str, repo_root: Path) -> Optional[Path]:
    """The config path a run-log row names, absolute or repo-relative."""
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = repo_root / path
    return path if path.is_file() else None


def expected_images_subpath(logged_images: str, data_root: Path) -> Optional[str]:
    """The `images_subpath` that would produce the logged absolute path."""
    if not logged_images:
        return None
    try:
        return str(Path(logged_images).relative_to(data_root / "datasets"))
    except ValueError:
        return None


def compare(config_path: Path, row: dict, data_root: Path
            ) -> Tuple[List[str], List[str]]:
    """Return (verified field names, mismatch descriptions).

    A field the run log left blank is not checkable and is silently not
    verified: older rows predate several columns. A field that IS present and
    disagrees is a mismatch, and one is enough to disqualify the file.
    """
    parser = configparser.ConfigParser()
    try:
        parser.read(config_path, encoding="utf-8")
    except (OSError, configparser.Error) as exc:
        return [], [f"unreadable: {exc}"]

    def cfg(section: str, key: str, default: str = "") -> str:
        return (parser.get(section, key, fallback=default) or "").strip()

    verified: List[str] = []
    mismatches: List[str] = []

    def check(name: str, expected: Optional[str], actual: str) -> None:
        if expected in (None, ""):
            return
        if str(expected) == str(actual):
            verified.append(name)
        else:
            mismatches.append(f"{name}: log says {expected!r}, file has {actual!r}")

    check("images_subpath",
          expected_images_subpath(row.get("images_path", ""), data_root),
          cfg("dataset", "images_subpath"))
    check("max_num_iterations", row.get("max_num_iterations"),
          cfg("train", "max_num_iterations", "30000"))
    check("train_method", row.get("train_method"),
          cfg("train", "method", "splatfacto"))

    # The run log stores the CONFIGURED downscale, writing the literal `auto`
    # when the key is 0 or absent, so the comparison is against that spelling.
    logged_ds = (row.get("downscale_factor") or "").strip()
    if logged_ds:
        file_ds = cfg("train", "downscale_factor", "0")
        check("downscale_factor", logged_ds,
              "auto" if file_ds in ("0", "") else file_ds)

    logged_masks = (row.get("masks") or "").strip()
    if logged_masks:
        file_masks = cfg("dataset", "use_masks", "false").lower()
        check("use_masks", logged_masks,
              "used" if file_masks in ("true", "yes", "1", "on") else "none")

    return verified, mismatches


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT,
                        help="repository the run log's relative config paths resolve against")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stamp", default="",
                        help="date recorded in the archived file; defaults to today")
    args = parser.parse_args(argv)

    data_root: Path = args.data_root.expanduser().resolve()
    rows = read_run_log(data_root)
    if not rows:
        print(f"ERROR: no readable run-log.csv under {data_root}", file=sys.stderr)
        return 2

    stamp = args.stamp or __import__("datetime").date.today().isoformat()
    archive = data_root / "configs" / "runs"
    recovered: List[str] = []
    changed: List[str] = []
    missing: List[str] = []
    already: List[str] = []

    for run_id in sorted(rows):
        destination = archive / f"{run_id}.ini"
        if destination.is_file():
            already.append(run_id)
            continue
        config_path = resolve_config(rows[run_id].get("config_file", ""), args.repo_root)
        if config_path is None:
            missing.append(f"{run_id}: config file no longer on disk "
                           f"({rows[run_id].get('config_file') or 'not recorded'})")
            continue
        verified, mismatches = compare(config_path, rows[run_id], data_root)
        if mismatches:
            changed.append(f"{run_id}: {mismatches[0]}")
            continue
        if not verified:
            changed.append(f"{run_id}: nothing checkable in the run log, not claimed")
            continue
        if not args.dry_run:
            archive.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                config_path.read_text(encoding="utf-8")
                + FOOTER.format(stamp=stamp, run_id=run_id, source=config_path,
                                logged_at=rows[run_id].get("timestamp", ""),
                                verified=", ".join(verified)),
                encoding="utf-8")
        recovered.append(f"{run_id}  <- {config_path.name}  "
                         f"(verified {len(verified)}: {', '.join(verified)})")

    verb = "would recover" if args.dry_run else "RECOVERED"
    print(f"{verb} {len(recovered)} real config(s):")
    for line in recovered:
        print(f"  {line}")
    if already:
        print(f"\nalready archived, left alone ({len(already)}):")
        for run_id in already:
            print(f"  {run_id}")
    if changed:
        print(f"\nNOT recoverable, the file has changed since the run ({len(changed)}).")
        print("Recover these by hand if you know the old value; nothing is guessed:")
        for line in changed:
            print(f"  {line}")
    if missing:
        print(f"\nconfig file gone ({len(missing)}):")
        for line in missing:
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
