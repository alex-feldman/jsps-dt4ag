#!/usr/bin/env python3
"""Reconstruct a run record for every run that predates the config archive.

Runs executed before 2026-08-21 froze no config, because nothing did. This walks
what those runs DID leave behind and writes one record per run to
``<data_root>/configs/runs/<run-id>.reconstructed.ini``.

WHAT THIS CANNOT DO, stated first because it decides how the output must be
read: **it cannot recover the config file a historical run used.** Run configs
are gitignored, nothing versioned them, and editing one overwrites it. So a
reconstructed record is assembled from evidence the run left elsewhere, and
every field carries the source it came from. It is a reconstruction and it says
so, in its filename, in its ``record_type``, and in a header no reader can miss.

**A reconstructed record is deliberately NOT a loadable config.** It carries no
``[paths]`` or ``[dataset]`` section, so ``load_config`` refuses it. That is the
point: the pipeline-level settings of a historical run (COLMAP flags, masking
route, file extensions, export labels) are frequently unknown, and a file that
looked runnable would invite someone to re-run it believing they had reproduced
the original. Genuine frozen configs, written by the pipeline from 2026-08-21,
are ``<run-id>.ini`` with no ``.reconstructed`` and those ARE runnable.

Evidence used, in descending order of authority:

1. ``outputs/**/<run-id>/splatfacto/<timestamp>/config.yml`` -- written BY the
   run, so it cannot drift or overstate. Authoritative for training: iteration
   count, downscale factor, seed, method, and the data path.
2. ``run-log.csv`` -- written by the pipeline at run START, so it states intent.
   The only source for masking, COLMAP version and the config file's PATH. Not
   every run is in it: the log postdates the oldest runs.
3. ``exports/*.ply`` and the COLMAP workspace -- existence and vertex counts,
   which corroborate that a run produced something.

Idempotent, and read-only against everything except ``configs/runs/``. Safe to
re-run: it rewrites reconstructed records and never touches a genuine one.

Usage::

    python pipeline/scripts/run-records/backfill-run-records.py --data-root <path>
    python pipeline/scripts/run-records/backfill-run-records.py --data-root <path> --dry-run
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

RECORD_SUFFIX = ".reconstructed.ini"

HEADER = """\
# =====================================================================
# RECONSTRUCTED RUN RECORD -- NOT the config this run used.
#
# This run predates the per-run config archive, so the config it actually
# ran with was never captured and has since been overwritten or lost. The
# fields below were reassembled from what the run left on disk. [sources]
# says where each one came from and [unrecoverable] says what could not be
# established at all.
#
# This file is deliberately not loadable as a config: it has no [paths] or
# [dataset] section, so the pipeline refuses it. Do not treat it as a recipe
# for reproducing the run. A genuine frozen config is <run-id>.ini, with no
# ".reconstructed", and that one IS runnable.
#
# Written by pipeline/scripts/run-records/backfill-run-records.py.
# =====================================================================
"""

# Scalars worth lifting out of nerfstudio's config.yml. It is a python-tagged
# YAML dump, so it is read line-wise rather than with a YAML loader: parsing it
# properly would mean importing nerfstudio, and this script must run anywhere.
YAML_SCALARS = {
    "experiment_name": re.compile(r"^experiment_name:\s*(\S.*)$"),
    "method_name": re.compile(r"^method_name:\s*(\S.*)$"),
    "timestamp": re.compile(r"^timestamp:\s*(\S.*)$"),
    "max_num_iterations": re.compile(r"^max_num_iterations:\s*(\S.*)$"),
    "seed": re.compile(r"^\s+seed:\s*(\S.*)$"),
    "downscale_factor": re.compile(r"^\s+downscale_factor:\s*(\S.*)$"),
    "background_color": re.compile(r"^\s+background_color:\s*(\S.*)$"),
}


def parse_training_config(path: Path) -> Dict[str, str]:
    """Lift the scalars and the `data:` PosixPath out of a run's config.yml."""
    found: Dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return found

    for line in lines:
        for key, pattern in YAML_SCALARS.items():
            if key in found:
                continue
            match = pattern.match(line)
            if match:
                found[key] = match.group(1).strip()

    # `data:` is dumped as a pathlib.PosixPath apply, i.e. a tagged line
    # followed by one `- <part>` per path component. Rebuild it by hand.
    for index, line in enumerate(lines):
        if line.startswith("data:") and "PosixPath" in line:
            parts: List[str] = []
            for follower in lines[index + 1:]:
                stripped = follower.strip()
                if not stripped.startswith("- "):
                    break
                # A component the dumper had to quote, e.g. `- '02'`, arrives
                # with its quotes attached; they are not part of the name.
                parts.append(stripped[2:].strip("'\""))
            if parts:
                found["data"] = str(Path(*parts)) if parts != ["/"] else "/"
            break
    return found


def read_run_log(data_root: Path) -> Dict[str, dict]:
    """Latest run-log row per run id. Empty when the log does not exist."""
    log = data_root / "run-log.csv"
    if not log.is_file():
        return {}
    rows: Dict[str, dict] = {}
    with log.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            run_id = (row.get("run_id") or "").strip()
            if run_id:
                # Later rows win: a re-logged run id is a resume or a re-export,
                # and the last row is the most complete statement of intent.
                rows[run_id] = row
    return rows


def discover_runs(data_root: Path) -> Dict[str, List[Path]]:
    """run id -> its config.yml files, newest last. Keyed by the run directory.

    A run id can hold several training attempts (a resume, or a re-train into
    the same workspace), each its own timestamped directory. All are recorded.
    """
    runs: Dict[str, List[Path]] = {}
    outputs = data_root / "outputs"
    if not outputs.is_dir():
        return runs
    for config_yml in sorted(outputs.rglob("splatfacto/*/config.yml")):
        run_dir = config_yml.parent.parent.parent
        runs.setdefault(run_dir.name, []).append(config_yml)
    return runs


def export_belongs_to(name: str, run_id: str) -> bool:
    """Does export ``name`` belong to ``run_id``?

    Exports are named ``<capture>_<run-id>_splat_<labels>.ply``, so the run id
    is the tail of everything left of ``_splat``. Matching on that boundary
    rather than on a substring is not fussiness: the legacy run ids are short
    (``02``, ``03``, ``04-312``) and a substring match on ``02`` claimed eight
    unrelated exports, every one of them from a different capture.
    """
    if "_splat" not in name:
        return False
    left = name.split("_splat", 1)[0]
    return left == run_id or left.endswith(f"_{run_id}")


def find_artefacts(data_root: Path, run_id: str) -> Dict[str, str]:
    """Exports and COLMAP workspace belonging to a run id, if any survive."""
    found: Dict[str, str] = {}
    exports = sorted(p for p in (data_root / "exports").glob("*.ply")
                     if export_belongs_to(p.name, run_id))
    if exports:
        found["export_ply"] = ", ".join(str(p) for p in exports)
    workspaces = sorted((data_root / "colmap").rglob(run_id))
    workspaces = [w for w in workspaces if w.is_dir()]
    if workspaces:
        found["colmap_workspace"] = ", ".join(str(w) for w in workspaces)
    return found


def build_record(run_id: str, configs: List[Path], log_row: Optional[dict],
                 artefacts: Dict[str, str]) -> str:
    """Render one reconstructed record as INI text."""
    record: Dict[str, str] = {"run_id": run_id, "record_type": "reconstructed"}
    sources: Dict[str, str] = {}
    unrecoverable: List[str] = []

    def put(key: str, value: Optional[str], source: str) -> None:
        if value not in (None, "", "null"):
            record[key] = str(value)
            sources[key] = source

    # 1. The run's own training config: the strongest evidence there is.
    training = parse_training_config(configs[-1]) if configs else {}
    if training:
        yml = f"config.yml ({configs[-1]})"
        put("train_method", training.get("method_name"), yml)
        put("max_num_iterations", training.get("max_num_iterations"), yml)
        put("downscale_factor", training.get("downscale_factor"), yml)
        put("seed", training.get("seed"), yml)
        put("background_color", training.get("background_color"), yml)
        put("trained_at", training.get("timestamp"), yml)
        put("nerfstudio_data_path", training.get("data"), yml)
        if training.get("downscale_factor") in (None, "null"):
            unrecoverable.append(
                "effective downscale_factor: config.yml records null, meaning "
                "nerfstudio probed for a level on disk, and which level it "
                "found was never recorded")
    else:
        unrecoverable.append(
            "everything training-side: no readable config.yml under this run")

    record["training_configs"] = ", ".join(str(p) for p in configs) or "(none)"
    sources["training_configs"] = "filesystem (the run's own output directory)"
    if len(configs) > 1:
        record["training_attempts"] = str(len(configs))
        sources["training_attempts"] = "filesystem (the run's own output directory)"

    # 2. The run log: intent at start, and the only place masking and the
    #    config file's path are recorded.
    if log_row:
        log = "run-log.csv"
        put("logged_at", log_row.get("timestamp"), log)
        put("images_path", log_row.get("images_path"), log)
        put("dataset", log_row.get("dataset"), log)
        put("masks", log_row.get("masks"), log)
        put("colmap_version", log_row.get("colmap_version"), log)
        put("object_id", log_row.get("object_id"), log)
        put("imaging_date", log_row.get("imaging_date"), log)
        put("note", log_row.get("note"), log)
        put("config_source", log_row.get("config_file"),
            f"{log} (the PATH only; its contents at the time are lost)")
        put("configured_max_num_iterations", log_row.get("max_num_iterations"),
            f"{log} (intent at start)")
        put("configured_downscale_factor", log_row.get("downscale_factor"),
            f"{log} (intent at start)")
        unrecoverable.append(
            "the contents of the config file named by config_source: it was "
            "gitignored, nothing versioned it, and it has since been edited")
    else:
        unrecoverable.append(
            "everything the run log holds (masking route, COLMAP version, "
            "config file path, capture provenance): this run predates the run "
            "log or was driven outside the pipeline")

    # The weakest category, and labelled as such. These are the only fields not
    # read out of something the run itself wrote: they are ASSOCIATIONS, made by
    # matching the run id against a filename or a directory name. Strong ones,
    # and anchored rather than substring-matched, but an association a reader
    # should be able to re-check rather than take on trust.
    for key, value in artefacts.items():
        put(key, value,
            "filesystem (associated by run id in the name, not a link the "
            "run recorded; re-checkable by inspection)")

    unrecoverable.append(
        "the pipeline git commit: nothing recorded one until the config "
        "archive landed, so it is unknown for every run this script covers")

    lines = [HEADER, "[run-record]"]
    width = max(len(k) for k in record)
    lines += [f"{k.ljust(width)} = {v}" for k, v in record.items()]
    lines += ["", "# Where each field above came from.", "[sources]"]
    width = max(len(k) for k in sources) if sources else 1
    lines += [f"{k.ljust(width)} = {v}" for k, v in sources.items()]
    lines += ["", "# What no surviving evidence can establish.", "[unrecoverable]"]
    lines += [f"item{n} = {text}" for n, text in enumerate(unrecoverable, 1)]
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", required=True, type=Path,
                        help="the pipeline data root holding outputs/ and run-log.csv")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be written, write nothing")
    args = parser.parse_args(argv)

    data_root: Path = args.data_root.expanduser().resolve()
    if not (data_root / "outputs").is_dir():
        print(f"ERROR: no outputs/ under {data_root}", file=sys.stderr)
        return 2

    log_rows = read_run_log(data_root)
    runs = discover_runs(data_root)
    if not runs:
        print(f"No runs found under {data_root / 'outputs'}")
        return 0

    archive = data_root / "configs" / "runs"
    written = skipped = 0
    for run_id in sorted(runs):
        genuine = archive / f"{run_id}.ini"
        if genuine.is_file():
            # A real frozen config outranks any reconstruction of it.
            skipped += 1
            print(f"  skip  {run_id}  (genuine frozen config exists)")
            continue
        text = build_record(run_id, runs[run_id], log_rows.get(run_id),
                            find_artefacts(data_root, run_id))
        destination = archive / f"{run_id}{RECORD_SUFFIX}"
        if not args.dry_run:
            archive.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
        written += 1
        print(f"  write {run_id}  ({'logged' if run_id in log_rows else 'NOT in run log'},"
              f" {len(runs[run_id])} training attempt(s))")

    verb = "would write" if args.dry_run else "wrote"
    print(f"\n{verb} {written} reconstructed record(s), skipped {skipped} "
          f"with a genuine frozen config, into {archive}")
    print(f"run ids on disk: {len(runs)}   run ids in run-log.csv: {len(log_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
