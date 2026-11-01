#!/usr/bin/env python3
"""Reconstruct every capture beneath a collection, one at a time, from one base config.

Usage, from the repository root::

    uv run python pipeline/run_collection.py --config pipeline/configs/base.ini \\
        --collection {collection path} --catalog-root /data --catalog-root-id wsl-data --dry-run

A capture is a directory containing ``images/`` (``LAYOUT.md``, "The discovery
rule"). Each capture runs ``run_pipeline.py`` in its own process, with a config
derived from the base by replacing ``[dataset] images_subpath`` and making
``[paths] data_root`` absolute, and with catalog emission turned on so the
child leaves the job record this module later reads to decide completion.
Nothing here writes a second record: completion is read from catalog jobs
(``why_incomplete``), never from a store of its own.

Captures run sequentially. One failing capture does not stop the others; an
interrupted or terminated capture stops the batch. ``SEQUENTIAL-RUNS.md`` is
the operator documentation, including every conservative choice made here.
"""

from __future__ import annotations

import argparse
import configparser
import datetime as _dt
import hashlib
import io
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import run_pipeline  # noqa: E402
from dt4ag_config import ConfigError, find_config, load_config  # noqa: E402
from plant_catalog import ID_PATTERN, read_progress  # noqa: E402

#: The directory whose presence makes its parent a capture (LAYOUT.md).
IMAGES_DIRNAME = "images"
#: Artifact roles ``run_pipeline`` gives an exported splat; one must survive for completion.
EXPORT_ROLES = ("gaussian-splat", "point-cloud")
#: Derived per-capture configs go under ``{data_root}/{derived_dirname}/`` + this.
DERIVED_CONFIGS_DIRNAME = "collection-configs"
#: Child exit codes that stop the batch: interrupted (Ctrl-C) and terminated (SIGTERM).
STOP_CODES = (130, run_pipeline.EXIT_TERMINATED)
#: After Ctrl-C, how long the child (which got the same SIGINT) has to finalize its record.
CHILD_GRACE_SECONDS = 30
EXIT_FAILED = 1
EXIT_COLLECTION = 2


class CollectionError(Exception):
    """The collection as a whole cannot run, so nothing is started (exit 2)."""


@dataclass
class Base:
    path: Path
    text: str
    data_root: Path
    datasets_dir: Path


@dataclass
class Capture:
    path: Path
    rel: Path
    capture_id: str
    images_subpath: str
    config_text: str = ""
    sections: dict = field(default_factory=dict)
    photographs: List[str] = field(default_factory=list)
    inputs: List[Path] = field(default_factory=list)
    cfg: object = None
    problem: Optional[str] = None
    complete_job: Optional[str] = None
    reason: str = ""
    status: str = "pending"
    detail: str = ""
    job_id: Optional[str] = None
    elapsed: Optional[float] = None


@dataclass
class Plan:
    base: Base
    collection: Path
    catalog_root: Path
    root_id: str
    captures: List[Capture]
    ignored: List[Tuple[Path, str]]


# --------------------------------------------------------------------------
# discovery and derived configs
# --------------------------------------------------------------------------

def _ini_parser() -> configparser.ConfigParser:
    # Values are carried verbatim (no interpolation), with inline comments
    # dropped exactly as dt4ag_config.load_config drops them.
    return configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#", ";"))


def load_base(path) -> Base:
    """Read the base config and resolve its data root; the loader validates the rest per capture."""
    path = Path(path).expanduser()
    parser = _ini_parser()
    try:
        text = path.read_text(encoding="utf-8")
        parser.read_string(text, source=str(path))
    except (OSError, configparser.Error) as exc:
        raise CollectionError(f"cannot read base config {path}: {exc}") from exc
    raw = parser.get("paths", "data_root", fallback="").strip()
    if not raw:
        raise CollectionError(f"key 'data_root' in section [paths] of base config {path} is required")
    data_root = Path(os.path.expandvars(os.path.expanduser(raw)))
    if not data_root.is_absolute():
        data_root = (path.parent / data_root).resolve()
    datasets_dir = data_root / (parser.get("paths", "datasets_dirname", fallback="").strip() or "datasets")
    if not datasets_dir.is_dir():
        raise CollectionError(f"datasets directory does not exist: {datasets_dir} (from base config {path})")
    return Base(path=path, text=text, data_root=data_root, datasets_dir=datasets_dir)


def derive_config_text(base: Base, images_subpath: str) -> str:
    """The base config with this capture's ``images_subpath`` and an absolute ``data_root``."""
    parser = _ini_parser()
    parser.read_string(base.text, source=str(base.path))
    parser.set("paths", "data_root", str(base.data_root))
    if not parser.has_section("dataset"):
        parser.add_section("dataset")
    parser.set("dataset", "images_subpath", images_subpath)
    stream = io.StringIO()
    stream.write(f"# Derived by run_collection.py from {base.path} for {images_subpath}.\n"
                 "# Named by its content hash and never rewritten: edit the base config, not this file.\n\n")
    parser.write(stream)
    return stream.getvalue()


def discover_captures(collection: Path) -> Tuple[List[Path], List[Tuple[Path, str]]]:
    """Captures beneath ``collection`` in path order, plus the symlinks not followed.

    Hidden directories are not searched. Symbolic links are never followed,
    so a capture is reached by one path only. A capture found inside another
    capture is ambiguous and refuses the whole collection.
    """
    def unreadable(error):
        raise CollectionError(f"cannot read {error.filename}: {error.strerror}")

    captures, ignored = [], []
    for folder, dirnames, _ in os.walk(collection, onerror=unreadable):
        folder = Path(folder)
        kept = []
        for name in sorted(dirnames):
            if name.startswith("."):
                continue
            if (folder / name).is_symlink():
                ignored.append((folder / name, "symbolic link, not followed"
                                + ("; its parent is not a capture" if name == IMAGES_DIRNAME else "")))
                continue
            kept.append(name)
        dirnames[:] = kept
        if IMAGES_DIRNAME in kept:
            captures.append(folder)
    nested = sorted(f"{inner.relative_to(collection)} inside {outer.relative_to(collection)}"
                    for inner in captures for outer in captures if outer in inner.parents)
    if nested:
        raise CollectionError(
            "a capture lies inside another capture, so discovery is ambiguous: "
            + "; ".join(nested) + ". A capture is the nearest directory holding images/, "
            "and nothing beneath a capture may hold images/ again (pipeline/LAYOUT.md).")
    order = lambda path: path.relative_to(collection).parts  # noqa: E731
    return sorted(captures, key=order), sorted(ignored, key=lambda item: order(item[0]))


def capture_id_for(rel: Path) -> str:
    """Catalog capture ID from the capture's path under ``datasets/``, as kebab-case."""
    return re.sub(r"[^a-z0-9]+", "-", rel.as_posix().lower()).strip("-")


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def capture_record_conflict(catalog_root: Path, root_id: str, capture_id: str, images_path: Path) -> Optional[str]:
    """Why an existing catalog capture record forbids reusing this ID for this capture, or None."""
    path = catalog_root / "catalog" / "captures" / f"{capture_id}.json"
    if not path.exists():
        return None
    try:
        source = json.loads(path.read_text(encoding="utf-8")).get("source") or {}
    except (OSError, ValueError) as exc:
        return f"catalog capture record {path} is unreadable: {exc}"
    expected = images_path.resolve().relative_to(catalog_root.resolve()).as_posix()
    if source.get("root_id") != root_id or source.get("relative_path") != expected:
        return (f"catalog capture id {capture_id!r} already names "
                f"{source.get('root_id')}:{source.get('relative_path')}, not {root_id}:{expected}")
    return None


def prepare(capture: Capture, base: Base, catalog_root: Path, root_id: str) -> None:
    """Derive and validate one capture's config; set ``capture.problem`` instead of raising."""
    capture.config_text = derive_config_text(base, capture.images_subpath)
    if not re.fullmatch(ID_PATTERN, capture.capture_id):
        capture.problem = f"no catalog capture id can be derived from {capture.rel}"
        return
    with tempfile.TemporaryDirectory() as folder:
        probe = Path(folder) / "capture.ini"
        probe.write_text(capture.config_text, encoding="utf-8")
        try:
            cfg = load_config(probe)
        except ConfigError as exc:
            capture.problem = f"config error: {exc}"
            return
        capture.sections = run_pipeline.ini_sections(probe)
    capture.cfg = cfg
    if cfg.run_count:
        capture.problem = ("[run] run_count is pinned; a collection run needs it empty so a rerun "
                           "gets a new run id and never writes into an earlier run's outputs")
        return
    if not cfg.quit_on_train_completion:
        capture.problem = ("[train] quit_on_train_completion is false; an unattended run "
                           "would never return from ns-train")
        return
    capture.photographs = run_pipeline.selected_images(cfg)
    if not capture.photographs:
        capture.problem = f"no photographs under {cfg.images_path}"
        return
    outside = [str(p) for p in (cfg.images_path, cfg.colmap_dir, cfg.outputs_dir, cfg.exports_dir, cfg.run_log)
               if not _within(p, catalog_root)]
    if outside:
        capture.problem = f"not beneath the catalog root {catalog_root}: {', '.join(outside)}"
        return
    capture.problem = capture_record_conflict(catalog_root, root_id, capture.capture_id, cfg.images_path)
    capture.inputs = run_pipeline.count_images(cfg.images_path)
    if cfg.use_masks and cfg.masks_path != cfg.images_path:
        capture.inputs += run_pipeline.count_images(cfg.masks_path)


def write_derived_config(capture: Capture) -> Path:
    """Write the derived config, named by its content hash so no file is ever rewritten."""
    digest = hashlib.sha256(capture.config_text.encode("utf-8")).hexdigest()[:12]
    target = capture.cfg.derived_dir / DERIVED_CONFIGS_DIRNAME / capture.cfg.capture_rel / f"{digest}.ini"
    if target.exists():
        if target.read_text(encoding="utf-8") != capture.config_text:
            raise OSError(f"{target} exists with different content; refusing to overwrite it")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(capture.config_text, encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


# --------------------------------------------------------------------------
# completion, read from catalog job records
# --------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[dict]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def read_jobs(catalog_root: Path) -> List[dict]:
    """Every readable job record, oldest first. An unreadable one proves nothing and is left out."""
    jobs = [job for job in map(_load_json, sorted((catalog_root / "catalog" / "jobs").glob("*.json")))
            if job and job.get("kind") == "job"]
    return sorted(jobs, key=lambda job: (str(job.get("started_at") or ""), str(job.get("id"))))


def _section_changes(recorded: dict, current: dict) -> List[str]:
    keys = {(s, k) for s, values in list(recorded.items()) + list(current.items()) for k in values}
    return sorted(f"[{s}] {k}" for s, k in keys
                  if (recorded.get(s) or {}).get(k) != (current.get(s) or {}).get(k))


def _check_exports(job: dict, catalog_root: Path, root_id: str) -> Optional[str]:
    exports = []
    for artifact_id in job.get("output_artifact_ids") or []:
        record = (_load_json(catalog_root / "catalog" / "artifacts" / f"{artifact_id}.json")
                  if re.fullmatch(r"[a-z0-9-]+", str(artifact_id)) else None)
        if record is None:
            return f"artifact record {artifact_id} is missing or unreadable"
        if record.get("role") in EXPORT_ROLES:
            exports.append(record)
    if not exports:
        return "no export artifact recorded"
    recorded_sizes = {entry.get("artifact_id"): entry.get("size_bytes")
                      for entry in read_progress(job)["artifacts"] or []}
    for record in exports:
        location = record.get("location") or {}
        relative = str(location.get("relative_path") or "")
        path = catalog_root / relative
        if location.get("root_id") != root_id or not relative or not _within(path, catalog_root):
            return f"export {relative or record.get('id')} is recorded under another catalog root"
        if not path.is_file():
            return f"export {relative} no longer exists"
        size = path.stat().st_size
        if size < run_pipeline.MIN_EXPORT_BYTES:
            return f"export {relative} is {size} bytes, below MIN_EXPORT_BYTES"
        recorded = recorded_sizes.get(record.get("id"))
        if recorded is not None and recorded != size:
            return f"export {relative} is {size} bytes but its record says {recorded}"
        try:
            vertices = run_pipeline.read_ply_vertex_count(path)
        except OSError as exc:
            return f"export {relative} is unreadable: {exc}"
        if vertices == 0:
            return f"export {relative} declares 0 vertices"
    return None


def why_incomplete(job: dict, capture: Capture, catalog_root: Path, root_id: str) -> Optional[str]:
    """None when this job proves the capture finished with today's settings and inputs; else why not."""
    if job.get("execution_status") != "completed" or job.get("exit_code") != 0:
        return f"status {job.get('execution_status')}, exit {job.get('exit_code')}"
    progress = read_progress(job)
    if not progress["schema_version"]:
        return "record predates progress milestones, so completion is unproven"
    if "verified" not in [m.get("name") for m in progress["milestones"] or []]:
        return f"never verified (furthest milestone {progress['furthest_milestone']})"
    problem = _check_exports(job, catalog_root, root_id)
    if problem:
        return problem
    recorded = (job.get("config") or {}).get("ini")
    if not isinstance(recorded, dict):
        return "record holds no configuration to compare"
    changes = _section_changes(recorded, capture.sections)
    if changes:
        return "settings changed since that run: " + ", ".join(changes)
    images = job.get("selected_images")
    if images != capture.photographs:
        before, now = set(images or []), set(capture.photographs)
        return (f"image set changed since that run: {len(now - before)} added, "
                f"{len(before - now)} removed" + ("" if before != now else ", order differs"))
    try:
        started = _dt.datetime.fromisoformat(str(job.get("started_at"))).timestamp()
    except ValueError:
        return "record has no readable start time"
    for path in capture.inputs:
        try:
            if path.stat().st_mtime > started:
                return f"input {path.relative_to(capture.path)} was modified after that run started"
        except OSError:
            return f"input {path} is unreadable"
    return None


def find_completed(capture: Capture, jobs: Sequence[dict], catalog_root: Path, root_id: str) -> Tuple[Optional[str], str]:
    """(newest completing job id, reason) for this capture; the reason explains a None."""
    candidates = [job for job in jobs if job.get("capture_id") == capture.capture_id]
    if not candidates:
        return None, "no catalog job for this capture"
    newest_reason = None
    for job in reversed(candidates):
        reason = why_incomplete(job, capture, catalog_root, root_id)
        if reason is None:
            return job.get("id"), f"completed by {job.get('id')}"
        newest_reason = newest_reason or f"newest job {job.get('id')}: {reason}"
    return None, newest_reason


# --------------------------------------------------------------------------
# planning and execution
# --------------------------------------------------------------------------

def plan_collection(args: argparse.Namespace) -> Plan:
    try:
        base = load_base(find_config(args.config))
    except ConfigError as exc:
        raise CollectionError(str(exc)) from exc
    if not args.catalog_root or not args.catalog_root_id:
        raise CollectionError("a collection run needs --catalog-root and --catalog-root-id (or "
                              "PLANT_CATALOG_ROOT and PLANT_CATALOG_ROOT_ID): completion is read "
                              "from the catalog job records its runs write")
    if not re.fullmatch(ID_PATTERN, args.catalog_root_id):
        raise CollectionError(f"catalog root id {args.catalog_root_id!r} must be kebab-case")
    catalog_root = Path(args.catalog_root).expanduser().resolve()
    if not catalog_root.is_dir():
        raise CollectionError(f"catalog root does not exist: {catalog_root}")

    raw = Path(args.collection).expanduser()
    if raw.is_absolute():
        try:
            rel = Path(os.path.abspath(raw)).relative_to(os.path.abspath(base.datasets_dir))
        except ValueError:
            raise CollectionError(f"collection {raw} is not inside the datasets directory {base.datasets_dir}") from None
    else:
        rel = raw
    if ".." in rel.parts:
        raise CollectionError(f"collection {raw} must not climb out of {base.datasets_dir}")
    collection = base.datasets_dir / rel
    if not collection.is_dir():
        raise CollectionError(f"collection directory does not exist: {collection}")

    paths, ignored = discover_captures(collection)
    if not paths:
        raise CollectionError(f"no capture (a directory containing {IMAGES_DIRNAME}/) beneath {collection}"
                              + "".join(f"\n  ignored {p}: {why}" for p, why in ignored))
    captures = []
    for path in paths:
        rel_capture = path.relative_to(base.datasets_dir)
        captures.append(Capture(path=path, rel=rel_capture, capture_id=capture_id_for(rel_capture),
                                images_subpath=(rel_capture / IMAGES_DIRNAME).as_posix()))
    seen = {}
    for capture in captures:
        seen.setdefault(capture.capture_id, []).append(str(capture.rel))
    clashes = [f"{cid}: {', '.join(rels)}" for cid, rels in seen.items() if len(rels) > 1 and cid]
    if clashes:
        raise CollectionError("captures map to the same catalog capture id; rename one of each: "
                              + "; ".join(clashes))
    for capture in captures:
        prepare(capture, base, catalog_root, args.catalog_root_id)
    jobs = read_jobs(catalog_root)
    for capture in captures:
        if capture.problem is None:
            capture.complete_job, capture.reason = find_completed(
                capture, jobs, catalog_root, args.catalog_root_id)
    return Plan(base, collection, catalog_root, args.catalog_root_id, captures, ignored)


def child_argv(plan: Plan, capture: Capture, config_path: Path, args: argparse.Namespace) -> List[str]:
    argv = [sys.executable, str(PIPELINE_DIR / "run_pipeline.py"), "--config", str(config_path),
            "--catalog-root", str(plan.catalog_root), "--catalog-root-id", plan.root_id,
            "--catalog-capture-id", capture.capture_id]
    if args.allow_full_resolution:
        argv.append("--allow-full-resolution")
    if args.catalog_hash_artifacts:
        argv.append("--catalog-hash-artifacts")
    return argv


def run_child(argv: Sequence[str]) -> int:
    """Run one capture's ``run_pipeline.py`` and return its exit code.

    On Ctrl-C the child received the same SIGINT and finalizes its own job
    record, so it gets ``CHILD_GRACE_SECONDS`` before SIGTERM. On SIGTERM to
    this process the child is sent SIGTERM at once; it finalizes as terminated.
    """
    process = subprocess.Popen(list(argv))
    try:
        return process.wait()
    except run_pipeline.Terminated:
        process.terminate()
        process.wait()
        raise
    except KeyboardInterrupt:
        try:
            process.wait(timeout=CHILD_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait()
        raise


def record_outcome(plan: Plan, capture: Capture, code: int, before: set) -> None:
    """Status from the job record this capture's child wrote, never from its exit code alone."""
    new = [job for job in read_jobs(plan.catalog_root)
           if job.get("id") not in before and job.get("capture_id") == capture.capture_id]
    job = new[-1] if new else None
    if job is None:
        capture.status = "interrupted" if code in STOP_CODES else "failed"
        capture.detail = f"exit {code}; no catalog job was written"
        return
    capture.job_id = job.get("id")
    reason = why_incomplete(job, capture, plan.catalog_root, plan.root_id)
    if code == 0 and reason is None:
        capture.status, capture.detail = "completed", "verified"
        return
    if code == 0:
        capture.status, capture.detail = "failed", f"exit 0 but the record does not show completion: {reason}"
        return
    failure = read_progress(job)["failure"] or {}
    capture.status = "interrupted" if code in STOP_CODES else "failed"
    capture.detail = f"exit {code}"
    if failure.get("cause"):
        capture.detail += f", {failure['cause']}"
    if failure.get("stage"):
        capture.detail += f" in stage {failure['stage']}"
    if failure.get("after_milestone"):
        capture.detail += f" after {failure['after_milestone']}"
    summary = (job.get("error_summary") or "").strip().splitlines()
    if summary:
        capture.detail += f": {summary[0]}"


def execute(plan: Plan, args: argparse.Namespace, runner: Callable[[Sequence[str]], int]) -> int:
    pending = [c for c in plan.captures if c.problem is None and c.complete_job is None]
    for capture in plan.captures:
        if capture.problem is not None:
            capture.status, capture.detail = "invalid", capture.problem
        elif capture.complete_job is not None:
            capture.status, capture.job_id, capture.detail = "skipped", capture.complete_job, "already completed"
    stopped = None
    restore = run_pipeline._catch_sigterm()
    try:
        for index, capture in enumerate(pending, 1):
            if stopped is not None:
                break
            run_pipeline.log_banner(f"capture {index}/{len(pending)}: {capture.rel}")
            try:
                config_path = write_derived_config(capture)
            except OSError as exc:
                capture.status, capture.detail = "failed", f"derived config: {exc}"
                continue
            argv = child_argv(plan, capture, config_path, args)
            run_pipeline.log(f"command           : {run_pipeline.render(argv)}")
            before = {job.get("id") for job in read_jobs(plan.catalog_root)}
            started = time.monotonic()
            try:
                code = runner(argv)
            except KeyboardInterrupt:
                code = 130
            except run_pipeline.Terminated:
                code = run_pipeline.EXIT_TERMINATED
            capture.elapsed = time.monotonic() - started
            record_outcome(plan, capture, code, before)
            if code in STOP_CODES:
                stopped = code
    except KeyboardInterrupt:
        stopped = 130
    except run_pipeline.Terminated:
        stopped = run_pipeline.EXIT_TERMINATED
    finally:
        if restore is not None:
            signal.signal(signal.SIGTERM, restore)
    for capture in pending:
        if capture.status == "pending":
            capture.status, capture.detail = "not started", "the batch was stopped"
    print_summary(plan)
    if stopped is not None:
        return stopped
    return EXIT_FAILED if any(c.status in ("failed", "invalid") for c in plan.captures) else 0


# --------------------------------------------------------------------------
# reporting and entry point
# --------------------------------------------------------------------------

def print_plan(plan: Plan) -> None:
    run_pipeline.log_banner(f"collection {plan.collection}")
    print(f"base config       : {plan.base.path}", flush=True)
    print(f"catalog           : {plan.catalog_root} ({plan.root_id})", flush=True)
    print(f"captures found    : {len(plan.captures)}, in this order", flush=True)
    for capture in plan.captures:
        if capture.problem is not None:
            action, why = "invalid", capture.problem
        elif capture.complete_job is not None:
            action, why = "skip", capture.reason
        else:
            action, why = "run", capture.reason
        print(f"  {action:<8} {capture.rel}  [{capture.capture_id}]  {why}", flush=True)
    for path, why in plan.ignored:
        print(f"  ignored  {path.relative_to(plan.collection)}  {why}", flush=True)


def print_summary(plan: Plan) -> None:
    counts = {}
    for capture in plan.captures:
        counts[capture.status] = counts.get(capture.status, 0) + 1
    run_pipeline.log_banner("collection summary: " + ", ".join(f"{n} {s}" for s, n in counts.items()))
    for capture in plan.captures:
        elapsed = run_pipeline.format_elapsed(capture.elapsed) if capture.elapsed is not None else "-"
        print(f"  {capture.status:<11} {capture.rel}  {elapsed}  {capture.job_id or '-'}  {capture.detail}",
              flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_collection.py",
        description="Reconstruct every capture beneath a collection, one at a time, from one base "
                    "config, skipping captures whose catalog record shows verified completion.")
    parser.add_argument("--config", help="the base INI; [dataset] images_subpath is replaced per capture. "
                        "Defaults as for run_pipeline.py.")
    parser.add_argument("--collection", required=True,
                        help="directory to search, relative to the datasets directory (or absolute inside it)")
    parser.add_argument("--catalog-root", default=os.environ.get("PLANT_CATALOG_ROOT"),
                        help="catalog storage root; the data must lie beneath it")
    parser.add_argument("--catalog-root-id", default=os.environ.get("PLANT_CATALOG_ROOT_ID"),
                        help="logical catalog root ID, kebab-case")
    parser.add_argument("--dry-run", action="store_true",
                        help="discover, validate and decide; write and run nothing")
    parser.add_argument("--allow-full-resolution", action="store_true",
                        help="passed to every run_pipeline.py invocation")
    parser.add_argument("--catalog-hash-artifacts", action="store_true",
                        help="passed to every run_pipeline.py invocation")
    return parser


def main(argv: Optional[Sequence[str]] = None,
         runner: Callable[[Sequence[str]], int] = run_child) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = plan_collection(args)
    except CollectionError as exc:
        print(f"\nCOLLECTION ERROR: {exc}", file=sys.stderr)
        return EXIT_COLLECTION
    print_plan(plan)
    if args.dry_run:
        return EXIT_FAILED if any(c.problem is not None for c in plan.captures) else 0
    return execute(plan, args, runner)


if __name__ == "__main__":
    raise SystemExit(main())
