"""Atomic, portable Plant Reconstruction Catalog v1 producer (stdlib only)."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
import importlib.metadata
import re
import shlex
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


#: The milestone ladder, in order. Skipped stages leave gaps; nothing is synthesized.
MILESTONES = ("config_ok", "prereqs_ok", "masks_composited", "colmap_done", "transforms_ok",
              "pyramid_ok", "trained", "checkpoint_found", "exported", "verified")
PROGRESS_SCHEMA_VERSION = 2
#: Every progress field of the current schema; ``read_progress`` returns them all.
PROGRESS_FIELDS = ("milestones", "furthest_milestone", "active_stage", "stages", "resolved",
                   "dependencies", "gpu", "artifacts", "failure")
#: Runner exit codes that name a failure cause; any other nonzero code is "error".
EXIT_CAUSES = {2: "config", 130: "interrupted", 143: "terminated"}
CANCELLED_CAUSES = ("interrupted", "terminated")
PROBE_TIMEOUT = 10
_COLMAP_VERSION = re.compile(r"COLMAP\s+\d+\.\d+(?:\.\d+)?[^\n]*", re.IGNORECASE)
_NVIDIA_SMI_WSL = Path("/usr/lib/wsl/lib/nvidia-smi")


def read_progress(record):
    """Progress of any job record in the current shape.

    A field the record's schema did not have is None; a job written before
    progress existed reads as schema_version 0.
    """
    progress = record.get("progress") or {}
    return dict(schema_version=progress.get("schema_version", 0),
                **{field: progress.get(field) for field in PROGRESS_FIELDS})


def _probe(argv):
    """Combined output of a bounded command, or None. Never raises."""
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=PROBE_TIMEOUT, check=False)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def colmap_version():
    """COLMAP banner version; 3.12 rejects --version and prints it bare."""
    for argv in (["colmap", "--version"], ["colmap"]):
        probed = _probe(argv)
        match = _COLMAP_VERSION.search(probed[1]) if probed else None
        if match:
            return match.group(0).strip()
    return None


def ffmpeg_version():
    probed = _probe(["ffmpeg", "-version"])
    if not probed or probed[0] != 0 or not probed[1].strip():
        return None
    return probed[1].strip().splitlines()[0]


def gpu_inventory():
    """GPUs the driver reports, or None when nvidia-smi cannot answer."""
    binary = shutil.which("nvidia-smi") or (str(_NVIDIA_SMI_WSL) if _NVIDIA_SMI_WSL.is_file() else None)
    if binary is None:
        return None
    for fields in (("name", "compute_cap", "memory.total", "driver_version"),
                   ("name", "memory.total", "driver_version")):
        probed = _probe([binary, "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"])
        if not probed or probed[0] != 0:
            continue
        devices = []
        for line in probed[1].strip().splitlines():
            values = {key: value.strip() for key, value in zip(fields, line.split(","))
                      if value.strip() not in ("", "N/A", "[N/A]")}
            memory = values.get("memory.total", "")
            devices.append(dict(name=values.get("name"), compute_capability=values.get("compute_cap"),
                memory_total_mib=int(memory) if memory.isdigit() else None,
                driver_version=values.get("driver_version")))
        return dict(devices=devices, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    return None


def measure(path):
    """(size_bytes, file_count) without following links; file_count only for directories."""
    path = Path(path)
    try:
        if path.is_dir() and not path.is_symlink():
            total = count = 0
            for folder, _, files in os.walk(path):
                for name in files:
                    total += os.lstat(os.path.join(folder, name)).st_size
                    count += 1
            return total, count
        return path.stat().st_size, None
    except OSError:
        return None, None


def sha256_file(path):
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_value(repo, *args):
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() or None if result.returncode == 0 else None


class CatalogJob:
    """One invocation; completion never implies review or acceptance."""
    def __init__(self, root, root_id, capture_id, repo, *, title, job_type,
                 command, config, selected_images=None, context_refs=None, pm_refs=None,
                 hash_artifacts=False):
        self.root = Path(root).expanduser().resolve()
        for value in (root_id, capture_id):
            if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value):
                raise ValueError("Catalog root and capture IDs must be kebab-case")
        self.root.mkdir(parents=True, exist_ok=True)
        self.root_id = root_id
        self.capture_id = capture_id
        self.id = "job-" + uuid.uuid4().hex
        self.directory = self.root / "catalog"
        self.context_refs = context_refs or []
        self.pm_refs = pm_refs or []
        self.hash_artifacts = hash_artifacts
        version = None
        project = Path(repo) / "pyproject.toml"
        if project.is_file():
            match = re.search(r'^version\s*=\s*"([^"]+)"', project.read_text(), re.M)
            version = match.group(1) if match else None
        self.record = self.common("job", self.id, title)
        self.record.update(capture_id=capture_id, job_type=job_type,
            started_at=now(), ended_at=None, machine=platform.node() or None,
            execution_status="running", review_status="not_reviewed", superseded_by_job_id=None,
            tool=dict(repository=git_value(repo, "remote", "get-url", "origin"),
                      commit=git_value(repo, "rev-parse", "HEAD"), version=version),
            command=command, config=config, selected_images=selected_images,
            logs=[], input_artifact_ids=[], output_artifact_ids=[], exit_code=None, error_summary=None)
        if git_value(repo, "status", "--porcelain"):
            self.record["diagnostics"].append("Producer checkout has uncommitted changes; commit alone does not identify its working tree.")
        self.record["progress"] = dict(schema_version=PROGRESS_SCHEMA_VERSION, milestones=[],
            furthest_milestone=None, active_stage=None, stages={}, resolved={}, dependencies={},
            gpu=None, artifacts=[], failure=None)
        self._stage_started = None
        self.save()

    def milestone(self, name):
        if name not in MILESTONES:
            raise ValueError(f"Unknown milestone {name!r}; the ladder is plant_catalog.MILESTONES")
        progress = self.record["progress"]
        if not any(item["name"] == name for item in progress["milestones"]):
            progress["milestones"].append(dict(name=name, verified_at=now()))
        progress["furthest_milestone"] = max((item["name"] for item in progress["milestones"]),
                                             key=MILESTONES.index)
        self.save()

    def resolved(self, **values):
        self.record["progress"]["resolved"].update(values)
        self.save()

    def dependencies(self):
        """Tool versions, None when unknown. A probe failure never fails the run."""
        versions = {"python": platform.python_version()}
        for package in ("nerfstudio", "gsplat", "torch"):
            try:
                versions[package] = importlib.metadata.version(package)
            except Exception:  # missing or broken metadata is an unknown version
                versions[package] = None
        for name, probe in (("colmap", colmap_version), ("ffmpeg", ffmpeg_version)):
            try:
                versions[name] = probe()
            except Exception:
                versions[name] = None
        self.record["progress"]["dependencies"] = versions
        self.save()

    def gpu(self):
        try:
            self.record["progress"]["gpu"] = gpu_inventory()
        except Exception:
            self.record["progress"]["gpu"] = None
        self.save()

    def start_stage(self, stage):
        self.record["progress"]["active_stage"] = stage
        self.record["progress"]["stages"][stage] = dict(started_at=now(), ended_at=None,
            elapsed_seconds=None, status="running")
        self._stage_started = time.monotonic()
        self.save()

    def end_stage(self, status):
        progress = self.record["progress"]
        stage = progress["active_stage"]
        if stage is not None:
            progress["stages"][stage].update(ended_at=now(), status=status,
                elapsed_seconds=max(0, time.monotonic() - self._stage_started))
            progress["active_stage"] = None
            self.save()

    def pointer(self, path):
        relative = Path(path).expanduser().resolve().relative_to(self.root).as_posix()
        if relative == ".":
            raise ValueError("Catalog pointer must name a path beneath its root")
        return dict(root_id=self.root_id, relative_path=relative)

    def common(self, kind, identity, title):
        return dict(schema_version=1, kind=kind, id=identity, title=title,
                    context_refs=self.context_refs, pm_refs=self.pm_refs, evidence=[], diagnostics=[])

    def write(self, record):
        path = self.directory / (record["kind"] + "s") / (record["id"] + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.parent.resolve().relative_to(self.root)
        if path.is_symlink():
            raise ValueError("Catalog manifest symlink refused")
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump(record, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def save(self):
        self.write(self.record)

    def capture(self, *, source=None, source_sha256=None, image_count=None, image_inventory=None):
        path = self.directory / "captures" / (self.capture_id + ".json")
        # Existing capture identity is not replaced by a new invocation.
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("id") != self.capture_id or existing.get("kind") != "capture":
                raise ValueError("Existing capture record has a conflicting identity")
            if source_sha256 and existing.get("source_sha256") not in (None, source_sha256):
                raise ValueError("Capture ID already binds a different source checksum")
            return
        record = self.common("capture", self.capture_id, self.capture_id)
        record.update(captured_on=None, specimen_id=None, species=None, operator=None,
            description=None, source=(source if isinstance(source, dict) else self.pointer(source)) if source else None,
            source_sha256=source_sha256, image_count=image_count,
            image_inventory=self.pointer(image_inventory) if image_inventory else None,
            camera_notes=None, preview=None)
        self.write(record)

    def evidence(self, path, *, log=False):
        pointer = self.pointer(path)
        target = self.record["logs"] if log else self.record["evidence"]
        if pointer not in target:
            target.append(pointer)
        self.save()

    def freeze_config(self, path):
        target = self.directory / "evidence" / (self.id + ".ini")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.parent.resolve().relative_to(self.root)
        target.write_bytes(Path(path).read_bytes())
        self.evidence(target)

    def artifact(self, path, *, role, media_type=None, coordinate_frame=None):
        identity = "artifact-" + uuid.uuid4().hex
        record = self.common("artifact", identity, Path(path).name)
        location = self.pointer(path)
        # Large outputs are linked in place; hashing is opt-in and files only.
        digest = sha256_file(path) if self.hash_artifacts and Path(path).is_file() else None
        record.update(capture_id=self.capture_id, producing_job_id=self.id,
            media_type=media_type, role=role, location=location, sha256=digest,
            coordinate_frame=coordinate_frame, calibration_ref=None, review_status="not_reviewed")
        self.write(record)
        size, files = measure(path)
        self.record["output_artifact_ids"].append(identity)
        self.record["progress"]["artifacts"].append(dict(artifact_id=identity, role=role,
            relative_path=location["relative_path"], size_bytes=size, file_count=files))
        self.save()

    def finish(self, code, error=None):
        progress = self.record["progress"]
        if code == 0:
            status = "completed"
        else:
            cause = EXIT_CAUSES.get(code, "error")
            status = "cancelled" if cause in CANCELLED_CAUSES else "failed"
            progress["failure"] = dict(cause=cause, stage=progress["active_stage"],
                                       after_milestone=progress["furthest_milestone"])
        self.end_stage(status)
        self.record.update(ended_at=now(), execution_status=status, exit_code=code, error_summary=error)
        self.save()
