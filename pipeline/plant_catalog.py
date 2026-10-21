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
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_value(repo, *args):
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() or None if result.returncode == 0 else None


class CatalogJob:
    """One invocation; completion never implies review or acceptance."""
    def __init__(self, root, root_id, capture_id, repo, *, title, job_type,
                 command, config, selected_images=None, context_refs=None, pm_refs=None):
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
        self.record["progress"] = dict(schema_version=1, milestones=[], furthest_milestone=None,
            active_stage=None, stages={}, resolved={}, dependencies={})
        self._stage_started = None
        self.save()

    def milestone(self, name):
        progress = self.record["progress"]
        if not any(item["name"] == name for item in progress["milestones"]):
            progress["milestones"].append(dict(name=name, verified_at=now()))
        progress["furthest_milestone"] = name
        self.save()

    def resolved(self, **values):
        self.record["progress"]["resolved"].update(values)
        self.save()

    def dependencies(self):
        versions = {"python": platform.python_version()}
        for package in ("nerfstudio", "gsplat", "torch"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        for name, command in (("colmap", ["colmap", "--version"]), ("ffmpeg", ["ffmpeg", "-version"])):
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
                lines = (result.stdout + result.stderr).strip().splitlines()
                versions[name] = lines[0] if lines and result.returncode == 0 else None
            except (OSError, subprocess.TimeoutExpired):
                versions[name] = None
        self.record["progress"]["dependencies"] = versions
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
        record.update(capture_id=self.capture_id, producing_job_id=self.id,
            media_type=media_type, role=role, location=self.pointer(path), sha256=None,
            coordinate_frame=coordinate_frame, calibration_ref=None, review_status="not_reviewed")
        # Large outputs are linked in place; hashing is explicitly unknown.
        self.write(record)
        self.record["output_artifact_ids"].append(identity)
        self.save()

    def finish(self, code, error=None):
        self.end_stage("completed" if code == 0 else "cancelled" if code == 130 else "failed")
        self.record.update(ended_at=now(), execution_status="completed" if code == 0 else
            "cancelled" if code == 130 else "failed", exit_code=code, error_summary=error)
        self.save()
