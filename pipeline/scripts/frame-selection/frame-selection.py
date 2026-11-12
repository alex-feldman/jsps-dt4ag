#!/usr/bin/env python3
"""Choose which frames of an existing capture to keep, and build a new capture from that choice.

Two subcommands, and neither launches a pipeline stage or touches the source images.

``preview``
    Inventories ``{capture}/images/`` (top-level image files, sorted by name), assigns each file
    to a clip, makes thumbnails with ffmpeg and writes an offline ``index.html`` in a NEW review
    folder. On the page a person excludes frame ranges (several per clip, also in the middle),
    sets an every-Nth sampling with an offset per clip, and downloads a selection JSON. Optional
    annotations from a prior camera solve (camera-review's ``camera-checks.json``) are shown as
    labeled context; they never decide the selection.

``build``
    Validates a selection JSON against the source capture as it is now (same image listing, same
    rules, same kept list) and writes a NEW capture: ``images/`` as relative symlinks to the
    source (or copies with ``--materialize``), plus ``provenance/selection.json`` (the input,
    verbatim), ``provenance/frame-selection.json`` and ``provenance/frames.csv``. ``--dry-run``
    prints the plan and writes nothing.

Sampling is applied to the ORIGINAL base sequence of each clip (its files in name order) before
exclusions: position ``p`` (0-based) is sampled when ``p >= offset`` and ``(p - offset) % every
== 0``. A frame is kept when it is sampled and not inside any excluded range. Excluding frames
therefore never shifts the sampling phase.

What it does not do: it does not measure overlap, does not copy masks, and does not prove that
the bytes of an image are unchanged (the inventory hash covers names, sizes and modification
times). A changed selection is a different image set and needs its own camera solve.

Standard library only; ``preview`` needs ``ffmpeg`` on ``PATH`` or under ``--ffmpeg-prefix``.

Usage::

    python3 pipeline/scripts/frame-selection/frame-selection.py preview \\
        --capture {data_root}/datasets/{capture} \\
        --out {data_root}/evals/{subject}/frame-selection-preview-{yyyymmdd-hhmm} \\
        [--clips clips.csv] [--annotations camera-checks.json] [--title ...]

    python3 pipeline/scripts/frame-selection/frame-selection.py build \\
        --selection {downloaded selection JSON} \\
        --capture {data_root}/datasets/{new-capture-id} [--source {capture}] [--materialize] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

TOOL = "frame-selection.py"
FORMAT = "dt4ag-frame-selection/1"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
TIME_COLUMNS = ("source_seconds_approx", "output_grid_seconds")
THUMB_EDGE = 320          # longest edge of a grid thumbnail, pixels
LARGE_EDGE = 1280         # longest edge of the enlarged view, pixels
MIN_KEPT_FRAMES = 3
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
STATUS_BUILT = "selection_built_pending_review"

INVENTORY_COLUMNS = ["index", "filename", "clip", "clip_index", "seconds", "width", "height",
                     "size_bytes", "mtime_ns"]

INVENTORY_HASH_NOTE = (
    "inventory_sha256 is the SHA-256 of one line per image file in images/, in name order: "
    "filename, size in bytes and modification time in nanoseconds, tab-separated. It detects added, "
    "removed, renamed, resized or re-saved files; it does not prove the bytes are unchanged."
)

LIMITS = (
    "A selection does not measure overlap between frames. A changed selection is a different image "
    "set and needs its own camera solve. Annotations describe only the named prior solve of the image "
    "set it was run on. Masks are not copied."
)


class ToolError(Exception):
    """A refusal or failure the operator should read and act on."""


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    """Streamed SHA-256 (``hashlib.file_digest`` needs Python 3.11)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def check_filename(name: object) -> str:
    """A bare image filename: no path separators, no ``..``, no control characters."""
    if not isinstance(name, str) or not name:
        raise ToolError(f"filename {name!r} is not a non-empty string")
    if "/" in name or "\\" in name or ".." in name or name in (".",):
        raise ToolError(f"filename {name!r} has a path separator or '..'; only bare filenames are allowed")
    if any(ord(c) < 32 for c in name):
        raise ToolError(f"filename {name!r} has a control character")
    return name


def read_png_size(path: Path) -> Optional[Tuple[int, int]]:
    """Width and height from a PNG header, or None for anything that is not a readable PNG."""
    try:
        with path.open("rb") as handle:
            head = handle.read(24)
    except OSError:
        return None
    if len(head) < 24 or head[:8] != PNG_SIGNATURE or head[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", head[16:24])


def tool_version() -> Optional[str]:
    """``git describe`` of the checkout holding this script, or None outside a checkout."""
    try:
        result = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "describe",
                                 "--tags", "--always", "--dirty"],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    text = result.stdout.decode(errors="replace").strip()
    return text if result.returncode == 0 and text else None


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

def inventory_images(capture: Path) -> Tuple[List[Dict], List[str]]:
    """Top-level image files of ``{capture}/images``, sorted by name, with size and mtime.

    Returns the entries and notes about what was skipped (subdirectories, other files). A link
    that does not resolve to a file is refused, not skipped: it is an image the capture claims.
    """
    images = capture / "images"
    if not images.is_dir():
        raise ToolError(f"{capture} has no images/ directory; a capture is a directory holding images/")
    entries, notes, subdirs, others = [], [], [], []
    for path in sorted(images.iterdir(), key=lambda p: p.name):
        if path.is_dir():
            subdirs.append(path.name)
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            others.append(path.name)
            continue
        if not path.is_file():
            raise ToolError(f"{path} is a link that does not resolve to a file")
        check_filename(path.name)
        stat = path.stat()
        entries.append({"filename": path.name, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    if subdirs:
        notes.append(f"ignored {len(subdirs)} subdirectories of images/ (only top-level files are "
                     f"inventoried): {', '.join(subdirs[:5])}")
    if others:
        notes.append(f"ignored {len(others)} non-image files in images/: {', '.join(others[:5])}")
    if not entries:
        raise ToolError(f"{images} holds no top-level image files ({', '.join(sorted(IMAGE_SUFFIXES))})")
    return entries, notes


def inventory_sha256(entries: Sequence[Dict]) -> str:
    """See INVENTORY_HASH_NOTE."""
    text = "".join(f"{e['filename']}\t{e['size_bytes']}\t{e['mtime_ns']}\n" for e in entries)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_seconds(text: str, where: str) -> Optional[float]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        raise ToolError(f"{where}: seconds {text!r} is not a number")
    if value != value or value in (float("inf"), float("-inf")):
        raise ToolError(f"{where}: seconds {text!r} is not finite")
    return value


def assign_clips(capture: Path, names: Sequence[str], clips_csv: Optional[Path]) -> Tuple[Dict, Dict]:
    """filename -> (clip, seconds or None), and a record of where the assignment came from.

    From ``--clips`` (columns ``filename,clip`` and optional ``seconds``) when given; else one clip
    from ``provenance/frames.csv`` (seconds from ``source_seconds_approx`` or
    ``output_grid_seconds``); else one clip named after the capture with no times.
    """
    inventory = set(names)
    if clips_csv is not None:
        if not clips_csv.is_file():
            raise ToolError(f"clips CSV not found: {clips_csv}")
        with clips_csv.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = [f.strip() for f in (reader.fieldnames or [])]
            if "filename" not in fields or "clip" not in fields:
                raise ToolError(f"{clips_csv} needs the columns filename and clip (and optionally seconds); "
                                f"it has {fields}")
            rows = [{(k or "").strip(): (v or "") for k, v in row.items()} for row in reader]
        assigned: Dict[str, Tuple[str, Optional[float]]] = {}
        duplicates, unknown = [], []
        for number, row in enumerate(rows, start=2):
            name, clip = row["filename"].strip(), row["clip"].strip()
            if not clip:
                raise ToolError(f"{clips_csv} line {number}: empty clip name for {name!r}")
            if name in assigned:
                duplicates.append(name)
            elif name not in inventory:
                unknown.append(name)
            else:
                assigned[name] = (clip, parse_seconds(row.get("seconds", ""), f"{clips_csv} line {number}"))
        missing = [n for n in names if n not in assigned]
        problems = []
        if duplicates:
            problems.append(f"{len(duplicates)} duplicated filenames ({', '.join(duplicates[:5])})")
        if unknown:
            problems.append(f"{len(unknown)} filenames not in images/ ({', '.join(unknown[:5])})")
        if missing:
            problems.append(f"{len(missing)} images/ files it does not list ({', '.join(missing[:5])})")
        if problems:
            raise ToolError(f"REFUSED: the clips CSV {clips_csv} must list every image exactly once: "
                            + "; ".join(problems))
        return assigned, {"source": "clips_csv", "path": str(clips_csv), "sha256": sha256_of(clips_csv)}

    clip_name = capture.name
    frames_csv = capture / "provenance" / "frames.csv"
    if frames_csv.is_file():
        with frames_csv.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        column = next((c for c in TIME_COLUMNS if rows and c in rows[0]), None)
        times = {}
        if column:
            for number, row in enumerate(rows, start=2):
                times[row.get("filename", "")] = parse_seconds(row.get(column, ""), f"{frames_csv} line {number}")
        unmatched = sum(1 for n in names if n not in times)
        record = {"source": "provenance_frames_csv", "path": str(frames_csv), "sha256": sha256_of(frames_csv),
                  "time_column": column, "images_without_a_time": unmatched}
        return {n: (clip_name, times.get(n)) for n in names}, record
    return {n: (clip_name, None) for n in names}, {"source": "single_clip_no_times"}


def build_inventory(capture: Path, clips_csv: Optional[Path]) -> Dict:
    entries, notes = inventory_images(capture)
    names = [e["filename"] for e in entries]
    assigned, clip_record = assign_clips(capture, names, clips_csv)
    order: List[str] = []
    members: Dict[str, List[str]] = {}
    for name in names:
        clip = assigned[name][0]
        if clip not in members:
            order.append(clip)
            members[clip] = []
        members[clip].append(name)
    position = {n: i for clip in order for i, n in enumerate(members[clip])}
    images = capture / "images"
    rows = []
    for index, entry in enumerate(entries):
        name = entry["filename"]
        size = read_png_size(images / name)
        rows.append({
            "index": index, "filename": name, "clip": assigned[name][0], "clip_index": position[name],
            "seconds": assigned[name][1], "width": size[0] if size else None,
            "height": size[1] if size else None, "size_bytes": entry["size_bytes"], "mtime_ns": entry["mtime_ns"],
        })
    return {"rows": rows, "clips": [{"name": c, "frames": members[c]} for c in order],
            "inventory_sha256": inventory_sha256(entries), "notes": notes, "clip_assignment": clip_record}


def inventory_csv_text(rows: Sequence[Dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=INVENTORY_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("" if row[k] is None else row[k]) for k in INVENTORY_COLUMNS})
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Selection rules (mirrored by the page's JavaScript; keep the two identical)
# ---------------------------------------------------------------------------

def resolve_clip(frames: Sequence[str], every: int, offset: int,
                 exclude: Sequence[Sequence[str]]) -> List[bool]:
    """Which base frames of one clip are kept: sampled on the original sequence, then not excluded."""
    position = {name: i for i, name in enumerate(frames)}
    excluded = [False] * len(frames)
    for start, end in exclude:
        for i in range(position[start], position[end] + 1):
            excluded[i] = True
    return [i >= offset and (i - offset) % every == 0 and not excluded[i] for i in range(len(frames))]


def kept_ranges(frames: Sequence[str], kept: Sequence[bool], exclude: Sequence[Sequence[str]]) -> List[Dict]:
    """The stretches of a clip outside every excluded range, with how many frames each keeps."""
    position = {name: i for i, name in enumerate(frames)}
    excluded = [False] * len(frames)
    for start, end in exclude:
        for i in range(position[start], position[end] + 1):
            excluded[i] = True
    ranges, i = [], 0
    while i < len(frames):
        if excluded[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(frames) and not excluded[j + 1]:
            j += 1
        ranges.append({"start": frames[i], "end": frames[j], "base_frames": j - i + 1,
                       "kept": sum(1 for k in range(i, j + 1) if kept[k])})
        i = j + 1
    return ranges


def _whole_number(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError(f"{what} must be a whole number, got {value!r}")
    return value


def validate_selection(selection: Dict, entries: Sequence[Dict]) -> Dict:
    """Check a selection against the source's current inventory and recompute what it keeps.

    Refuses rather than repairs: every mismatch is an error naming what to fix.
    """
    if not isinstance(selection, dict):
        raise ToolError("the selection is not a JSON object")
    if selection.get("format") != FORMAT:
        raise ToolError(f"REFUSED: selection format is {selection.get('format')!r}, this tool reads {FORMAT!r}")
    source = selection.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("inventory_sha256"), str):
        raise ToolError("the selection has no source.inventory_sha256")
    actual = inventory_sha256(entries)
    if source["inventory_sha256"] != actual:
        raise ToolError(f"REFUSED: the source images changed since the preview: inventory hash is now {actual}, "
                        f"the selection was made on {source['inventory_sha256']}. Make a new preview.")

    names = [e["filename"] for e in entries]
    clips = selection.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ToolError("the selection has no clips")
    seen: Dict[str, str] = {}
    resolved = []
    for clip in clips:
        if not isinstance(clip, dict) or not isinstance(clip.get("name"), str) or not clip["name"]:
            raise ToolError("every clip needs a non-empty name")
        label = clip["name"]
        frames = clip.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ToolError(f"clip {label!r} has no frames list")
        for name in frames:
            check_filename(name)
            if name in seen:
                raise ToolError(f"REFUSED: {name!r} is listed twice (clips {seen[name]!r} and {label!r})")
            seen[name] = label
        if list(frames) != sorted(frames):
            raise ToolError(f"clip {label!r}: frames must be in filename order (the base order)")
        every = _whole_number(clip.get("every", 1), f"clip {label!r} every")
        offset = _whole_number(clip.get("offset", 0), f"clip {label!r} offset")
        if every < 1:
            raise ToolError(f"clip {label!r}: every must be 1 or more, got {every}")
        if not 0 <= offset < every:
            raise ToolError(f"clip {label!r}: offset must be from 0 to {every - 1}, got {offset}")
        members = set(frames)
        exclude = clip.get("exclude", [])
        if not isinstance(exclude, list):
            raise ToolError(f"clip {label!r}: exclude must be a list of [start, end] filename pairs")
        position = {n: i for i, n in enumerate(frames)}
        for pair in exclude:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ToolError(f"clip {label!r}: exclude range {pair!r} is not a [start, end] pair")
            for name in pair:
                check_filename(name)
                if name not in members:
                    raise ToolError(f"REFUSED: clip {label!r}: exclude range {pair!r} names {name!r}, "
                                    f"which is not a frame of that clip")
            if position[pair[0]] > position[pair[1]]:
                raise ToolError(f"REFUSED: clip {label!r}: exclude range {pair!r} starts after it ends")
        seconds = clip.get("seconds")
        if seconds is not None:
            if not isinstance(seconds, list) or len(seconds) != len(frames):
                raise ToolError(f"clip {label!r}: seconds must be a list as long as frames")
            for value in seconds:
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                    raise ToolError(f"clip {label!r}: seconds must be numbers or null, got {value!r}")
        kept = resolve_clip(frames, every, offset, exclude)
        resolved.append({"name": label, "frames": list(frames), "every": every, "offset": offset,
                         "exclude": [list(p) for p in exclude], "kept_flags": kept,
                         "seconds": list(seconds) if seconds is not None else [None] * len(frames),
                         "ranges": kept_ranges(frames, kept, exclude)})
    if set(seen) != set(names):
        extra = sorted(set(seen) - set(names))
        absent = sorted(set(names) - set(seen))
        raise ToolError(f"REFUSED: the clips do not cover the source images exactly: "
                        f"{len(extra)} listed but not in images/ ({', '.join(extra[:5])}), "
                        f"{len(absent)} in images/ but not listed ({', '.join(absent[:5])})")

    kept_set = {n for c in resolved for n, k in zip(c["frames"], c["kept_flags"]) if k}
    recomputed = [n for n in names if n in kept_set]
    recorded = selection.get("kept")
    if not isinstance(recorded, list):
        raise ToolError("the selection has no kept list")
    for name in recorded:
        check_filename(name)
    if len(set(recorded)) != len(recorded):
        raise ToolError("REFUSED: the kept list has duplicates")
    if recorded != recomputed:
        only_recorded = [n for n in recorded if n not in kept_set]
        only_rules = [n for n in recomputed if n not in set(recorded)]
        raise ToolError(f"REFUSED: the kept list does not follow from the rules: {len(only_recorded)} listed "
                        f"but not kept by the rules ({', '.join(only_recorded[:5])}), {len(only_rules)} kept "
                        f"by the rules but not listed ({', '.join(only_rules[:5])})"
                        + ("" if only_recorded or only_rules else ", or the order differs from filename order"))
    if len(recomputed) < MIN_KEPT_FRAMES:
        raise ToolError(f"REFUSED: the selection keeps {len(recomputed)} frames; need at least {MIN_KEPT_FRAMES}")
    return {"clips": resolved, "kept": recomputed, "total": len(names)}


# ---------------------------------------------------------------------------
# Annotations (optional context from a prior camera solve)
# ---------------------------------------------------------------------------

ANNOTATION_FIELDS = ("missing", "unexpected", "largest_step_from", "largest_step_to",
                     "largest_step_ratio_to_median", "camera_path_warnings",
                     # extended fields some camera reviews have written; read when present
                     "low_point_frames", "flagged_steps", "large_turns", "out_and_back")
ANNOTATION_CONTEXT_FIELDS = ("label", "registered", "expected", "sparse_models", "reviewed_model", "points",
                             "mean_reprojection_error_px", "mean_track_length", "cameras", "time_axis",
                             "registered_span", "median_step_between_adjacent_registered_cameras",
                             "largest_step_between_adjacent_registered_cameras", "largest_step_time_interval",
                             "camera_step_warning_ratio", "limits")


def _names(value) -> List[str]:
    return [v for v in value if isinstance(v, str)] if isinstance(value, list) else []


def load_annotations(path: Path, inventory: Sequence[str]) -> Dict:
    """Per-frame tags from camera-review's ``camera-checks.json``. Tolerant: unknown fields and absent
    known fields are reported, and names not in the inventory are reported, never fatal."""
    if not path.is_file():
        raise ToolError(f"annotations file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError) as err:
        raise ToolError(f"annotations file {path} is not JSON: {err}")
    if not isinstance(data, dict):
        raise ToolError(f"annotations file {path} is not a JSON object")
    tags: Dict[str, List[str]] = {}

    def tag(name, text):
        if isinstance(name, str) and name:
            tags.setdefault(name, []).append(text)

    for name in _names(data.get("missing")):
        tag(name, "not registered")
    if data.get("camera_path_warnings") and isinstance(data.get("largest_step_to"), str):
        ratio = data.get("largest_step_ratio_to_median")
        ratio_text = f" ({ratio:.1f}x median)" if isinstance(ratio, (int, float)) else ""
        tag(data.get("largest_step_from"), f"largest camera step leaves here{ratio_text}")
        tag(data.get("largest_step_to"), f"largest camera step arrives here{ratio_text}")
    for item in data.get("low_point_frames") or []:
        if isinstance(item, dict):
            tag(item.get("frame"), f"sees only {item.get('points')} points")
    for item in data.get("flagged_steps") or []:
        if isinstance(item, dict):
            tag(item.get("to"), "camera jumps here")
    for item in data.get("large_turns") or []:
        if isinstance(item, dict):
            tag(item.get("to"), "camera turns sharply here")
    for name in _names(data.get("out_and_back")):
        tag(name, "out and back")

    known = set(inventory)
    not_in_inventory = sorted(n for n in tags if n not in known)
    unexpected_in_inventory = [n for n in _names(data.get("unexpected")) if n in known]
    return {
        "path": str(path),
        "sha256": sha256_of(path),
        "label": data.get("label") if isinstance(data.get("label"), str) else "",
        "registered": data.get("registered"),
        "expected": data.get("expected"),
        "warnings": _names(data.get("camera_path_warnings")),
        "tags": {n: t for n, t in tags.items() if n in known},
        "fields_used": [f for f in ANNOTATION_FIELDS if f in data],
        "fields_absent": [f for f in ANNOTATION_FIELDS if f not in data],
        "fields_ignored": sorted(k for k in data if k not in ANNOTATION_FIELDS and k not in ANNOTATION_CONTEXT_FIELDS),
        "names_not_in_inventory": not_in_inventory,
        "unexpected_names_in_inventory": unexpected_in_inventory,
        "note": "Optional context from a prior camera solve of the image set it was run on. "
                "It does not decide the selection.",
    }


# ---------------------------------------------------------------------------
# ffmpeg
# ---------------------------------------------------------------------------

def find_tool(name: str, prefix: Optional[Path]) -> str:
    if prefix is not None:
        candidate = prefix / name
        if not candidate.is_file():
            raise ToolError(f"{name} not found at {candidate}")
        return str(candidate)
    found = shutil.which(name)
    if not found:
        raise ToolError(f"{name} is not on PATH; pass --ffmpeg-prefix with the directory that holds ffmpeg "
                        f"(QUICKSTART section 3 installs it)")
    return found


def ffmpeg_version(ffmpeg: str) -> str:
    result = subprocess.run([ffmpeg, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = result.stdout.decode(errors="replace") if result.returncode == 0 else ""
    return out.splitlines()[0].strip() if out else "unknown"


def thumb_command(ffmpeg: str, source: Path, thumb: Path, large: Path) -> List[str]:
    """One decode of one image, two JPEG outputs: a grid thumbnail and a larger view."""
    fit = "force_original_aspect_ratio=decrease"
    graph = (f"[0:v]split=2[a][b];[a]scale=w={THUMB_EDGE}:h={THUMB_EDGE}:{fit}[t];"
             f"[b]scale=w={LARGE_EDGE}:h={LARGE_EDGE}:{fit}[l]")
    return [ffmpeg, "-n", "-hide_banner", "-loglevel", "error", "-threads", "2", "-i", str(source),
            "-filter_complex", graph,
            "-map", "[t]", "-frames:v", "1", "-q:v", "5", str(thumb),
            "-map", "[l]", "-frames:v", "1", "-q:v", "3", str(large)]


# ---------------------------------------------------------------------------
# preview
# ---------------------------------------------------------------------------

def write_index(out: Path, data: Dict) -> Path:
    """Write ``index.html`` from the template beside this script; all content comes from ``data``."""
    template = (Path(__file__).resolve().parent / "frame-selection.template.html").read_text(encoding="utf-8")
    # Inside a script element "</" can close it early and "<!--<script" can keep it open; escape both.
    payload = json.dumps(data).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    page = out / "index.html"
    page.write_text(template.replace("__DATA__", payload), encoding="utf-8")
    return page


def guess_capture_root(path: Path) -> str:
    """``{data_root}/datasets`` when ``path`` sits under ``{data_root}/evals`` or ``datasets``, else empty."""
    for parent in (path, *path.parents):
        if parent.name in ("evals", "datasets"):
            return str(parent.parent / "datasets")
    return ""


def cmd_preview(args: argparse.Namespace) -> int:
    capture = Path(args.capture).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    clips_csv = Path(args.clips).expanduser().resolve() if args.clips else None
    if out.exists():
        raise ToolError(f"REFUSED: {out} already exists. Choose a new review folder; a preview is never updated in place.")
    try:
        out.relative_to(capture)
        raise ToolError(f"REFUSED: {out} is inside the capture; the preview must not write into the source")
    except ValueError:
        pass

    # Everything that can be refused is checked before anything is created.
    inventory = build_inventory(capture, clips_csv)
    names = [r["filename"] for r in inventory["rows"]]
    annotations = None
    if args.annotations:
        annotations = load_annotations(Path(args.annotations).expanduser().resolve(), names)
    prefix = Path(args.ffmpeg_prefix).expanduser() if args.ffmpeg_prefix else None
    ffmpeg = find_tool("ffmpeg", prefix)
    title = args.title or f"Frame selection: {capture.name}"

    out.mkdir(parents=True, exist_ok=False)
    (out / "thumbs").mkdir()
    (out / "large").mkdir()
    print(f"preview dir: {out}", flush=True)
    csv_text = inventory_csv_text(inventory["rows"])
    (out / "inventory.csv").write_text(csv_text, encoding="utf-8", newline="")
    csv_sha = hashlib.sha256(csv_text.encode("utf-8")).hexdigest()
    (out / "inventory.sha256").write_text(f"{csv_sha}  inventory.csv\n", encoding="utf-8")

    images = capture / "images"
    print(f"making {len(names)} thumbnails with ffmpeg ...", flush=True)
    with (out / "ffmpeg-thumbs.log").open("w", encoding="utf-8") as log:
        for row in inventory["rows"]:
            stem = f"{row['index']:06d}.jpg"
            cmd = thumb_command(ffmpeg, images / row["filename"], out / "thumbs" / stem, out / "large" / stem)
            log.write("$ " + " ".join(cmd) + "\n")
            log.flush()
            result = subprocess.run(cmd, stdout=log, stderr=log)
            if result.returncode != 0:
                raise ToolError(f"ffmpeg failed on {row['filename']}; see {out / 'ffmpeg-thumbs.log'}. "
                                f"The partial preview is kept; make a new one after fixing the cause.")
            if (row["index"] + 1) % 50 == 0:
                print(f"  {row['index'] + 1} of {len(names)}", flush=True)

    rows_by_name = {r["filename"]: r for r in inventory["rows"]}
    page_clips = []
    for clip in inventory["clips"]:
        page_clips.append({"name": clip["name"], "frames": [{
            "f": n, "i": rows_by_name[n]["index"], "s": rows_by_name[n]["seconds"],
            "t": f"thumbs/{rows_by_name[n]['index']:06d}.jpg", "l": f"large/{rows_by_name[n]['index']:06d}.jpg",
        } for n in clip["frames"]]})
    record = {
        "tool": TOOL, "version": tool_version(), "created": now_iso(), "title": title,
        "capture": str(capture), "images": str(images), "frame_count": len(names),
        "inventory_sha256": inventory["inventory_sha256"], "inventory_hash_note": INVENTORY_HASH_NOTE,
        "inventory_csv_sha256": csv_sha, "inventory_notes": inventory["notes"],
        "clip_assignment": inventory["clip_assignment"],
        "clips": [{"name": c["name"], "frames": len(c["frames"])} for c in inventory["clips"]],
        "annotations": {k: v for k, v in annotations.items() if k != "tags"} if annotations else None,
        "thumbnails": {"thumb_edge_px": THUMB_EDGE, "large_edge_px": LARGE_EDGE, "ffmpeg": ffmpeg_version(ffmpeg),
                       "note": "for choosing only; not pipeline input"},
        "limits": LIMITS,
    }
    (out / "preview.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    capture_root = guess_capture_root(capture.parent) or str(capture.parent)
    index = write_index(out, {
        "format": FORMAT, "title": title, "capture": str(capture), "capture_name": capture.name,
        "inventory_sha256": inventory["inventory_sha256"], "inventory_csv_sha256": csv_sha,
        "preview_dir": str(out), "created": record["created"], "clips": page_clips,
        "clip_assignment": inventory["clip_assignment"], "inventory_notes": inventory["notes"],
        "annotations": annotations, "min_kept": MIN_KEPT_FRAMES, "limits": LIMITS,
        "build_command": (f"python3 {Path(__file__).resolve()} build \\\n    --selection {{downloaded selection JSON}} \\\n"
                          f"    --capture {capture_root}/{capture.name}-sel-v1 --dry-run"),
    })

    print(f"frames: {len(names)} in {len(page_clips)} clip(s): "
          + ", ".join(f"{c['name']} {len(c['frames'])}" for c in page_clips))
    for note in inventory["notes"]:
        print(f"note: {note}")
    if annotations:
        print(f"annotations: {annotations['path']} ({len(annotations['tags'])} frames tagged; "
              f"fields absent: {', '.join(annotations['fields_absent']) or 'none'}; "
              f"ignored: {', '.join(annotations['fields_ignored']) or 'none'}; "
              f"{len(annotations['names_not_in_inventory'])} names not in this inventory)")
    print(f"inventory sha256: {inventory['inventory_sha256']}")
    print(f"open this, mark exclusions, download the selection JSON: {index}")
    print("The selection is your decision; the page does not check overlap.")
    return 0


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def _existing_parent(path: Path) -> Path:
    for parent in (path, *path.parents):
        if parent.exists():
            return parent
    return Path("/")


def cmd_build(args: argparse.Namespace) -> int:
    selection_path = Path(args.selection).expanduser().resolve()
    capture = Path(args.capture).expanduser().resolve()
    if not selection_path.is_file():
        raise ToolError(f"selection not found: {selection_path}")
    raw = selection_path.read_bytes()
    try:
        selection = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as err:
        raise ToolError(f"{selection_path} is not JSON: {err}")
    if not isinstance(selection, dict):
        raise ToolError(f"{selection_path} is not a JSON object")
    if selection.get("format") != FORMAT:
        raise ToolError(f"REFUSED: selection format is {selection.get('format')!r}, this tool reads {FORMAT!r}")
    recorded_source = (selection.get("source") or {}).get("capture") if isinstance(selection.get("source"), dict) else None
    if args.source:
        source = Path(args.source).expanduser().resolve()
    elif isinstance(recorded_source, str) and recorded_source:
        source = Path(recorded_source)
    else:
        raise ToolError("the selection names no source.capture; pass --source")
    if not source.is_dir():
        raise ToolError(f"source capture not found: {source}")
    if capture.exists() or capture.is_symlink():
        raise ToolError(f"REFUSED: {capture} already exists. Preserve it and choose a new capture name.")
    try:
        capture.relative_to(source)
        raise ToolError(f"REFUSED: {capture} is inside the source capture")
    except ValueError:
        pass

    entries, _notes = inventory_images(source)
    result = validate_selection(selection, entries)
    sizes = {e["filename"]: e["size_bytes"] for e in entries}
    kept = result["kept"]
    kept_bytes = sum(sizes[n] for n in kept)
    mode = "copy" if args.materialize else "link"
    masks = sorted(p.name for p in source.iterdir() if p.is_dir() and (p.name == "masks" or p.name.startswith("masks_")))

    print(f"source   : {source}  ({result['total']} images, inventory {selection['source']['inventory_sha256'][:12]}...)")
    for clip in result["clips"]:
        sampled = sum(1 for i in range(len(clip["frames"]))
                      if i >= clip["offset"] and (i - clip["offset"]) % clip["every"] == 0)
        print(f"clip {clip['name']}: {len(clip['frames'])} base, every {clip['every']} from offset {clip['offset']} "
              f"= {sampled} sampled, {len(clip['exclude'])} excluded range(s), {sum(clip['kept_flags'])} kept")
        for r in clip["ranges"]:
            print(f"    keep range {r['start']} .. {r['end']}: {r['base_frames']} base, {r['kept']} kept")
    print(f"kept     : {len(kept)} of {result['total']} frames")
    if mode == "copy":
        print(f"mode     : copy (--materialize), about {kept_bytes} bytes ({kept_bytes / 2**30:.2f} GiB)")
    else:
        print("mode     : relative symlinks to the source images/ (no image copied)")
    print(f"capture  : {capture}")
    if masks:
        print(f"note     : the source has {', '.join(masks)}; masks are NOT copied or linked by this tool")
    if mode == "copy":
        free = shutil.disk_usage(_existing_parent(capture.parent)).free
        if free < kept_bytes:
            raise ToolError(f"REFUSED: {kept_bytes} bytes to copy but only {free} free at the destination")
    if args.dry_run:
        print("dry run: nothing written.")
        return 0

    src_images = source / "images"
    capture.mkdir(parents=True, exist_ok=False)
    try:
        (capture / "images").mkdir()
        (capture / "provenance").mkdir()
        dest = capture / "images"
        resolved = src_images.resolve()
        for name in kept:
            if mode == "copy":
                shutil.copy2(src_images / name, dest / name)
                if (dest / name).stat().st_size != sizes[name]:
                    raise ToolError(f"the copy of {name} has the wrong size")
                continue
            try:
                # Relative links, so moving the source and the new capture together keeps them valid.
                (dest / name).symlink_to(os.path.relpath(resolved / name, dest))
            except ValueError as err:     # Windows: different drives
                raise ToolError(f"cannot make a relative link between these locations ({err}); put the new "
                                f"capture on the same drive as the source, or use --materialize.")
            except (OSError, NotImplementedError) as err:
                raise ToolError(f"cannot make symlinks here ({err}); this tool does not silently copy. "
                                f"Use a filesystem that supports links, or pass --materialize to copy.")
        broken = [n for n in kept if not (dest / n).is_file()]
        if broken:
            raise ToolError(f"{len(broken)} images do not resolve to a file ({', '.join(broken[:5])})")
    except BaseException:
        shutil.rmtree(capture, ignore_errors=True)
        raise

    prov = capture / "provenance"
    (prov / "selection.json").write_bytes(raw)
    by_name = {}
    for clip in result["clips"]:
        for i, (name, sec) in enumerate(zip(clip["frames"], clip["seconds"])):
            by_name[name] = (clip["name"], i, sec)
    with_times = all(by_name[n][2] is not None for n in kept)
    columns = ["filename", "clip", "clip_base_index", "source_filename"] + (["source_seconds_approx"] if with_times else [])
    with (prov / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for name in kept:
            clip, index, sec = by_name[name]
            writer.writerow([name, clip, index, name] + ([float(sec)] if with_times else []))
    record = {
        "tool": TOOL, "version": tool_version(), "created": now_iso(),
        "kind": "frame selection of an existing capture",
        "source_capture": str(source), "source_images": str(src_images),
        "inventory_sha256": selection["source"]["inventory_sha256"], "inventory_hash_note": INVENTORY_HASH_NOTE,
        "selection_file": str(selection_path), "selection_sha256": hashlib.sha256(raw).hexdigest(),
        "mode": mode,
        "link_style": "relative" if mode == "link" else None,
        "link_target_dir": str(resolved) if mode == "link" else None,
        "copied_bytes": kept_bytes if mode == "copy" else 0,
        "source_frames": result["total"], "kept_frames": len(kept),
        "clips": [{"name": c["name"], "base_frames": len(c["frames"]), "every": c["every"], "offset": c["offset"],
                   "exclude": c["exclude"], "kept": sum(c["kept_flags"]), "kept_ranges": c["ranges"]}
                  for c in result["clips"]],
        "frames_csv_times": "source_seconds_approx from the selection" if with_times else
                            "omitted: not every kept frame has a time, so camera-review uses frame order",
        "masks_copied": False, "source_mask_dirs": masks,
        "warning": ("the images are relative links to the source capture's images/; keep the source and this "
                    "capture together" if mode == "link" else "the images are copies; the source is not needed"),
        "limits": LIMITS,
        "status": STATUS_BUILT,
    }
    (prov / "frame-selection.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"frames: {len(kept)} {'linked (no copies)' if mode == 'link' else 'copied'}; capture: {capture}")
    print("Pending visual review and a new camera solve; no pipeline stage was launched.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    preview = sub.add_parser("preview", help="inventory a capture and write an offline selection page; "
                                             "writes only a new review folder")
    preview.add_argument("--capture", required=True, help="the existing capture (the directory holding images/)")
    preview.add_argument("--out", required=True, help="the NEW review folder; must not exist")
    preview.add_argument("--clips", help="CSV with columns filename,clip and optionally seconds; "
                                         "must list every image exactly once")
    preview.add_argument("--annotations", help="optional camera-checks.json from camera-review of a prior solve")
    preview.add_argument("--title", default="", help="page title (default: Frame selection: {capture name})")
    preview.add_argument("--ffmpeg-prefix", help="directory holding ffmpeg (default: PATH)")
    preview.set_defaults(func=cmd_preview)

    build = sub.add_parser("build", help="validate a selection JSON and write a NEW capture from it")
    build.add_argument("--selection", required=True, help="the selection JSON downloaded from the preview page")
    build.add_argument("--capture", required=True, help="the NEW capture directory; must not exist")
    build.add_argument("--source", help="the source capture (default: the path recorded in the selection)")
    build.add_argument("--materialize", action="store_true",
                       help="copy the images instead of linking them, so the capture travels alone")
    build.add_argument("--dry-run", action="store_true", help="validate and print the plan; write nothing")
    build.set_defaults(func=cmd_build)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ToolError as err:
        print(f"{TOOL}: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
