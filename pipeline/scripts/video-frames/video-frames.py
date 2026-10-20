#!/usr/bin/env python3
"""Review a movie and extract a reviewed, traceable frame set for the pipeline.

The pipeline starts from an image sequence and never decodes a movie
(``pipeline/QUICKSTART.md``, section 7). This tool is the preparation step in
front of it. It has three subcommands and none launches a pipeline stage.

``review``
    Read-only on the source. Hashes the movie, records ``ffprobe`` output and
    writes contact sheets so a person can choose a time range and a sampling
    interval from the footage. Nothing it writes is a pipeline input.

``extract``
    Stages a hash-checked copy of the movie inside a NEW capture directory,
    extracts frames at the chosen interval into ``images/``, and records how:
    ``provenance/extraction.json``, ``ffprobe.json``, ``ffmpeg.log`` and
    ``frames.csv``. It refuses to touch an existing capture, refuses a source
    whose hash is not the one you name, and stops (non-zero, record kept) when
    it produced no frames or frames of more than one shape.

``subset``
    Makes a sparser capture from an extracted one: every Nth frame becomes a
    symlink in a new capture's ``images/``, with its own provenance. It copies no
    photograph and refuses, rather than copying, where links are impossible.
    Used to compare sampling densities of one video (0.5, 1 and 2 s from one
    0.5 s extraction).

What it deliberately does not do: it does not choose the interval, it does not
measure overlap, and the times it records are the ``fps`` filter's output grid,
not source-frame timestamps. A chosen interval is the person's decision and
does not verify that consecutive frames overlap. Judge that from the camera
review that follows, before training.

Standard library only; needs ``ffmpeg`` and ``ffprobe`` on ``PATH`` or under
``--ffmpeg-prefix``.

Usage::

    python3 pipeline/scripts/video-frames/video-frames.py review \\
        --source /path/to/movie.mp4 --review-root /path/to/evals/subject-video

    python3 pipeline/scripts/video-frames/video-frames.py extract \\
        --source /path/to/movie.mp4 --sha256 {hash from the review} \\
        --capture {data_root}/datasets/{new-capture-id} --interval 0.5 [--dry-run]

    python3 pipeline/scripts/video-frames/video-frames.py subset \\
        --from {data_root}/datasets/{extracted-capture} \\
        --capture {data_root}/datasets/{new-capture-id} --every 2 [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

TOOL = "video-frames.py"

DEFAULT_THREADS = 2
COARSE_SECONDS = 5.0
FINEST_INTERVAL = 0.5             # the preview is decoded once at this interval
PREVIEW_STEPS = (1, 2, 4)         # the page offers every 1st, 2nd and 4th preview frame
PREVIEW_SECONDS = 30.0
END_TOLERANCE = Fraction(1, 20)   # seconds an --end may overshoot the probed duration (rounding)
THUMB_WIDTH = 480

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

TIMESTAMP_WARNING = (
    "output_grid_seconds is the fps filter's nominal output grid (round=up, which samples on the "
    "grid to within one source frame); it is not an exact source-frame timestamp. "
    "source_seconds_approx adds the range start. Captures extracted before 2026-10-20 used ffmpeg's "
    "default rounding, which takes frames up to half an interval LATER than their label."
)

STATUS_PENDING_REVIEW = "extraction_finished_pending_visual_review"
STATUS_FAILED = "failed_preserved"
STATUS_NEEDS_REVIEW = "needs_review_do_not_train"


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


def parse_interval(text: str) -> Fraction:
    """Seconds between extracted frames, as an exact fraction."""
    try:
        value = Fraction(text)
    except (ValueError, ZeroDivisionError):
        raise ToolError(f"interval {text!r} is not a number of seconds")
    if value <= 0:
        raise ToolError(f"interval must be greater than zero, got {text!r}")
    return value


def fps_expression(interval: Fraction) -> str:
    """The ``fps`` filter argument for one frame every ``interval`` seconds."""
    rate = 1 / interval
    return str(rate.numerator) if rate.denominator == 1 else f"{rate.numerator}/{rate.denominator}"


def fps_filter(interval: Fraction) -> str:
    """The ``fps`` filter for one frame every ``interval`` seconds, ON the nominal grid.

    ffmpeg's default rounding (``near``) takes each output frame from the END of its time slot,
    so frames come out late by up to half an interval: measured 2026-10-20 on a clip whose
    brightness encodes its time, ``fps=1/5`` returned frames from 2.4 s and 7.4 s, and ``fps=2``
    from about 0.2 s, 0.7 s, 1.2 s. ``round=up`` returned 0.0, 4.99, 9.98 and 0.0, 0.47, 0.99:
    the nominal times, to within one source frame.
    """
    return f"fps={fps_expression(interval)}:round=up"


def frame_seconds(facts: Dict) -> Fraction:
    """Duration of one source frame, from ``r_frame_rate`` (``24000/1001``); zero if unknown."""
    try:
        num, den = str(facts.get("frame_rate")).split("/")
        return Fraction(int(den), int(num))
    except (ValueError, ZeroDivisionError):
        return Fraction(0)


def interval_label(interval: Fraction) -> str:
    """Filename-safe milliseconds label: 0.5 s becomes ``0500ms``."""
    return f"{int(round(interval * 1000)):04d}ms"


def estimate_frames(duration: float, interval: Fraction, frame_seconds: Fraction = Fraction(0)) -> int:
    """Approximate count for ``fps=...:round=up``: one frame at 0, then one per interval up to
    the last source frame, ``floor((duration - one frame) / interval) + 1``. Measured against
    ffmpeg 9.0.1 on small synthetic clips: exact for 23 of 24 whole-clip cases (a 23.976 fps clip
    was one low), and one or two HIGH for a seeked range, where the seek lands inside a frame.
    It is a guide for the dry run and the page, never a record; the real count is written after."""
    if duration <= 0:
        return 0
    last = Fraction(str(duration)) - frame_seconds
    return 0 if last < 0 else int(math.floor(last / interval)) + 1


def read_png_size(path: Path) -> Tuple[int, int]:
    """Width and height from the PNG header, with no imaging library."""
    with path.open("rb") as handle:
        head = handle.read(24)
    if len(head) < 24 or head[:8] != PNG_SIGNATURE or head[12:16] != b"IHDR":
        raise ToolError(f"{path.name} is not a PNG with a readable header")
    width, height = struct.unpack(">II", head[16:24])
    return width, height


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


# ---------------------------------------------------------------------------
# ffmpeg / ffprobe
# ---------------------------------------------------------------------------

def find_tool(name: str, prefix: Optional[Path]) -> str:
    if prefix is not None:
        candidate = prefix / name
        if not candidate.is_file():
            raise ToolError(f"{name} not found at {candidate}")
        return str(candidate)
    found = shutil.which(name)
    if not found:
        raise ToolError(
            f"{name} is not on PATH; pass --ffmpeg-prefix with the directory "
            f"that holds ffmpeg and ffprobe (QUICKSTART section 3 installs both)"
        )
    return found


def run_capture(cmd: Sequence[str]) -> str:
    result = subprocess.run(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise ToolError(
            f"command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"{result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout.decode(errors="replace")


def probe(ffprobe: str, source: Path) -> Dict:
    raw = run_capture([ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(source)])
    return json.loads(raw)


def summarize_probe(data: Dict) -> Dict:
    """The facts a person needs to pick an interval, from ``ffprobe`` JSON."""
    fmt = data.get("format", {})
    video = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video" and s.get("disposition", {}).get("attached_pic", 0) != 1),
        None,
    )
    if video is None:
        raise ToolError("the source has no video stream")
    rotation = video.get("tags", {}).get("rotate")
    for side in video.get("side_data_list", []):
        if "rotation" in side:
            rotation = side["rotation"]
    duration = float(video.get("duration") or fmt.get("duration") or 0)
    return {
        "video_stream_index": video.get("index"),
        "codec": video.get("codec_name"),
        "pix_fmt": video.get("pix_fmt"),
        "color_transfer": video.get("color_transfer"),
        "width": video.get("width"),
        "height": video.get("height"),
        "frame_rate": video.get("r_frame_rate"),
        "frames_in_stream": video.get("nb_frames"),
        "duration_seconds": duration,
        "rotation": rotation,
        "container": fmt.get("format_name"),
        "size_bytes": int(fmt["size"]) if fmt.get("size") else None,
    }


def ffmpeg_version(ffmpeg: str) -> str:
    out = run_capture([ffmpeg, "-version"])
    return out.splitlines()[0].strip() if out else "unknown"


def review_command(ffmpeg: str, source: Path, review: Path, coarse: Fraction, start: Fraction,
                   window: Fraction, finest: Fraction) -> List[str]:
    """ONE decode of the clip, two outputs: an overview of the whole clip at ``coarse``
    seconds, and the preview window at the finest interval. Coarser preview intervals are
    every Nth of the finest one, so they need no decode of their own."""
    graph = (
        f"[0:v:0]split=2[a][b];"
        f"[a]{fps_filter(coarse)},scale={THUMB_WIDTH}:-1[ov];"
        f"[b]trim=start={float(start)}:duration={float(window)},setpts=PTS-STARTPTS,"
        f"{fps_filter(finest)},scale={THUMB_WIDTH}:-1[fine]"
    )
    return [ffmpeg, "-n", "-hide_banner", "-hwaccel", "none", "-threads", str(DEFAULT_THREADS),
            "-i", str(source), "-filter_complex", graph, "-filter_complex_threads", "1", "-an",
            "-map", "[ov]", "-fps_mode", "passthrough", "-q:v", "4", str(review / "overview" / "ov_%03d.jpg"),
            "-map", "[fine]", "-fps_mode", "passthrough", "-q:v", "4", str(review / "preview" / "pv_%03d.jpg")]


def plan_preview(duration: float, start: Fraction, preview: Fraction, finest: Fraction,
                 steps: Sequence[int], frame_seconds: Fraction = Fraction(0)) -> Dict:
    """What the preview will cover, decided before anything is created. The window is
    ``preview`` seconds from ``start``, cut short by the end of the clip."""
    if start >= Fraction(str(duration)):
        raise ToolError(f"--preview-start {float(start)} is past the end of the {duration:.2f} s clip")
    chosen = sorted(set(steps))
    if not chosen or chosen[0] < 1:
        raise ToolError("--steps must be whole numbers of 1 or more")
    window = min(preview, Fraction(str(duration)) - start)
    return {
        "start_seconds": float(start),
        "window_seconds": float(window),
        "finest_interval_seconds": float(finest),
        "frames": estimate_frames(float(window), finest, frame_seconds),
        "steps": [{
            "n": n,
            "interval_seconds": float(finest * n),
            "label": interval_label(finest * n),
            "preview_frames": len(range(0, estimate_frames(float(window), finest, frame_seconds), n)),
            "whole_clip_frames": estimate_frames(duration, finest * n, frame_seconds),
        } for n in chosen],
    }


def guess_capture_root(review_root: Path) -> str:
    """``{data_root}/datasets`` when the review sits at ``{data_root}/evals/{subject}``
    (the layout the pipeline's LAYOUT.md describes), else empty."""
    for parent in (review_root, *review_root.parents):
        if parent.name == "evals":
            return str(parent.parent / "datasets")
    return ""


def write_index(review: Path, data: Dict) -> Path:
    """Write ``index.html``: labeled thumbnails, a choice of interval, and the ready-made
    ``extract`` command for it. Static and offline; all of its content comes from ``data``."""
    template = (Path(__file__).resolve().parent / "review-index.template.html").read_text(encoding="utf-8")
    # Inside a script element, "<!--<script" stops the next "</script>" from closing it, and a
    # plain "</" can close it early. Escape every character that matters to the HTML parser.
    payload = (json.dumps(data).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e"))
    page = review / "index.html"
    page.write_text(template.replace("__DATA__", payload), encoding="utf-8")
    return page


def extract_command_template(source: Path, digest: str, capture_root: str, prefix: Optional[Path]) -> str:
    """The extract command with ``__INTERVAL__`` and ``__LABEL__`` left for the page to fill.
    Quoting is done here, in tested Python, not in the page."""
    stem = re.sub(r"[^a-z0-9]+", "-", source.stem.lower()).strip("-") or "capture"
    root = capture_root or str(Path.cwd() / "captures")
    parts = [f"python3 {shlex.quote(str(Path(__file__).resolve()))} extract",
             f"--source {shlex.quote(str(source))}", f"--sha256 {digest}",
             f"--capture {shlex.quote(f'{root}/{stem}-__LABEL__-v1')}", "--interval __INTERVAL__"]
    if prefix:
        parts.append(f"--ffmpeg-prefix {shlex.quote(str(prefix))}")
    parts.append("--dry-run")
    return " \\\n    ".join(parts)


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------

def cmd_review(args: argparse.Namespace) -> int:
    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        raise ToolError(f"source not found: {source}")
    prefix = Path(args.ffmpeg_prefix).expanduser() if args.ffmpeg_prefix else None
    ffmpeg, ffprobe = find_tool("ffmpeg", prefix), find_tool("ffprobe", prefix)

    # Everything that can be refused is checked before anything is created.
    data = probe(ffprobe, source)
    facts = summarize_probe(data)
    duration = facts["duration_seconds"]
    coarse = parse_interval(str(args.coarse_seconds))
    finest = parse_interval(str(args.finest_interval))
    start = Fraction(str(args.preview_start))
    preview = Fraction(str(args.preview_seconds))
    if start < 0 or preview <= 0:
        raise ToolError("--preview-start must not be negative and --preview-seconds must be positive")
    plan = plan_preview(duration, start, preview, finest, args.steps, frame_seconds(facts))

    review_root = Path(args.review_root).expanduser().resolve()
    review = review_root / f"sampling-review-{now_stamp()}"
    review.mkdir(parents=True, exist_ok=False)
    (review / "overview").mkdir()
    (review / "preview").mkdir()
    print(f"review dir: {review}", flush=True)

    digest = sha256_of(source)
    (review / "source.sha256").write_text(f"{digest}  {source}\n")
    (review / "ffprobe.json").write_text(json.dumps(data, indent=2))
    summary = {
        "source": str(source), "sha256": digest, "tool": TOOL, **facts,
        "preview": plan,
        "estimate_note": "approximate; the real count is in extraction.json",
    }
    (review / "summary.json").write_text(json.dumps(summary, indent=2))

    cmd = review_command(ffmpeg, source, review, coarse, start, Fraction(str(plan["window_seconds"])), finest)
    print("decoding the clip once for the overview and the preview ...", flush=True)
    with (review / "ffmpeg-review.log").open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        result = subprocess.run(cmd, stdout=log, stderr=log)
    if result.returncode != 0:
        raise ToolError(f"the review decode failed; see {review / 'ffmpeg-review.log'}")
    overview = sorted(p.name for p in (review / "overview").glob("ov_*.jpg"))
    frames = sorted(p.name for p in (review / "preview").glob("pv_*.jpg"))

    index = write_index(review, {
        "source": str(source), "sha256": digest, "facts": facts,
        "overview": {"interval_seconds": float(coarse), "files": [f"overview/{n}" for n in overview]},
        "preview": {**plan, "files": [f"preview/{n}" for n in frames]},
        "command_template": extract_command_template(source, digest, guess_capture_root(review_root), prefix),
    })

    notes = {
        "source": str(source),
        "sha256": digest,
        "purpose": "Disposable CPU-only sampling review; not frozen pipeline input",
        "autorotation": "FFmpeg default pixel autorotation enabled in the previews",
        "thumbnails": {"overview": len(overview), "preview": len(frames)},
        "human_source_identity": None,
        "human_selected_range": None,
        "human_selected_interval_seconds": None,
        "human_verdict": None,
        "agent_proposal": None,
        "selected_interval_does_not_verify_overlap": True,
    }
    (review / "review-notes.json").write_text(json.dumps(notes, indent=2))

    print(f"sha256: {digest}")
    print(f"video : {facts['codec']} {facts['width']}x{facts['height']} "
          f"{facts['frame_rate']} fps, {duration:.2f} s, rotation={facts['rotation']}")
    print("whole clip at each offered interval, approximate frames: " +
          ", ".join(f"{st['interval_seconds']} s: {st['whole_clip_frames']}" for st in plan["steps"]))
    print(f"thumbnails: {len(overview)} overview, {len(frames)} preview")
    print(f"open this, choose an interval, copy the extract command: {index}")
    print("The interval and range are your decision; the page does not check overlap.")
    return 0


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------

def extract_command(ffmpeg: str, staged: Path, images: Path, interval: Fraction,
                    start: Optional[Fraction], end: Optional[Fraction], threads: int) -> List[str]:
    cmd = [ffmpeg, "-n", "-hide_banner", "-hwaccel", "none", "-threads", str(threads)]
    if start:
        cmd += ["-ss", str(float(start))]
    if end is not None:
        cmd += ["-t", str(float(end - (start or 0)))]
    cmd += [
        "-i", str(staged), "-map", "0:v:0", "-an",
        "-vf", f"{fps_filter(interval)},showinfo",
        "-filter_threads", "1", "-fps_mode", "passthrough",
        str(images / "frame_%06d.png"),
    ]
    return cmd


def write_frames_csv(path: Path, frames: Sequence[Path], interval: Fraction,
                     start: Fraction) -> set:
    shapes = set()
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "output_grid_seconds", "source_seconds_approx", "width", "height"])
        for index, frame in enumerate(frames):
            shape = read_png_size(frame)
            shapes.add(shape)
            grid = index * interval
            writer.writerow([frame.name, float(grid), float(start + grid), *shape])
    return shapes


def cmd_extract(args: argparse.Namespace) -> int:
    source = Path(args.source).expanduser().resolve()
    capture = Path(args.capture).expanduser().resolve()
    interval = parse_interval(args.interval)
    start = Fraction(str(args.start)) if args.start else None
    end = Fraction(str(args.end)) if args.end is not None else None
    if start is not None and start < 0:
        raise ToolError("--start must not be negative")
    if end is not None and end <= (start or 0):
        raise ToolError("--end must be greater than --start")
    if args.threads < 1:
        raise ToolError("--threads must be at least 1")
    expected = args.sha256.strip().lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ToolError("--sha256 must be the 64-character hex hash from the review")

    if not source.is_file():
        raise ToolError(f"source not found: {source}")
    if capture.exists():
        raise ToolError(f"REFUSED: {capture} already exists. Preserve it and choose a new capture "
                        f"name or version; a changed sampling plan is a new capture.")
    prefix = Path(args.ffmpeg_prefix).expanduser() if args.ffmpeg_prefix else None
    ffmpeg, ffprobe = find_tool("ffmpeg", prefix), find_tool("ffprobe", prefix)

    actual = sha256_of(source)
    if actual != expected:
        raise ToolError(f"REFUSED: source hash is {actual}, not the {expected} you named.")

    facts = summarize_probe(probe(ffprobe, source))
    duration = facts["duration_seconds"]
    # A range that does not fit the clip is refused here, before anything is created or copied.
    # (A typo such as --end 90 on a 54 s clip is refused, not clamped, so the record never states
    # a range the clip does not have.)
    if start is not None and start >= Fraction(str(duration)):
        raise ToolError(f"--start {float(start)} is at or past the end of the {duration:.2f} s clip")
    if end is not None and end > Fraction(str(duration)) + END_TOLERANCE:
        raise ToolError(f"--end {float(end)} is past the end of the {duration:.2f} s clip")
    window = (end if end is not None else Fraction(str(duration))) - (start or 0)
    if window <= 0:
        raise ToolError(f"the chosen range is outside the {duration:.2f} s clip")
    count = estimate_frames(float(window), interval, frame_seconds(facts))

    staged = capture / "source" / source.name
    images = capture / "images"
    cmd = extract_command(ffmpeg, staged, images, interval, start, end, args.threads)

    print(f"source   : {source}  ({facts['width']}x{facts['height']}, {duration:.2f} s, rotation={facts['rotation']})")
    print(f"range    : {float(start or 0)} to {float(end) if end is not None else 'end'} s")
    print(f"interval : {float(interval)} s ({fps_filter(interval)}); about {count} frames (a guide; the real count is recorded)")
    print(f"capture  : {capture}")
    print("command  : " + " ".join(cmd))
    if args.dry_run:
        print("dry run: nothing written.")
        return 0

    capture.mkdir(parents=True, exist_ok=False)
    for sub in ("images", "source", "provenance"):
        (capture / sub).mkdir()
    prov = capture / "provenance"
    shutil.copyfile(source, staged)
    if sha256_of(staged) != expected:
        raise ToolError("REFUSED: staged movie hash mismatch; the copy is not the source.")
    (prov / "ffprobe.json").write_text(json.dumps(probe(ffprobe, staged), indent=2))

    record = {
        "tool": TOOL,
        "source_original": str(source),
        "source_staged": str(staged),
        "source_sha256": expected,
        "source_identity": args.identity,
        "operator": args.operator,
        "range": {"start_seconds": float(start or 0),
                  "end_seconds": float(end) if end is not None else "end of clip"},
        "interval_seconds": float(interval),
        "interval_selected_by": "person; not verified for overlap",
        "stream": "0:v:0",
        "autorotation": "default FFmpeg pixel autorotation enabled",
        "source_rotation": facts["rotation"],
        "ffmpeg_version": ffmpeg_version(ffmpeg),
        "command": cmd,
        "timestamp_warning": TIMESTAMP_WARNING,
        "expected_frame_count_approx": count,
        "status": "started",
    }
    manifest = prov / "extraction.json"
    manifest.write_text(json.dumps(record, indent=2))

    print(f"extracting on CPU to {images}", flush=True)
    with (prov / "ffmpeg.log").open("w") as log:
        result = subprocess.run(cmd, stdout=log, stderr=log)
    frames = sorted(images.glob("frame_*.png"))
    record["returncode"] = result.returncode
    record["actual_frame_count"] = len(frames)
    if result.returncode != 0:
        record["status"] = STATUS_FAILED
        manifest.write_text(json.dumps(record, indent=2))
        raise ToolError(f"ffmpeg failed ({result.returncode}); see {prov / 'ffmpeg.log'}. Preserve this attempt.")

    shapes = write_frames_csv(prov / "frames.csv", frames, interval, start or Fraction(0))
    record["dimensions"] = [list(s) for s in sorted(shapes)]
    if not frames or len(shapes) != 1:
        record["status"] = STATUS_NEEDS_REVIEW
        manifest.write_text(json.dumps(record, indent=2))
        raise ToolError(f"{len(frames)} frames in {len(shapes)} shapes; do not train. "
                        f"A single COLMAP camera needs one image shape; see pipeline/QUICKSTART.md, section 7, step 4.")
    record["status"] = STATUS_PENDING_REVIEW
    manifest.write_text(json.dumps(record, indent=2))

    print(f"frames: {len(frames)}; dimensions: {sorted(shapes)[0][0]}x{sorted(shapes)[0][1]}")
    print(f"capture: {capture}")
    print("Pending visual review of the frames; no pipeline stage was launched.")
    return 0


# ---------------------------------------------------------------------------
# subset
# ---------------------------------------------------------------------------

MIN_SUBSET_FRAMES = 3


def read_frames_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def cmd_subset(args: argparse.Namespace) -> int:
    """Make a sparser capture from an extracted one without copying a photograph.

    Every Nth frame of the parent becomes a symlink in a NEW capture's
    ``images/``. The pipeline already stages photographs by link
    (``run_pipeline.stage_sfm_inputs``), so a capture of links runs like any other.
    This never falls back to copying: if links cannot be made it stops and removes
    only what it created.
    """
    parent = Path(args.parent).expanduser().resolve()
    capture = Path(args.capture).expanduser().resolve()
    every = args.every
    offset = args.offset
    if every < 2:
        raise ToolError("--every must be at least 2 (1 would be the parent itself)")
    if not 0 <= offset < every:
        raise ToolError(f"--offset must be from 0 to {every - 1}")
    manifest_path = parent / "provenance" / "extraction.json"
    frames_csv = parent / "provenance" / "frames.csv"
    if not manifest_path.is_file() or not frames_csv.is_file():
        raise ToolError(f"{parent} is not a capture made by extract: it needs "
                        f"provenance/extraction.json and provenance/frames.csv")
    parent_record = json.loads(manifest_path.read_text())
    if parent_record.get("status") != STATUS_PENDING_REVIEW:
        raise ToolError(f"the parent's status is {parent_record.get('status')!r}, not "
                        f"{STATUS_PENDING_REVIEW!r}; do not derive from a capture that is not good")
    if capture.exists():
        raise ToolError(f"REFUSED: {capture} already exists. Preserve it and choose a new capture name.")

    rows = read_frames_csv(frames_csv)
    images = parent / "images"
    on_disk = sorted(p.name for p in images.glob("frame_*.png"))
    if [r["filename"] for r in rows] != on_disk:
        raise ToolError("the parent's frames.csv does not match the frames in its images/; "
                        "it has changed since extraction")
    chosen = rows[offset::every]
    if len(chosen) < MIN_SUBSET_FRAMES:
        raise ToolError(f"every {every} from offset {offset} leaves {len(chosen)} of {len(rows)} frames; "
                        f"need at least {MIN_SUBSET_FRAMES}")

    interval = Fraction(str(parent_record["interval_seconds"])) * every
    print(f"parent   : {parent}  ({len(rows)} frames at {parent_record['interval_seconds']} s)")
    print(f"subset   : every {every}th from offset {offset} = {len(chosen)} frames at {float(interval)} s")
    print(f"capture  : {capture}")
    if args.dry_run:
        print("dry run: nothing written.")
        return 0

    capture.mkdir(parents=True, exist_ok=False)
    try:
        (capture / "images").mkdir()
        (capture / "provenance").mkdir()
        resolved = images.resolve()
        for row in chosen:
            try:
                # Relative links, so moving or restoring the whole datasets tree keeps them valid.
                relative = os.path.relpath(resolved / row["filename"], (capture / "images"))
                (capture / "images" / row["filename"]).symlink_to(relative)
            except ValueError as err:     # Windows: the two captures are on different drives
                raise ToolError(f"cannot make a relative link between these locations ({err}); "
                                f"put the new capture on the same drive as its parent.")
            except (OSError, NotImplementedError) as err:
                raise ToolError(f"cannot make symlinks here ({err}); this tool never copies photographs. "
                                f"Use a filesystem that supports links (ext4, not exFAT).")
    except ToolError:
        shutil.rmtree(capture)
        raise

    with (capture / "provenance" / "frames.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(chosen[0].keys()))
        writer.writeheader()
        writer.writerows(chosen)
    record = {
        "tool": TOOL,
        "kind": "subset of an extracted capture, by symlink; no photograph was copied",
        "parent_capture": str(parent),
        "parent_extraction_sha256": sha256_of(manifest_path),
        "parent_source_sha256": parent_record.get("source_sha256"),
        "parent_interval_seconds": parent_record["interval_seconds"],
        "every": every,
        "offset": offset,
        "interval_seconds": float(interval),
        "frame_count": len(chosen),
        "frames": [r["filename"] for r in chosen],
        "link_target_dir": str(resolved),
        "link_style": "relative",
        "dimensions": parent_record.get("dimensions"),
        "warning": "the links point at the parent's images/ by relative path; moving or deleting the "
                   "parent on its own breaks them, moving the whole datasets tree does not",
        "status": STATUS_PENDING_REVIEW,
    }
    (capture / "provenance" / "subset.json").write_text(json.dumps(record, indent=2))

    broken = [n for n in record["frames"] if not (capture / "images" / n).is_file()]
    if broken:
        record["status"] = STATUS_NEEDS_REVIEW
        (capture / "provenance" / "subset.json").write_text(json.dumps(record, indent=2))
        raise ToolError(f"{len(broken)} links do not resolve to a file; do not train")
    print(f"frames: {len(chosen)} linked (no copies); capture: {capture}")
    print("Pending visual review; no pipeline stage was launched.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    review = sub.add_parser("review", help="hash, probe and make contact sheets; writes only a new review directory")
    review.add_argument("--source", required=True, help="the movie")
    review.add_argument("--review-root", required=True,
                        help="directory that will hold a new sampling-review-{timestamp}/ folder")
    review.add_argument("--coarse-seconds", type=float, default=COARSE_SECONDS,
                        help="seconds between overview thumbnails of the whole clip (default 5)")
    review.add_argument("--finest-interval", type=float, default=FINEST_INTERVAL,
                        help="seconds between preview thumbnails; the clip is decoded once at this (default 0.5)")
    review.add_argument("--steps", type=int, nargs="*", default=list(PREVIEW_STEPS),
                        help="the page offers every Nth preview frame for each N (default 1 2 4)")
    review.add_argument("--preview-seconds", type=float, default=PREVIEW_SECONDS,
                        help="how much of the clip the preview covers (default 30; use 60 for a minute)")
    review.add_argument("--preview-start", type=float, default=0.0,
                        help="where in the clip the preview begins (s)")
    review.add_argument("--ffmpeg-prefix", help="directory holding ffmpeg and ffprobe (default: PATH)")
    review.set_defaults(func=cmd_review)

    extract = sub.add_parser("extract", help="stage the movie and extract frames into a NEW capture")
    extract.add_argument("--source", required=True)
    extract.add_argument("--sha256", required=True, help="expected SHA-256 of the source (from the review)")
    extract.add_argument("--capture", required=True, help="the NEW capture directory; must not exist")
    extract.add_argument("--interval", required=True, help="seconds between frames, for example 0.5")
    extract.add_argument("--start", type=float, help="range start in seconds (default: clip start)")
    extract.add_argument("--end", type=float, help="range end in seconds (default: clip end)")
    extract.add_argument("--identity", default="",
                         help="what the footage shows, in the person's words (recorded verbatim)")
    extract.add_argument("--operator", default="Alex launches the command unless explicitly delegated",
                         help="who ran it, recorded verbatim")
    extract.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                         help="ffmpeg decode threads (default 2, to protect host RAM)")
    extract.add_argument("--ffmpeg-prefix", help="directory holding ffmpeg and ffprobe (default: PATH)")
    extract.add_argument("--dry-run", action="store_true",
                         help="verify the source and print the plan; write nothing")
    extract.set_defaults(func=cmd_extract)

    subset = sub.add_parser("subset", help="make a sparser capture from an extracted one, by symlink; copies nothing")
    subset.add_argument("--from", dest="parent", required=True, help="the extracted capture to thin out")
    subset.add_argument("--capture", required=True, help="the NEW capture directory; must not exist")
    subset.add_argument("--every", type=int, required=True,
                        help="keep every Nth frame (2 doubles the interval, 4 quadruples it)")
    subset.add_argument("--offset", type=int, default=0, help="which frame to start from, 0 to N-1 (default 0)")
    subset.add_argument("--dry-run", action="store_true", help="print the plan; write nothing")
    subset.set_defaults(func=cmd_subset)
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
