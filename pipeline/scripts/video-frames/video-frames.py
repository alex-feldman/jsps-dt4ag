#!/usr/bin/env python3
"""Review a movie and extract a reviewed, traceable frame set for the pipeline.

The pipeline starts from an image sequence and never decodes a movie
(``pipeline/QUICKSTART.md``, section 7). This tool is the preparation step in
front of it. It has two subcommands and neither launches a pipeline stage.

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

What it deliberately does not do: it does not choose the interval, it does not
measure overlap, and the times it records are the ``fps`` filter's output grid,
not source-frame timestamps. A chosen interval is the person's decision and
does not verify that consecutive frames overlap. Judge that from the camera
review that follows, before training.

Standard library only; needs ``ffmpeg`` and ``ffprobe`` on ``PATH`` or under
``--ffmpeg-prefix``.

Usage::

    python pipeline/scripts/video-frames/video-frames.py review \\
        --source /path/to/movie.mp4 --review-root /path/to/evals/subject-video

    python pipeline/scripts/video-frames/video-frames.py extract \\
        --source /path/to/movie.mp4 --sha256 {hash from the review} \\
        --capture {data_root}/datasets/{new-capture-id} --interval 0.5 [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
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
CANDIDATE_INTERVALS = (2.0, 1.0, 0.5)
ESTIMATE_INTERVALS = (4.0, 2.0, 1.0, 0.5, 0.25)
SHEET_COLUMNS = 5
SHEET_ROWS = 4
PREVIEW_WIDTH = 320

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

TIMESTAMP_WARNING = (
    "output_grid_seconds is the fps filter's output grid, not an exact "
    "source-frame timestamp; source_seconds_approx adds the range start"
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


def interval_label(interval: Fraction) -> str:
    """Filename-safe milliseconds label: 0.5 s becomes ``0500ms``."""
    return f"{int(round(interval * 1000)):04d}ms"


def estimate_frames(duration: float, interval: Fraction) -> int:
    """Approximate count: ffmpeg's ``fps`` filter keeps a frame at the start and
    one per interval, and drops the one that would land at the clip's end."""
    if duration <= 0:
        return 0
    return int(math.ceil(Fraction(str(duration)) / interval))


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


def sheet_command(ffmpeg: str, source: Path, vf: str, out_pattern: str,
                  seek: Optional[Fraction] = None, one_frame: bool = False) -> List[str]:
    cmd = [ffmpeg, "-n", "-hide_banner", "-hwaccel", "none", "-threads", str(DEFAULT_THREADS)]
    if seek:
        cmd += ["-ss", str(float(seek))]
    cmd += ["-i", str(source), "-map", "0:v:0", "-an", "-vf", vf, "-filter_threads", "1"]
    if one_frame:
        cmd += ["-frames:v", "1"]
    cmd += ["-fps_mode", "passthrough", out_pattern]
    return cmd


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------

def cmd_review(args: argparse.Namespace) -> int:
    source = Path(args.source)
    if not source.is_file():
        raise ToolError(f"source not found: {source}")
    prefix = Path(args.ffmpeg_prefix) if args.ffmpeg_prefix else None
    ffmpeg, ffprobe = find_tool("ffmpeg", prefix), find_tool("ffprobe", prefix)

    review = Path(args.review_root) / f"sampling-review-{now_stamp()}"
    review.mkdir(parents=True, exist_ok=False)
    print(f"review dir: {review}", flush=True)

    digest = sha256_of(source)
    (review / "source.sha256").write_text(f"{digest}  {source}\n")
    data = probe(ffprobe, source)
    (review / "ffprobe.json").write_text(json.dumps(data, indent=2))
    facts = summarize_probe(data)
    duration = facts["duration_seconds"]

    estimates = {
        str(float(i)): estimate_frames(duration, Fraction(str(i))) for i in ESTIMATE_INTERVALS
    }
    summary = {
        "source": str(source),
        "sha256": digest,
        "tool": TOOL,
        **facts,
        "estimated_frames_by_interval_seconds": estimates,
        "estimate_note": "approximate: ceil(duration / interval); the real count is in extraction.json",
    }
    (review / "summary.json").write_text(json.dumps(summary, indent=2))

    coarse = Fraction(str(args.coarse_seconds))
    tile = f"tile={SHEET_COLUMNS}x{SHEET_ROWS}"
    steps = [
        (f"contact-%02d.jpg", f"fps={fps_expression(coarse)},scale={PREVIEW_WIDTH}:-1,{tile}", None, False, "coarse")
    ]
    for value in args.candidate_intervals:
        interval = parse_interval(str(value))
        steps.append((
            f"candidate-{interval_label(interval)}.jpg",
            f"fps={fps_expression(interval)},scale={PREVIEW_WIDTH}:-1,{tile}",
            Fraction(str(args.candidate_start)), True, f"candidate {float(interval)} s",
        ))
    for name, vf, seek, one_frame, label in steps:
        print(f"making {label} sheet", flush=True)
        cmd = sheet_command(ffmpeg, source, vf, str(review / name), seek, one_frame)
        with (review / "ffmpeg-sheets.log").open("a") as log:
            log.write("$ " + " ".join(cmd) + "\n")
            log.flush()
            result = subprocess.run(cmd, stdout=log, stderr=log)
        if result.returncode != 0:
            raise ToolError(f"{label} sheet failed; see {review / 'ffmpeg-sheets.log'}")
    sheets = sorted(p.name for p in review.glob("*.jpg"))

    notes = {
        "source": str(source),
        "sha256": digest,
        "purpose": "Disposable CPU-only sampling review; not frozen pipeline input",
        "autorotation": "FFmpeg default pixel autorotation enabled in the previews",
        "sheets": sheets,
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
    print("approximate frames by interval (s): " +
          ", ".join(f"{k}: {v}" for k, v in estimates.items()))
    print("sheets: " + ", ".join(sheets))
    print("Choose the range and interval yourself from the sheets, then run extract.")
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
        "-vf", f"fps={fps_expression(interval)},showinfo",
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
    source = Path(args.source)
    capture = Path(args.capture)
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
    prefix = Path(args.ffmpeg_prefix) if args.ffmpeg_prefix else None
    ffmpeg, ffprobe = find_tool("ffmpeg", prefix), find_tool("ffprobe", prefix)

    actual = sha256_of(source)
    if actual != expected:
        raise ToolError(f"REFUSED: source hash is {actual}, not the {expected} you named.")

    facts = summarize_probe(probe(ffprobe, source))
    duration = facts["duration_seconds"]
    window = (end if end is not None else Fraction(str(duration))) - (start or 0)
    if window <= 0:
        raise ToolError(f"the chosen range is outside the {duration:.2f} s clip")
    count = estimate_frames(float(window), interval)

    staged = capture / "source" / source.name
    images = capture / "images"
    cmd = extract_command(ffmpeg, staged, images, interval, start, end, args.threads)

    print(f"source   : {source}  ({facts['width']}x{facts['height']}, {duration:.2f} s, rotation={facts['rotation']})")
    print(f"range    : {float(start or 0)} to {float(end) if end is not None else 'end'} s")
    print(f"interval : {float(interval)} s (fps={fps_expression(interval)}); about {count} frames")
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
                        f"A single COLMAP camera needs one image shape (operating-jsps-dt4ag.md, section 4).")
    record["status"] = STATUS_PENDING_REVIEW
    manifest.write_text(json.dumps(record, indent=2))

    print(f"frames: {len(frames)}; dimensions: {sorted(shapes)[0][0]}x{sorted(shapes)[0][1]}")
    print(f"capture: {capture}")
    print("Pending visual review of the frames; no pipeline stage was launched.")
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
                        help="seconds between coarse overview samples (default 5)")
    review.add_argument("--candidate-intervals", type=float, nargs="*", default=list(CANDIDATE_INTERVALS),
                        help="intervals (s) to preview as candidate sheets (default 2 1 0.5)")
    review.add_argument("--candidate-start", type=float, default=0.0,
                        help="where in the clip the candidate sheets begin (s)")
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
