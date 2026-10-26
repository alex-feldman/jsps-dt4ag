#!/usr/bin/env python3
"""Build a stabilized Nerfstudio camera path from a Nerfstudio transforms.json."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

SCRIPT_VERSION = "1.0.0"
WORLD_UP = [0.0, 0.0, 1.0]


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


def mul(v, amount):
    return [x * amount for x in v]


def length(v):
    return math.sqrt(sum(x * x for x in v))


def unit(v, label):
    norm = length(v)
    if norm < 1e-10:
        raise ValueError(f"{label} has zero length")
    return [x / norm for x in v]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def smooth(values, window):
    radius = window // 2
    out = []
    for i in range(len(values)):
        samples = [values[min(max(j, 0), len(values) - 1)] for j in range(i - radius, i + radius + 1)]
        out.append([sum(row[k] for row in samples) / len(samples) for k in range(len(values[0]))])
    return out


def catmull(values, t):
    if t <= 0:
        return values[0][:]
    if t >= len(values) - 1:
        return values[-1][:]
    i = math.floor(t)
    u = t - i
    p0, p1, p2, p3 = values[max(i - 1, 0)], values[i], values[i + 1], values[min(i + 2, len(values) - 1)]
    u2, u3 = u * u, u * u * u
    return [
        0.5 * (2 * p1[k] + (-p0[k] + p2[k]) * u
               + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * u2
               + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * u3)
        for k in range(len(p1))
    ]


def frame_id(frame):
    path = frame.get("file_path")
    if not isinstance(path, str):
        raise ValueError("every frame needs a string file_path")
    match = re.search(r"(\d+)(?=\.[^.]+$)", Path(path).name)
    if not match:
        raise ValueError(f"cannot find a numeric frame id in {path!r}")
    return int(match.group(1))


def load_source(path):
    source = json.loads(path.read_text(encoding="utf-8"))
    frames = source.get("frames")
    if not isinstance(frames, list) or len(frames) < 4:
        raise ValueError("transforms.json must contain at least four frames")
    frames = sorted(frames, key=frame_id)
    ids = [frame_id(frame) for frame in frames]
    if len(set(ids)) != len(ids):
        raise ValueError("frame filenames have duplicate numeric ids")
    positions, forwards = [], []
    for frame in frames:
        matrix = frame.get("transform_matrix")
        if not isinstance(matrix, list) or len(matrix) != 4 or any(not isinstance(row, list) or len(row) != 4 for row in matrix):
            raise ValueError(f"{frame.get('file_path')}: transform_matrix must be 4x4")
        matrix = [[float(value) for value in row] for row in matrix]
        positions.append([matrix[k][3] for k in range(3)])
        # In Nerfstudio camera-to-world poses, camera rays travel along local -Z.
        forwards.append(unit([-matrix[k][2] for k in range(3)], "camera forward"))
    return source, frames, positions, forwards


def pose_matrix(position, forward):
    backward = unit(mul(forward, -1.0), "camera backward")
    up = WORLD_UP
    right = cross(up, backward)
    if length(right) < 1e-8:
        up = [0.0, 1.0, 0.0]
        right = cross(up, backward)
    right = unit(right, "camera right")
    camera_up = unit(cross(backward, right), "camera up")
    rows = [
        [right[0], camera_up[0], backward[0], position[0]],
        [right[1], camera_up[1], backward[1], position[1]],
        [right[2], camera_up[2], backward[2], position[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return [x for row in rows for x in row]


def smoothstep(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def build(args):
    source_path = args.transforms.resolve()
    source, frames, raw_positions, raw_forwards = load_source(source_path)
    width, height = int(source["w"]), int(source["h"])
    fl_y = float(source.get("fl_y", 0.0))
    if width < 1 or height < 1 or fl_y <= 0:
        raise ValueError("transforms.json needs positive w, h, and fl_y")
    if args.width < 1 or args.height < 1 or abs(args.width / args.height - width / height) > 0.02:
        raise ValueError("output width and height must preserve the source image aspect ratio")
    count_exact = args.seconds * args.fps
    if args.seconds <= 0 or args.fps <= 0 or not math.isclose(count_exact, round(count_exact), abs_tol=1e-7):
        raise ValueError("seconds * fps must be a positive integer frame count")
    if args.window < 1 or args.window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    if not 0 <= args.end_ease_start < 1 or args.bow_fraction < 0 or args.rise_fraction < 0:
        raise ValueError("end-ease-start must be in [0, 1); bow and rise fractions cannot be negative")

    positions = smooth(raw_positions, args.window)
    forwards = [unit(v, "smoothed gaze") for v in smooth(raw_forwards, args.window)]
    route_length = sum(length(sub(b, a)) for a, b in zip(positions, positions[1:]))
    steps = sorted(length(sub(b, a)) for a, b in zip(raw_positions, raw_positions[1:]))
    median_step = steps[len(steps) // 2]
    if route_length <= 0 or median_step <= 0:
        raise ValueError("camera poses do not define a moving path")

    mean_forward = unit([sum(v[k] for v in forwards) / len(forwards) for k in range(3)], "mean gaze")
    inward = unit([mean_forward[0], mean_forward[1], 0.0], "horizontal mean gaze")
    bow = args.bow_fraction * route_length
    rise = args.rise_fraction * route_length
    if bow > median_step or rise > median_step:
        raise ValueError("bow/rise exceeds one median source-camera step; reduce the fractions")

    fov = math.degrees(2 * math.atan(height / (2 * fl_y)))
    path_frames = []
    for index in range(round(count_exact)):
        u = index / (round(count_exact) - 1)
        t = u * (len(positions) - 1)
        position = catmull(positions, t)
        forward = unit(catmull(forwards, t), "interpolated gaze")
        position = add(position, mul(inward, bow * math.sin(math.pi * u)))
        ease = smoothstep((u - args.end_ease_start) / (1.0 - args.end_ease_start))
        position = add(position, mul(WORLD_UP, rise * ease))
        yaw = math.radians(args.end_yaw_deg * ease)
        right = unit(cross(forward, WORLD_UP), "camera right")
        forward = add(mul(forward, math.cos(yaw)), mul(right, math.sin(yaw)))
        pitch = math.radians(args.end_pitch_deg * ease)
        forward = unit(add(mul(forward, math.cos(pitch)), mul(WORLD_UP, -math.sin(pitch))), "final gaze")
        path_frames.append({"camera_to_world": pose_matrix(position, forward), "fov": fov, "aspect": args.width / args.height})

    camera_path = {
        "camera_type": "perspective",
        "render_width": args.width,
        "render_height": args.height,
        "seconds": args.seconds,
        "is_cycle": False,
        "smoothness_value": 0.0,
        "camera_path": path_frames,
    }
    provenance = {
        "schema": "dt4ag-camera-path-provenance-v1",
        "script_version": SCRIPT_VERSION,
        "source_transforms": str(source_path),
        "source_transforms_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_frame_count": len(frames),
        "source_frame_sort": "numeric filename component, ascending",
        "source_first_frame": frames[0]["file_path"],
        "source_last_frame": frames[-1]["file_path"],
        "source_path_length": route_length,
        "median_source_camera_step": median_step,
        "output_camera_count": round(count_exact),
        "output_fps": args.fps,
        "output_seconds": args.seconds,
        "output_width": args.width,
        "output_height": args.height,
        "vertical_fov_degrees": fov,
        "smooth_window_source_frames": args.window,
        "inward_bow_fraction_of_path": args.bow_fraction,
        "inward_bow_max_scene_units": bow,
        "end_rise_fraction_of_path": args.rise_fraction,
        "end_rise_scene_units": rise,
        "end_reveal_start": args.end_ease_start,
        "end_yaw_degrees": args.end_yaw_deg,
        "end_pitch_down_degrees": args.end_pitch_deg,
        "world_up": WORLD_UP,
        "notes": [
            "Input is Nerfstudio transforms.json, not raw COLMAP text models.",
            "Source capture and trained reconstruction are read-only.",
            "The inward bow and end rise are each limited to one median source-camera step.",
        ],
    }
    return camera_path, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transforms", required=True, type=Path, help="Nerfstudio transforms.json from the solved capture")
    parser.add_argument("--out", required=True, type=Path, help="new camera-path JSON path")
    parser.add_argument("--seconds", type=float, default=24.0)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--window", type=int, default=7, help="odd source-frame smoothing window")
    parser.add_argument("--bow-fraction", type=float, default=0.01, help="inward bow as a fraction of route length")
    parser.add_argument("--rise-fraction", type=float, default=0.01, help="final rise as a fraction of route length")
    parser.add_argument("--end-yaw-deg", type=float, default=8.0, help="final pan toward camera-right")
    parser.add_argument("--end-pitch-deg", type=float, default=6.0, help="final downward tilt")
    parser.add_argument("--end-ease-start", type=float, default=0.85, help="route fraction where final reveal starts")
    parser.add_argument("--dry-run", action="store_true", help="print planned metadata without writing files")
    args = parser.parse_args()
    try:
        camera_path, provenance = build(args)
        output = args.out.resolve()
        sidecar = output.with_suffix(".provenance.json")
        if not args.dry_run and (output.exists() or sidecar.exists()):
            raise FileExistsError(f"refusing to overwrite {output} or {sidecar}")
        print(
            f"source_frames={provenance['source_frame_count']} "
            f"range={provenance['source_first_frame']}..{provenance['source_last_frame']} "
            f"path_length={provenance['source_path_length']:.3f} "
            f"render_frames={provenance['output_camera_count']} "
            f"duration={provenance['output_seconds']:.2f}s "
            f"fps={provenance['output_fps']:.2f} "
            f"max_bow={provenance['inward_bow_max_scene_units']:.3f} "
            f"end_rise={provenance['end_rise_scene_units']:.3f}"
        )
        if args.dry_run:
            print(json.dumps(provenance, indent=2))
            return 0
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(camera_path, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        sidecar.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"camera_path={output}")
        print(f"provenance={sidecar}")
        return 0
    except (FileExistsError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"camera-path: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
