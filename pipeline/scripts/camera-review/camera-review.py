"""Descriptive camera and registration review of one COLMAP solve. It reports; it does not accept.

Run it after the ``colmap`` and ``process`` stages and BEFORE training. It draws the camera path over
the sparse points and writes the numbers a person needs to judge the solve: how many frames
registered, which did not, the reprojection error, and whether there is one connected model.

    uv run python pipeline/scripts/camera-review/camera-review.py \\
        --capture {data_root}/datasets/{capture} \\
        --workspace {data_root}/colmap/{capture}/{run-id} \\
        --out {data_root}/evals/{capture}/camera-review

``--capture`` is the capture directory (the one holding ``images/``). If it has a
``provenance/frames.csv`` (written by ``video-frames.py extract``), tiles and the path are labeled with
clip times; otherwise with frame order. ``--workspace`` is the run's COLMAP folder, the one that holds
``sparse/0``. Needs the pipeline environment (nerfstudio, numpy, matplotlib). Read-only on the capture
and the workspace; writes only into ``--out``. See README.md in this directory.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sqlite3
import statistics
from typing import Dict, List, Optional, Sequence, Tuple

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

#: ``frames.csv`` columns that hold a time in seconds, best first.
TIME_COLUMNS = ('source_seconds_approx', 'output_grid_seconds')

# Review alert, not an automatic acceptance threshold. Reference video
# measurements are about 2.8x median; the teaching-greenhouse bad solve was
# about 19x median and had a visibly displaced camera center.
CAMERA_STEP_WARNING_RATIO = 10.0

# Review alert on the turn between two adjacent registered cameras, in degrees. A person walking turns a
# few degrees in half a second (the median in a real walking solve is single digits); a camera put
# facing backward is near 180. It scales with sampling interval, so read it as an alert, not a verdict.
CAMERA_TURN_WARNING_DEGREES = 90.0

# A registered frame that sees fewer than this fraction of the median 3D points per frame is flagged.
LOW_POINT_FRACTION = 0.1

# Verified pairs joining frames further apart than this many frames in order count as "distant".
DISTANT_PAIR_GAP = 3


def frame_times(capture: Path) -> Tuple[Dict[str, float], str]:
    """name -> seconds from the capture's frames.csv, or name -> order if there is none.

    Also returns the axis label to use. A ``frames.csv`` that has none of the known time columns, or
    that is empty, falls back to frame order (the sorted image names) rather than failing.
    """
    csv_path = capture / 'provenance' / 'frames.csv'
    if csv_path.is_file():
        with csv_path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        for column in TIME_COLUMNS:
            if rows and column in rows[0]:
                return {r['filename']: float(r[column]) for r in rows}, 'clip seconds (approx)'
    names = sorted(p.name for p in (capture / 'images').rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES)
    return {n: float(i) for i, n in enumerate(names)}, 'frame order'


def compare_frames(expected: Sequence[str], registered: Sequence[str]) -> Dict[str, List[str]]:
    """Which expected frames did not register, and which registered ones were not expected."""
    expected_set, registered_set = set(expected), set(registered)
    return {'missing': sorted(expected_set - registered_set),
            'unexpected': sorted(registered_set - expected_set)}


def sparse_models(workspace: Path) -> List[str]:
    """The names of the folders under ``sparse/`` that hold a model (more than one means a split solve)."""
    root = workspace / 'sparse'
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir() and (d / 'images.bin').exists())


def finite_span(values: Sequence[float]) -> List[float]:
    """[min, max] of the finite values, or [] when there are none."""
    finite = [v for v in values if v == v and v not in (float('inf'), float('-inf'))]
    return [min(finite), max(finite)] if finite else []


def summarize_camera_steps(
    names: Sequence[str],
    times: Sequence[float],
    steps: Sequence[float],
    axis_label: str,
) -> Dict[str, object]:
    """Summarize camera-center steps and flag a single extreme discontinuity.

    Distances are scale-free. The ratio to the median is a review heuristic,
    not a verdict: a genuinely fast camera move can also trigger it.
    """
    values = [float(step) for step in steps]
    if len(values) != max(0, len(names) - 1) or len(times) != len(names):
        raise ValueError("camera names, times and adjacent steps must have matching lengths")
    if not values:
        return {
            "median_step_between_adjacent_registered_cameras": None,
            "largest_step_between_adjacent_registered_cameras": None,
            "largest_step_from": None,
            "largest_step_to": None,
            "largest_step_time_interval": None,
            "largest_step_ratio_to_median": None,
            "camera_path_warnings": [],
        }

    median_step = float(statistics.median(values))
    peak_index = max(range(len(values)), key=values.__getitem__)
    largest_step = values[peak_index]
    ratio = largest_step / median_step if median_step > 0 else None
    extreme = (
        largest_step > 0 and median_step == 0
    ) or (
        ratio is not None and ratio >= CAMERA_STEP_WARNING_RATIO
    )
    start_time, end_time = float(times[peak_index]), float(times[peak_index + 1])
    interval = (
        [start_time, end_time]
        if math.isfinite(start_time) and math.isfinite(end_time)
        else None
    )
    warnings = []
    if extreme:
        ratio_text = f"{ratio:.1f}x median" if ratio is not None else "unbounded relative to the median"
        location = (
            f" at {start_time:g} to {end_time:g} {axis_label}"
            if interval is not None else ""
        )
        warnings.append(
            f"Largest adjacent camera step is {largest_step:.3f} scene units "
            f"({ratio_text}) from {names[peak_index]} to {names[peak_index + 1]}{location}. "
            "Inspect this transition before training; the ratio is a review alert, not proof of a bad pose."
        )

    return {
        "median_step_between_adjacent_registered_cameras": median_step,
        "largest_step_between_adjacent_registered_cameras": largest_step,
        "largest_step_from": names[peak_index],
        "largest_step_to": names[peak_index + 1],
        "largest_step_time_interval": interval,
        "largest_step_ratio_to_median": ratio,
        "camera_path_warnings": warnings,
    }


def rotation_steps(qvecs: Sequence[Sequence[float]]) -> List[float]:
    """Degrees of rotation between each adjacent pair of COLMAP quaternions (w, x, y, z).

    Uses the absolute dot product, so q and -q (the same rotation) give 0 degrees.
    """
    angles = []
    for a, b in zip(qvecs[:-1], qvecs[1:]):
        dot = abs(sum(float(x) * float(y) for x, y in zip(a, b)))
        norm = math.sqrt(sum(float(x) ** 2 for x in a)) * math.sqrt(sum(float(y) ** 2 for y in b))
        cosine = min(1.0, dot / norm) if norm > 0 else 1.0
        angles.append(math.degrees(2.0 * math.acos(cosine)))
    return angles


def points_per_image(point3d_ids: Sequence[Sequence[int]]) -> List[int]:
    """How many 3D points each image sees (COLMAP marks a 2D feature with no 3D point as -1)."""
    return [sum(1 for p in ids if int(p) >= 0) for ids in point3d_ids]


def flag_camera_steps(
    names: Sequence[str],
    times: Sequence[float],
    steps: Sequence[float],
    turns: Sequence[float],
    spans: Sequence[float],
    ratio_limit: float = CAMERA_STEP_WARNING_RATIO,
    turn_limit: float = CAMERA_TURN_WARNING_DEGREES,
) -> Dict[str, object]:
    """Every adjacent-camera step and turn that looks wrong, not only the largest.

    ``steps[i]`` and ``turns[i]`` join camera i and i+1. ``spans[j - 1]`` is the distance from camera j-1
    to camera j+1 for the interior camera j (so ``len(spans) == len(names) - 2``). A step is divided by
    the time (or frame count) between its two cameras, so a gap left by an unregistered stretch is not
    mistaken for a jump, and is then compared with the median of those rates. Clip boundaries are not
    known here and show up as flagged steps; judge them against where the clips join.

    ``out_and_back`` lists cameras where the path jumps away and returns (two adjacent flagged steps whose
    ends are close together): one frame was placed in the wrong spot.
    """
    n = len(names)
    if len(steps) != max(0, n - 1) or len(turns) != len(steps) or len(times) != n:
        raise ValueError("names, times, steps and turns must have matching lengths")
    if len(spans) != max(0, n - 2):
        raise ValueError("spans must have one value per interior camera")
    rates = []
    for i, step in enumerate(steps):
        dt = float(times[i + 1]) - float(times[i])
        rates.append(float(step) / dt if math.isfinite(dt) and dt > 0 else float(step))
    median_rate = float(statistics.median(rates)) if rates else 0.0
    flagged_index = []
    flagged_steps = []
    for i, rate in enumerate(rates):
        ratio = rate / median_rate if median_rate > 0 else None
        if (ratio is not None and ratio >= ratio_limit) or (ratio is None and rate > 0):
            flagged_index.append(i)
            flagged_steps.append({
                "from": names[i], "to": names[i + 1], "step": float(steps[i]),
                "ratio_to_median_rate": ratio, "turn_degrees": float(turns[i]),
                "also_large_turn": float(turns[i]) >= turn_limit,
            })
    large_turns = [
        {"from": names[i], "to": names[i + 1], "turn_degrees": float(turn)}
        for i, turn in enumerate(turns) if float(turn) >= turn_limit
    ]
    flagged_set = set(flagged_index)
    out_and_back = []
    for j in range(1, n - 1):
        if (j - 1) in flagged_set and j in flagged_set and \
                float(spans[j - 1]) < 0.5 * min(float(steps[j - 1]), float(steps[j])):
            out_and_back.append(names[j])
    warnings = []
    if flagged_steps or large_turns:
        warnings.append(
            f"{len(flagged_steps)} adjacent camera step(s) are at least {ratio_limit:g}x the median rate and "
            f"{len(large_turns)} turn(s) between adjacent cameras are at least {turn_limit:g} degrees"
            + (f"; {len(out_and_back)} camera(s) jump away and straight back ({', '.join(out_and_back)})"
               if out_and_back else "")
            + ". A step with a large turn, or an out-and-back, is the clearest sign of a wrong pose. "
              "Inspect each before training.")
    return {
        "median_step_rate": median_rate,
        "flagged_steps": flagged_steps,
        "flagged_step_count": len(flagged_steps),
        "large_turns": large_turns,
        "large_turn_count": len(large_turns),
        "out_and_back": out_and_back,
        "warnings": warnings,
    }


def low_point_frames(names: Sequence[str], counts: Sequence[int]) -> Dict[str, object]:
    """The median 3D points per frame, and the frames seeing under LOW_POINT_FRACTION of it."""
    if len(names) != len(counts):
        raise ValueError("names and counts must have the same length")
    if not counts:
        return {"median_points_per_image": None, "low_point_frames": []}
    median = float(statistics.median(counts))
    low = [{"frame": n, "points": int(c)} for n, c in zip(names, counts) if c < LOW_POINT_FRACTION * median]
    return {"median_points_per_image": median, "low_point_frames": low}


def distant_pair_summary(database: Path, order: Dict[str, float], gap: int = DISTANT_PAIR_GAP) -> Dict[str, object]:
    """How many verified image pairs in COLMAP's database join frames more than ``gap`` apart in order.

    In a one-way walk most verified pairs should join near neighbors. A large share of distant pairs
    means matching joined look-alike places (racks, trays, lights), which is how a solve folds onto
    itself. Where the path really revisits a spot, some distant pairs are real. Read-only.
    """
    uri = f"file:{database.as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        names = dict(connection.execute("SELECT image_id, name FROM images"))
        pair_ids = [r[0] for r in connection.execute("SELECT pair_id FROM two_view_geometries WHERE rows > 0")]
    finally:
        connection.close()
    position = {n: i for i, n in enumerate(sorted(order, key=lambda k: (order[k], k)))}
    verified = distant = 0
    for pair_id in pair_ids:
        second = pair_id % 2147483647
        first = (pair_id - second) // 2147483647
        a, b = position.get(names.get(first)), position.get(names.get(second))
        if a is None or b is None:
            continue
        verified += 1
        if abs(a - b) > gap:
            distant += 1
    return {"verified_pairs": verified, "distant_pairs": distant, "distance_gap_frames": gap,
            "distant_pair_fraction": (distant / verified) if verified else None}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--capture', type=Path, required=True,
                   help='the capture directory (has images/, and optionally provenance/frames.csv)')
    p.add_argument('--workspace', type=Path, required=True,
                   help="the run's COLMAP workspace (has sparse/0)")
    p.add_argument('--out', type=Path, required=True, help='directory for camera-checks.json and camera-review.png')
    p.add_argument('--label', default='', help='title for the plot (default: the workspace folder name)')
    p.add_argument('--model', default='0',
                   help='which folder under sparse/ to review (default 0, the largest model COLMAP made)')
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    # Heavy imports here, so the pure helpers above load with the standard library alone.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from nerfstudio.data.utils.colmap_parsing_utils import (
        read_cameras_binary, read_images_binary, read_points3D_binary)

    seconds, axis_label = frame_times(args.capture)
    model = args.workspace / 'sparse' / args.model
    if not (model / 'images.bin').is_file():
        raise SystemExit(f'no COLMAP model at {model}. Did the colmap stage produce one?')
    images = read_images_binary(model / 'images.bin')
    cameras = read_cameras_binary(model / 'cameras.bin')
    points = read_points3D_binary(model / 'points3D.bin')
    if not images or not points:
        raise SystemExit(f'the model at {model} has {len(images)} images and {len(points)} points; '
                         'there is nothing to review.')
    ordered = sorted(images.values(), key=lambda i: i.name)
    names = [i.name for i in ordered]
    centers = np.stack([-i.qvec2rotmat().T @ i.tvec for i in ordered])
    t = np.array([seconds.get(i.name, np.nan) for i in ordered])
    step = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    cloud = np.stack([pt.xyz for pt in points.values()])
    rgb = np.stack([pt.rgb for pt in points.values()]) / 255.
    errors = np.array([pt.error for pt in points.values()])
    tracks = np.array([len(pt.image_ids) for pt in points.values()])
    components = sparse_models(args.workspace)
    frames = compare_frames(list(seconds), names)
    step_summary = summarize_camera_steps(names, t.tolist(), step.tolist(), axis_label)
    turns = rotation_steps([i.qvec for i in ordered])
    spans = np.linalg.norm(centers[2:] - centers[:-2], axis=1) if len(ordered) > 2 else np.array([])
    flags = flag_camera_steps(names, t.tolist(), step.tolist(), turns, spans.tolist())
    point_counts = points_per_image([i.point3D_ids for i in ordered])
    low_points = low_point_frames(names, point_counts)
    database = args.workspace / 'database.db'
    distant = distant_pair_summary(database, seconds) if database.is_file() else None

    report = {
        'label': args.label, 'registered': len(images), 'expected': len(seconds),
        'missing': frames['missing'], 'unexpected': frames['unexpected'],
        'sparse_models': components, 'reviewed_model': args.model, 'points': len(points),
        'mean_reprojection_error_px': float(errors.mean()), 'mean_track_length': float(tracks.mean()),
        'cameras': {str(k): dict(model=c.model, width=c.width, height=c.height, params=[float(x) for x in c.params])
                    for k, c in cameras.items()},
        'time_axis': axis_label,
        'registered_span': finite_span(t.tolist()),
        'median_step_between_adjacent_registered_cameras':
            step_summary['median_step_between_adjacent_registered_cameras'],
        'largest_step_between_adjacent_registered_cameras':
            step_summary['largest_step_between_adjacent_registered_cameras'],
        'largest_step_from': step_summary['largest_step_from'],
        'largest_step_to': step_summary['largest_step_to'],
        'largest_step_time_interval': step_summary['largest_step_time_interval'],
        'largest_step_ratio_to_median': step_summary['largest_step_ratio_to_median'],
        'camera_step_warning_ratio': CAMERA_STEP_WARNING_RATIO,
        'camera_path_warnings': step_summary['camera_path_warnings'] + flags['warnings'],
        'camera_turn_warning_degrees': CAMERA_TURN_WARNING_DEGREES,
        'median_step_rate': flags['median_step_rate'],
        'flagged_step_count': flags['flagged_step_count'],
        'flagged_steps': flags['flagged_steps'],
        'large_turn_count': flags['large_turn_count'],
        'large_turns': flags['large_turns'],
        'median_turn_degrees': float(statistics.median(turns)) if turns else None,
        'out_and_back': flags['out_and_back'],
        'median_points_per_image': low_points['median_points_per_image'],
        'low_point_frames': low_points['low_point_frames'],
        'distant_pairs': distant,
        'limits': 'Geometry is scale-free. A plausible path and a high registration count do not prove '
                  'correct poses or coverage of the whole subject; judge the plot, not the counts.',
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'camera-checks.json').write_text(json.dumps(report, indent=2))

    # Frames with no known time (registered but not in the capture's list) are drawn at the first
    # known time instead of being dropped, so the path stays whole.
    colors = np.where(np.isnan(t), np.nanmin(t) if np.isfinite(t).any() else 0.0, t)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    sample = np.linspace(0, len(cloud) - 1, min(20000, len(cloud)), dtype=int)
    for ax, (a, b) in zip(axes.flat[:3], ((0, 1), (0, 2), (1, 2))):
        ax.scatter(cloud[sample, a], cloud[sample, b], s=.5, c=rgb[sample], alpha=.25)
        ax.plot(centers[:, a], centers[:, b], lw=.7, color='gray')
        sc = ax.scatter(centers[:, a], centers[:, b], c=colors, cmap='viridis', s=14)
        ax.annotate('first', centers[0, [a, b]])
        ax.annotate('last', centers[-1, [a, b]])
        ax.set(xlabel='XYZ'[a] + ' (scene units)', ylabel='XYZ'[b] + ' (scene units)')
        ax.set_aspect('equal', adjustable='datalim')
        fig.colorbar(sc, ax=ax, label=axis_label)
    axes[1, 1].plot(colors[1:], step, marker='o', ms=3)
    axes[1, 1].set(xlabel=axis_label, ylabel='distance between adjacent registered cameras (scene units)')
    title = (f'{args.label or args.workspace.name}: {len(images)}/{len(seconds)} registered, '
                 f'{len(components)} sparse model(s), {len(points)} points, '
                 f'mean reprojection {errors.mean():.2f} px\n'
                 'Camera path over the sparse points. Review the geometry; this is not measured accuracy.')
    if report['camera_path_warnings']:
        marked = sorted({int(np.argmax(step))} | {names.index(f['to']) - 1 for f in flags['flagged_steps']})
        axes[1, 1].scatter(
            [colors[i + 1] for i in marked], [step[i] for i in marked], marker='x',
            s=65, color='red', zorder=5)
        median_step = step_summary['median_step_between_adjacent_registered_cameras']
        if median_step and median_step > 0:
            axes[1, 1].axhline(
                median_step * CAMERA_STEP_WARNING_RATIO, color='red',
                linestyle='--', linewidth=.8,
                label=f'{CAMERA_STEP_WARNING_RATIO:g}x median review alert')
            axes[1, 1].legend()
        title += f"\nREVIEW WARNING: {report['camera_path_warnings'][0]}"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(args.out / 'camera-review.png', dpi=130)
    print(json.dumps({k: v for k, v in report.items() if k not in ('missing', 'cameras', 'unexpected')}, indent=2))
    print('missing:', report['missing'])
    print(args.out / 'camera-review.png')


if __name__ == '__main__':
    main()
