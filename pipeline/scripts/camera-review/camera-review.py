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
import statistics
from typing import Dict, List, Optional, Sequence, Tuple

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

#: ``frames.csv`` columns that hold a time in seconds, best first.
TIME_COLUMNS = ('source_seconds_approx', 'output_grid_seconds')

# Review alert, not an automatic acceptance threshold. Reference video
# measurements are about 2.8x median; the teaching-greenhouse bad solve was
# about 19x median and had a visibly displaced camera center.
CAMERA_STEP_WARNING_RATIO = 10.0


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
    root = capture / 'images'
    names = sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES)
    return {n: float(i) for i, n in enumerate(names)}, 'frame order'


def compare_frames(expected: Sequence[str], registered: Sequence[str]) -> Dict[str, List[str]]:
    """Which expected frames did not register, and which registered ones were not expected."""
    expected_set, registered_set = set(expected), set(registered)
    return {'missing': sorted(expected_set - registered_set),
            'unexpected': sorted(registered_set - expected_set)}


def reconcile_frame_names(expected: Dict[str, float], registered: Sequence[str]) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    """Match capture images to COLMAP paths by relative stem; refuse ambiguous mappings.

    Mask compositing changes JPEGs to PNGs but preserves their relative paths.
    """
    def index(names):
        result = {}
        for name in names:
            key = Path(name).with_suffix('').as_posix()
            if key in result:
                raise ValueError(f'ambiguous frame stem: {key}')
            result[key] = name
        return result

    expected_by_stem = index(expected)
    registered_by_stem = index(registered)
    common = expected_by_stem.keys() & registered_by_stem.keys()
    times = {registered_by_stem[key]: expected[expected_by_stem[key]] for key in common}
    frames = {
        'missing': sorted(expected_by_stem[key] for key in expected_by_stem.keys() - common),
        'unexpected': sorted(registered_by_stem[key] for key in registered_by_stem.keys() - common),
    }
    return times, frames


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


def adjacent_expected_pairs(expected: Sequence[str], registered: Sequence[str]) -> List[Tuple[str, str]]:
    """Consecutive expected names within each folder when both registered.

    Lexical relative-name order is an assumption, not verified chronology.
    """
    def stem(name):
        return Path(name).with_suffix('').as_posix()
    registered_stems = {stem(name): name for name in registered}
    groups = {}
    for name in sorted(expected):
        groups.setdefault(Path(name).parent.as_posix(), []).append(name)
    return [(registered_stems[stem(a)], registered_stems[stem(b)])
            for group in groups.values() for a, b in zip(group, group[1:])
            if stem(a) in registered_stems and stem(b) in registered_stems]


def summarize_camera_steps(
    names: Sequence[str],
    times: Sequence[float],
    steps: Sequence[float],
    axis_label: str,
    valid_indices: Optional[Sequence[int]] = None,
    edge_indices: Optional[Sequence[Tuple[int, int]]] = None,
) -> Dict[str, object]:
    """Summarize camera-center steps and flag a single extreme discontinuity.

    Distances are scale-free. The ratio to the median is a review heuristic,
    not a verdict: a genuinely fast camera move can also trigger it.
    """
    values = [float(step) for step in steps]
    edges = list(edge_indices) if edge_indices is not None else list(zip(range(len(names) - 1), range(1, len(names))))
    if len(values) != len(edges) or len(times) != len(names) or any(a < 0 or b < 0 or a >= len(names) or b >= len(names) for a, b in edges):
        raise ValueError("camera names, times and adjacent steps must have matching lengths")
    indices = list(range(len(values))) if valid_indices is None else list(valid_indices)
    if any(i < 0 or i >= len(values) for i in indices):
        raise ValueError("invalid adjacent step index")
    if not indices:
        return {
            "median_step_between_adjacent_registered_cameras": None,
            "largest_step_between_adjacent_registered_cameras": None,
            "largest_step_from": None,
            "largest_step_to": None,
            "largest_step_time_interval": None,
            "largest_step_ratio_to_median": None,
            "camera_path_warnings": [],
        }

    median_step = float(statistics.median(values[i] for i in indices))
    peak_index = max(indices, key=values.__getitem__)
    largest_step = values[peak_index]
    ratio = largest_step / median_step if median_step > 0 else None
    extreme = (
        largest_step > 0 and median_step == 0
    ) or (
        ratio is not None and ratio >= CAMERA_STEP_WARNING_RATIO
    )
    start, end = edges[peak_index]
    start_time, end_time = float(times[start]), float(times[end])
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
            f"({ratio_text}) from {names[start]} to {names[end]}{location}. "
            "Inspect this transition before training; the ratio is a review alert, not proof of a bad pose."
        )

    return {
        "median_step_between_adjacent_registered_cameras": median_step,
        "largest_step_between_adjacent_registered_cameras": largest_step,
        "largest_step_from": names[start],
        "largest_step_to": names[end],
        "largest_step_time_interval": interval,
        "largest_step_ratio_to_median": ratio,
        "camera_path_warnings": warnings,
    }


def collect_camera_alerts(global_summary, folder_summaries):
    """Return all warning messages and unique flagged transitions with their scopes."""
    warnings = list(global_summary["camera_path_warnings"])
    flagged = {}
    def add(scope, summary):
        if summary["camera_path_warnings"]:
            edge = (summary["largest_step_from"], summary["largest_step_to"])
            flagged.setdefault(edge, []).append(scope)
    add("global", global_summary)
    for folder, summary in folder_summaries.items():
        for warning in summary["camera_path_warnings"]:
            warnings.append(f"Folder {folder}: {warning}")
        add(f"folder {folder}", summary)
    return warnings, flagged



SUPPORT_GRID_ROWS = 4
SUPPORT_GRID_COLUMNS = 4


def image_track_support(images, cameras, points):
    """Describe retained observations and their image-plane distribution.

    Count observation entries, not unique point IDs. Only IDs present in the
    reviewed sparse model count. Grid cells use finite pixels in the half-open
    image rectangle, with row zero at the top and column zero at the left.
    Invalid coordinates still count as retained observations but occupy no cell.
    """
    result = {}
    for image in sorted(images.values(), key=lambda image: image.name):
        camera = cameras.get(image.camera_id)
        width = float(camera.width) if camera is not None else 0.0
        height = float(camera.height) if camera is not None else 0.0
        dimensions_valid = (math.isfinite(width) and math.isfinite(height)
                            and width > 0 and height > 0)
        counts = [[0] * SUPPORT_GRID_COLUMNS for _ in range(SUPPORT_GRID_ROWS)]
        retained = in_bounds = 0
        for index, point_id in enumerate(image.point3D_ids):
            if point_id < 0 or point_id not in points:
                continue
            retained += 1
            if not dimensions_valid or index >= len(image.xys):
                continue
            x, y = image.xys[index]
            if not (math.isfinite(x) and math.isfinite(y)
                    and 0 <= x < width and 0 <= y < height):
                continue
            row = min(SUPPORT_GRID_ROWS - 1, int(y / height * SUPPORT_GRID_ROWS))
            column = min(SUPPORT_GRID_COLUMNS - 1, int(x / width * SUPPORT_GRID_COLUMNS))
            counts[row][column] += 1
            in_bounds += 1
        occupied = sum(count > 0 for row in counts for count in row)
        result[image.name] = {
            'camera_id': int(image.camera_id),
            'image_width': width if math.isfinite(width) else None,
            'image_height': height if math.isfinite(height) else None,
            'dimensions_valid': dimensions_valid,
            'retained_3d_observation_count': retained,
            'in_bounds_retained_3d_observation_count': in_bounds,
            'grid_observation_counts': counts,
            'occupied_cell_count': occupied,
            'occupied_cell_fraction': (occupied / (SUPPORT_GRID_ROWS * SUPPORT_GRID_COLUMNS)
                                       if dimensions_valid else None),
        }
    return result



def terminal_summary(report):
    """Keep per-image detail in the JSON file, with compact fields on stdout."""
    detail_fields = {'missing', 'cameras', 'unexpected', 'per_image_track_support'}
    return {key: value for key, value in report.items() if key not in detail_fields}


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
    aligned_seconds, frames = reconcile_frame_names(seconds, names)
    t = np.array([aligned_seconds.get(i.name, np.nan) for i in ordered])
    name_indices = {name: i for i, name in enumerate(names)}
    edges = [(name_indices[a], name_indices[b]) for a, b in adjacent_expected_pairs(list(seconds), names)]
    step = np.array([np.linalg.norm(centers[b] - centers[a]) for a, b in edges])
    cloud = np.stack([pt.xyz for pt in points.values()])
    rgb = np.stack([pt.rgb for pt in points.values()]) / 255.
    errors = np.array([pt.error for pt in points.values()])
    tracks = np.array([len(pt.image_ids) for pt in points.values()])
    components = sparse_models(args.workspace)
    step_summary = summarize_camera_steps(names, t.tolist(), step.tolist(), axis_label, edge_indices=edges)
    folders = sorted({Path(name).parent.as_posix() for name in seconds})
    folder_summaries = {}
    for folder in folders:
        selected = [i for i, (a, _) in enumerate(edges) if Path(names[a]).parent.as_posix() == folder]
        summary = summarize_camera_steps(names, t.tolist(), step.tolist(), axis_label,
                                         selected, edge_indices=edges)
        folder_summaries[folder] = {"adjacent_step_count": len(selected), **summary}

    warnings, flagged = collect_camera_alerts(step_summary, folder_summaries)

    report = {
        'label': args.label, 'registered': len(images), 'expected': len(seconds),
        'missing': frames['missing'], 'unexpected': frames['unexpected'],
        'sparse_models': components, 'reviewed_model': args.model, 'points': len(points),
        'mean_reprojection_error_px': float(errors.mean()), 'mean_track_length': float(tracks.mean()),
        'cameras': {str(k): dict(model=c.model, width=c.width, height=c.height, params=[float(x) for x in c.params])
                    for k, c in cameras.items()},
        'per_image_track_support': image_track_support(images, cameras, points),
        'track_support_grid': {
            'rows': SUPPORT_GRID_ROWS, 'columns': SUPPORT_GRID_COLUMNS,
            'semantics': 'Observation entries whose nonnegative point3D ID exists in the reviewed sparse model. '
                         'Cells partition [0,width) x [0,height), top-to-bottom rows and left-to-right columns. '
                         'Nonfinite or outside-image coordinates occupy no cell but remain in the retained count. '
                         'Invalid or missing camera dimensions yield zero cell counts and null coverage fraction. '
                         'Image-plane occupancy is descriptive, not subject coverage or a pose-quality decision.',
        },
        'time_axis': axis_label,
        'ordering_assumption': 'Lexical relative-name order within each folder; not verified capture chronology.',
        'adjacent_step_count': len(edges),
        'folder_step_summaries': folder_summaries,
        'step_policy': 'Only consecutive expected frames within the same folder when both registered; unexpected frames excluded.',
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
        'camera_path_warnings': warnings,
        'flagged_camera_steps': [dict(from_frame=a, to_frame=b, scopes=scopes)
                                 for (a, b), scopes in flagged.items()],
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
        for start, end in edges:
            ax.plot(centers[[start, end], a], centers[[start, end], b], lw=.7, color='gray')
        sc = ax.scatter(centers[:, a], centers[:, b], c=colors, cmap='viridis', s=14)
        ax.annotate('first', centers[0, [a, b]])
        ax.annotate('last', centers[-1, [a, b]])
        ax.set(xlabel='XYZ'[a] + ' (scene units)', ylabel='XYZ'[b] + ' (scene units)')
        ax.set_aspect('equal', adjustable='datalim')
        fig.colorbar(sc, ax=ax, label=axis_label)
    axes[1, 1].scatter([colors[end] for _, end in edges], step, s=12)
    axes[1, 1].set(xlabel=axis_label, ylabel='distance between adjacent registered cameras (scene units)')
    title = (f'{args.label or args.workspace.name}: {len(images)}/{len(seconds)} registered, '
                 f'{len(components)} sparse model(s), {len(points)} points, '
                 f'mean reprojection {errors.mean():.2f} px\n'
                 'Camera path over the sparse points. Review the geometry; this is not measured accuracy.')
    if flagged:
        edge_names = [(names[a], names[b]) for a, b in edges]
        for pair in flagged:
            index = edge_names.index(pair)
            axes[1, 1].scatter([colors[edges[index][1]]], [step[index]],
                               marker='x', s=65, color='red', zorder=5)
        axes[1, 1].scatter([], [], marker='x', color='red',
                           label='global or folder peak review alert')
        median_step = step_summary['median_step_between_adjacent_registered_cameras']
        if median_step and median_step > 0 and step_summary['camera_path_warnings']:
            axes[1, 1].axhline(
                median_step * CAMERA_STEP_WARNING_RATIO, color='red',
                linestyle='--', linewidth=.8,
                label=f'{CAMERA_STEP_WARNING_RATIO:g}x pooled median')
        axes[1, 1].legend()
        title += f"\nREVIEW: {len(flagged)} flagged transition(s); see JSON for scopes and details."
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(args.out / 'camera-review.png', dpi=130)
    print(json.dumps(terminal_summary(report), indent=2))
    print('missing:', report['missing'])
    print(args.out / 'camera-review.png')


if __name__ == '__main__':
    main()
