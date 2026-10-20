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
``sparse/0``. Needs the pipeline environment (nerfstudio, numpy, matplotlib).
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_binary, read_images_binary, read_points3D_binary

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def frame_times(capture: Path):
    """name -> seconds from the capture's frames.csv, or name -> order if there is none."""
    csv_path = capture / 'provenance' / 'frames.csv'
    if csv_path.is_file():
        with csv_path.open() as handle:
            rows = list(csv.DictReader(handle))
        column = 'source_seconds_approx' if rows and 'source_seconds_approx' in rows[0] else 'output_grid_seconds'
        if rows and column in rows[0]:
            return {r['filename']: float(r[column]) for r in rows}, 'clip seconds (approx)'
    names = sorted(p.name for p in (capture / 'images').rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES)
    return {n: float(i) for i, n in enumerate(names)}, 'frame order'


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--label', default='')
    args = p.parse_args()

    seconds, axis_label = frame_times(args.capture)
    expected = set(seconds)
    model = args.workspace / 'sparse' / '0'
    if not (model / 'images.bin').is_file():
        raise SystemExit(f'no COLMAP model at {model}. Did the colmap stage produce one?')
    images = read_images_binary(model / 'images.bin')
    cameras = read_cameras_binary(model / 'cameras.bin')
    points = read_points3D_binary(model / 'points3D.bin')
    ordered = sorted(images.values(), key=lambda i: i.name)
    names = {i.name for i in ordered}
    centers = np.stack([-i.qvec2rotmat().T @ i.tvec for i in ordered])
    t = np.array([seconds.get(i.name, np.nan) for i in ordered])
    step = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    cloud = np.stack([pt.xyz for pt in points.values()])
    rgb = np.stack([pt.rgb for pt in points.values()]) / 255.
    errors = np.array([pt.error for pt in points.values()])
    tracks = np.array([len(pt.image_ids) for pt in points.values()])
    components = sorted(d.name for d in (args.workspace / 'sparse').iterdir()
                        if d.is_dir() and (d / 'images.bin').exists())

    report = {
        'label': args.label, 'registered': len(images), 'expected': len(expected),
        'missing': sorted(expected - names), 'unexpected': sorted(names - expected),
        'sparse_models': components, 'points': len(points),
        'mean_reprojection_error_px': float(errors.mean()), 'mean_track_length': float(tracks.mean()),
        'cameras': {str(k): dict(model=c.model, width=c.width, height=c.height, params=[float(x) for x in c.params])
                    for k, c in cameras.items()},
        'time_axis': axis_label,
        'registered_span': [float(np.nanmin(t)), float(np.nanmax(t))] if len(t) else [],
        'largest_step_between_adjacent_registered_cameras': float(step.max()) if len(step) else None,
        'limits': 'Geometry is scale-free. A plausible path and a high registration count do not prove '
                  'correct poses or coverage of the whole subject; judge the plot, not the counts.',
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'camera-checks.json').write_text(json.dumps(report, indent=2))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    sample = np.linspace(0, len(cloud) - 1, min(20000, len(cloud)), dtype=int)
    for ax, (a, b) in zip(axes.flat[:3], ((0, 1), (0, 2), (1, 2))):
        ax.scatter(cloud[sample, a], cloud[sample, b], s=.5, c=rgb[sample], alpha=.25)
        ax.plot(centers[:, a], centers[:, b], lw=.7, color='gray')
        sc = ax.scatter(centers[:, a], centers[:, b], c=t, cmap='viridis', s=14)
        ax.annotate('first', centers[0, [a, b]])
        ax.annotate('last', centers[-1, [a, b]])
        ax.set(xlabel='XYZ'[a] + ' (scene units)', ylabel='XYZ'[b] + ' (scene units)')
        ax.set_aspect('equal', adjustable='datalim')
        fig.colorbar(sc, ax=ax, label=axis_label)
    axes[1, 1].plot(t[1:], step, marker='o', ms=3)
    axes[1, 1].set(xlabel=axis_label, ylabel='distance between adjacent registered cameras (scene units)')
    fig.suptitle(f'{args.label or args.workspace.name}: {len(images)}/{len(expected)} registered, '
                 f'{len(components)} sparse model(s), {len(points)} points, '
                 f'mean reprojection {errors.mean():.2f} px\n'
                 'Camera path over the sparse points. Review the geometry; this is not measured accuracy.')
    fig.tight_layout()
    fig.savefig(args.out / 'camera-review.png', dpi=130)
    print(json.dumps({k: v for k, v in report.items() if k not in ('missing', 'cameras', 'unexpected')}, indent=2))
    print('missing:', report['missing'])
    print(args.out / 'camera-review.png')


if __name__ == '__main__':
    main()
