#!/usr/bin/env python3
"""Compute the scene up direction, each plant's vertical axis and radius profile, and the
stacking assignment of detached clusters; record everything. Edits no derivative. See README.md."""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys
from datetime import datetime, timezone

import numpy as np

from analyze_organs import VERSION, read, write, sha, source_revision
import plant_clusters as pc
import plant_orientation as po

CONFIG_KEYS = {'capture_id', 'inputs', 'plants', 'detached', 'soil', 'parameters'}
INPUT_KEYS = {'transforms', 'dataparser_transforms', 'cluster_report'}
ITEM_KEYS = {'name', 'ply', 'clusters', 'rows'}
SOIL_KEYS = {'colmap_sparse', 'soil_masks', 'plant_masks', 'frame_map', 'mask_extension', 'base_plant'}
NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$')


def spec_from(path):
    """Read and validate the JSON config. Relative paths resolve against it."""
    path = Path(path).resolve()
    spec = read(path)
    if not isinstance(spec, dict) or set(spec) - CONFIG_KEYS:
        raise ValueError('Unknown config keys: ' + (str(sorted(set(spec) - CONFIG_KEYS)) if isinstance(spec, dict) else 'not an object'))
    if not isinstance(spec.get('capture_id'), str) or not spec['capture_id'].strip():
        raise ValueError('Expected a nonempty capture_id')
    files = []

    def resolve(value, what, directory=False):
        if not isinstance(value, str):
            raise ValueError('Expected a path for ' + what)
        p = (path.parent / value).resolve()
        if not (p.is_dir() if directory else p.is_file()):
            raise ValueError('Missing %s: %s' % (what, p))
        if not directory:
            files.append(str(p))
        return str(p)

    inputs = spec.get('inputs')
    if not isinstance(inputs, dict) or set(inputs) - INPUT_KEYS or not {'transforms', 'dataparser_transforms'} <= set(inputs):
        raise ValueError('inputs needs transforms and dataparser_transforms, optionally cluster_report')
    inputs = {k: resolve(v, 'inputs.' + k, directory=(k == 'cluster_report')) for k, v in inputs.items()}
    if 'cluster_report' in inputs:
        for name in ['clusters.json', 'labels.npz', 'run-record.json']:
            resolve(str(Path(inputs['cluster_report']) / name), 'cluster report ' + name)

    def item(d, kind, names):
        if not isinstance(d, dict) or set(d) - ITEM_KEYS:
            raise ValueError('Unknown %s keys' % kind)
        if not isinstance(d.get('name'), str) or not NAME.match(d['name']) or d['name'] in names:
            raise ValueError('Each %s needs a unique plain name (letters, digits, . _ -)' % kind)
        names.add(d['name'])
        if ('ply' in d) == ('clusters' in d):
            raise ValueError('%s %s: give exactly one of ply or clusters' % (kind, d['name']))
        out = {'name': d['name']}
        if 'ply' in d:
            out['ply'] = resolve(d['ply'], '%s %s ply' % (kind, d['name']))
        else:
            if 'cluster_report' not in inputs:
                raise ValueError('%s %s selects clusters, which needs inputs.cluster_report' % (kind, d['name']))
            c = d['clusters']
            if not isinstance(c, list) or not c or any(type(k) is not int or k < 0 for k in c) or len(set(c)) != len(c):
                raise ValueError('%s %s clusters must be distinct nonnegative integers' % (kind, d['name']))
            out['clusters'] = sorted(c)
        if 'rows' in d:
            if 'ply' not in d or 'cluster_report' not in inputs:
                raise ValueError('rows (source rows of a ply item) is only meaningful with ply and cluster_report')
            out['rows'] = resolve(d['rows'], '%s %s rows' % (kind, d['name']))
        return out

    plants = spec.get('plants')
    if not isinstance(plants, list) or not plants:
        raise ValueError('plants must be a nonempty list')
    names = set()
    plants = [item(d, 'plant', names) for d in plants]
    detached = spec.get('detached', [])
    if detached == 'auto':
        if 'cluster_report' not in inputs:
            raise ValueError('detached "auto" needs inputs.cluster_report')
    elif isinstance(detached, list):
        detached = [item(d, 'detached cluster', names) for d in detached]
    else:
        raise ValueError('detached must be a list or "auto"')
    soil = spec.get('soil')
    if soil is not None:
        if not isinstance(soil, dict) or set(soil) - SOIL_KEYS or not SOIL_KEYS - {'mask_extension', 'base_plant'} <= set(soil):
            raise ValueError('soil needs colmap_sparse, soil_masks, plant_masks and frame_map; optional mask_extension, base_plant')
        soil = dict(soil)
        soil['colmap_sparse'] = resolve(soil['colmap_sparse'], 'soil colmap_sparse', directory=True)
        for name in ['points3D.bin', 'images.bin']:
            resolve(str(Path(soil['colmap_sparse']) / name), 'COLMAP ' + name)
        for key in ['soil_masks', 'plant_masks']:
            soil[key] = resolve(soil[key], 'soil ' + key, directory=True)
        soil['frame_map'] = resolve(soil['frame_map'], 'soil frame_map')
        ext = soil.setdefault('mask_extension', '.png')
        if not isinstance(ext, str) or not ext.startswith('.') or '/' in ext or '\\' in ext:
            raise ValueError('Expected a filename extension for soil mask_extension')
        if 'base_plant' not in soil:
            if len(plants) != 1:
                raise ValueError('soil.base_plant must name the plant whose base anchors the soil neighborhood')
            soil['base_plant'] = plants[0]['name']
        if soil['base_plant'] not in [p['name'] for p in plants]:
            raise ValueError('soil.base_plant must be one of the plants')
    params = dict(po.DEFAULTS)
    given = spec.get('parameters', {})
    if not isinstance(given, dict) or set(given) - set(params):
        raise ValueError('Unknown orientation parameter: ' + str(sorted(set(given) - set(params)) if isinstance(given, dict) else given))
    params.update(given)
    po.check_parameters(params)
    spec.update(inputs=inputs, plants=plants, detached=detached, soil=soil, parameters=params)
    return spec, files


class Clusters:
    """A cluster_plants.py report: labels over opacity-kept rows of its hashed source PLY."""

    def __init__(self, directory):
        directory = Path(directory)
        report = read(directory / 'clusters.json')
        self.ply, self.ply_sha256 = report['source_ply'], report['source_ply_sha256']
        if sha(self.ply) != self.ply_sha256:
            raise ValueError('cluster report source PLY changed since clustering: ' + self.ply)
        with np.load(directory / 'labels.npz', allow_pickle=False) as z:
            self.kept, self.labels = z['kept_rows'], z['labels']
        prior = read(directory / 'run-record.json')
        if prior.get('status') != 'complete':
            raise ValueError('cluster report run is not complete: ' + str(directory))
        self.opacity_cutoff = prior['spec']['parameters']['opacity_cutoff']
        self.sizes = {int(k[1:]): int(v['size']) for k, v in report['clusters'].items()}
        self.X = pc.centers(pc.load_ply(self.ply))[self.kept]

    def rows(self, clusters):
        unknown = sorted(set(clusters) - set(self.sizes))
        if unknown:
            raise ValueError('Unknown cluster ids: %s' % unknown)
        return np.isin(self.labels, clusters)


def load_item(d, clusters, cutoff):
    """Opacity-kept centers of one plant or detached item, plus source rows when known."""
    if 'ply' in d:
        data = pc.load_ply(d['ply'])
        keep = pc.opacity_kept_rows(data, cutoff)
        rows = np.load(d['rows'], allow_pickle=False) if 'rows' in d else None
        if rows is not None and len(rows) != len(data):
            raise ValueError('%s rows has %d entries for %d PLY rows' % (d['name'], len(rows), len(data)))
        return pc.centers(data)[keep], (rows[keep] if rows is not None else None), \
            {'source': 'ply', 'rows': int(len(data)), 'opacity_kept_rows': int(len(keep))}
    sel = clusters.rows(d['clusters'])
    return clusters.X[sel], clusters.kept[sel], {'source': 'cluster_report', 'clusters': d['clusters'],
                                                 'opacity_kept_rows': int(sel.sum())}


def soil_inputs(soil, transforms, dataparser, hashes):
    """Mapped COLMAP points and soil votes with plant pixels removed from each soil mask in memory."""
    from PIL import Image
    t, d = read(transforms), read(dataparser)
    fmap = read(soil['frame_map'])
    if not isinstance(fmap, dict) or not fmap or len(set(fmap.values())) != len(fmap):
        raise ValueError('Soil frame map must be nonempty and one-to-one')
    name_by_id = {}
    for fr in t['frames']:
        if 'colmap_im_id' not in fr:
            raise ValueError('transforms.json frames need colmap_im_id to relate COLMAP tracks to photographs')
        stem = Path(fr['file_path']).stem
        if stem not in fmap:
            raise ValueError('Frame %s is missing from the soil frame map' % stem)
        name_by_id[fr['colmap_im_id']] = fmap[stem]
    names, views = set(), {}
    w, h = t['w'], t['h']
    for name in sorted(fmap.values()):
        p = Path(soil['soil_masks']) / (name + soil['mask_extension'])
        if not p.exists():
            views[name] = {'name': name, 'soil_mask': 'absent (unknown, not a negative)'}
            continue
        q = Path(soil['plant_masks']) / (name + soil['mask_extension'])
        if not q.exists():
            raise ValueError('Plant mask needed to clean soil mask is missing: ' + str(q))
        hashes[str(p)], hashes[str(q)] = sha(p), sha(q)
        names.add(name)
    if not names:
        raise ValueError('No soil mask matches a frame-map source name')

    def load(name):
        """One view's soil mask minus its plant mask; only this view is held in memory."""
        with Image.open(Path(soil['soil_masks']) / (name + soil['mask_extension'])) as im:
            m = np.array(im.convert('L')) > 127
        with Image.open(Path(soil['plant_masks']) / (name + soil['mask_extension'])) as im:
            m_plant = np.array(im.convert('L')) > 127
        if m.shape != (h, w) or m_plant.shape != (h, w):
            raise ValueError('Soil/plant mask size differs from the camera domain: ' + name)
        soil_px = int(m.sum())
        m &= ~m_plant
        del m_plant
        views[name] = {'name': name, 'soil_pixels': soil_px, 'soil_pixels_after_plant_removal': int(m.sum())}
        return m

    sparse = Path(soil['colmap_sparse'])
    points = po.read_points3d_binary(sparse / 'points3D.bin')
    images = po.read_images_binary(sparse / 'images.bin')
    ids, hits, seen = po.soil_votes(points, images, name_by_id, names, load)
    for name in names - {v for v in views if 'soil_pixels' in views[v]}:
        load(name)  # a view no track reaches: record its pixel counts too
    views = [views[k] for k in sorted(views)]
    SX = po.colmap_to_dataparser(np.array([points[i][0] for i in ids.tolist()]).reshape(-1, 3), d)
    return SX, ids, hits, seen, views


def auto_detached(clusters, plants, plant_rows, share):
    largest = max(clusters.sizes.values())
    used = {c for p in plants for c in p.get('clusters', [])}
    out, skipped = [], []
    for c, size in sorted(clusters.sizes.items()):
        if c in used or size >= share * largest:
            continue
        rows = clusters.kept[clusters.labels == c]
        overlap = {n: int(np.isin(rows, r).sum()) for n, r in plant_rows.items() if r is not None}
        if any(overlap.values()):
            skipped.append({'cluster': c, 'size': size, 'overlap_with_plant_rows': overlap})
            continue
        out.append({'name': 'c%d' % c, 'clusters': [c]})
    return out, {'largest_cluster_size': largest, 'threshold_size': share * largest, 'skipped_overlapping_plants': skipped,
                 'note': 'plants given only as a PLY without rows cannot exclude their own clusters'}


def run(spec, meter):
    params = spec['parameters']
    hashes = {}
    with meter.step('cameras'):
        stems, C, U = po.camera_poses(spec['inputs']['transforms'], spec['inputs']['dataparser_transforms'])
        up_c = po.camera_mean_up(U)
    with meter.step('load_points'):
        clusters = Clusters(spec['inputs']['cluster_report']) if 'cluster_report' in spec['inputs'] else None
        if clusters is not None:
            hashes[clusters.ply] = clusters.ply_sha256
            if clusters.opacity_cutoff != params['opacity_cutoff']:
                raise ValueError('cluster report used opacity_cutoff %s, this run %s; kept centers would differ'
                                 % (clusters.opacity_cutoff, params['opacity_cutoff']))
        plant_pts, plant_rows, plant_info = {}, {}, {}
        for d in spec['plants']:
            plant_pts[d['name']], plant_rows[d['name']], plant_info[d['name']] = load_item(d, clusters, params['opacity_cutoff'])
        auto = None
        detached = spec['detached']
        if detached == 'auto':
            detached, auto = auto_detached(clusters, spec['plants'], plant_rows, params['detached_max_share'])
        det_pts, det_info = {}, {}
        for d in detached:
            Y, rows, det_info[d['name']] = load_item(d, clusters, params['opacity_cutoff'])
            det_pts[d['name']] = Y
            if rows is not None:
                det_info[d['name']]['overlap_with_plant_rows'] = {
                    n: int(np.isin(rows, r).sum()) for n, r in plant_rows.items() if r is not None}
    result = {'schema': 'plant-orientation', 'schema_version': 1, 'version': VERSION, 'capture_id': spec['capture_id'],
              'units': 'scale_free_scene_units', 'frame': 'dataparser (exported model) frame',
              'parameters': params,
              'selection': 'algorithmic: every choice follows from the recorded settings; no visual judgment selects anything',
              'principle': 'the ground plane is orthogonal to up: the soil plane is the primary up; the camera mean '
                           'image-up is a fallback and cross-check only',
              'up_camera_mean_image_up': dict(up_c, frames=stems), 'up_soil_plane': None,
              'plants': {n: dict(plant_info[n]) for n in plant_pts}, 'detached_selection': auto,
              'detached': {}, 'cross_check_camera_mean_image_up': None}
    up_b = None
    if spec['soil'] is not None:
        with meter.step('soil_plane'):
            soil = spec['soil']
            if up_c['status'] != 'ok':
                up_b = {'method': 'soil_plane', 'status': 'not_run', 'normal': None,
                        'reason': 'the first base estimate needs the camera mean image-up, which is unavailable'}
            else:
                SX, ids, hits, seen, views = soil_inputs(soil, spec['inputs']['transforms'],
                                                        spec['inputs']['dataparser_transforms'], hashes)
                up_b = po.soil_plane_iterative(SX, hits, seen, plant_pts[soil['base_plant']], up_c['normal'], params)
                up_b.update(base_plant=soil['base_plant'], soil_mask_views=views, colmap_sparse=soil['colmap_sparse'],
                            mask_rule='soil mask > 127 minus plant mask > 127, in memory; originals untouched')
        result['up_soil_plane'] = up_b
    choice = po.choose_up(up_c, up_b, params)
    result['up'] = choice
    if choice['up'] is None:
        result['status'] = 'refused'
        return result, hashes

    def axes_and_assignments(up):
        axes = {n: po.plant_axis(X, up, params) for n, X in plant_pts.items()}
        assigned = {n: dict(det_info[n], **(po.assign_detached(Y, axes, up, params) if len(Y)
                    else {'assignment': 'unassigned', 'reason': 'no opacity-kept points'})) for n, Y in det_pts.items()}
        return axes, assigned

    with meter.step('axes'):
        up = np.array(choice['up'])
        e1, e2 = po.horizontal_basis(up)
        result['horizontal_basis'] = {'e1': po.floats(e1), 'e2': po.floats(e2)}
        axes, result['detached'] = axes_and_assignments(up)
        for n in axes:
            result['plants'][n]['axis'] = axes[n]
        if choice['method'] == 'soil_plane' and up_c['status'] == 'ok':
            axes_c, det_c = axes_and_assignments(np.array(up_c['normal']))
            result['cross_check_camera_mean_image_up'] = {
                'status': 'cross-check only; not used for any result above',
                'up': up_c['normal'],
                'plants': {n: {k: a.get(k) for k in ['status', 'height_range', 'straightness', 'gaps']} for n, a in axes_c.items()},
                'detached': det_c}
    result['status'] = 'complete'
    return result, hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='Explicit input/parameter JSON; see plant-orientation-example.json')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; existing paths are refused')
    parser.add_argument('--validate-only', action='store_true', help='Validate and hash inputs without creating output')
    args = parser.parse_args(argv)
    out = args.output.resolve()
    if out.exists():
        raise ValueError('Output already exists; choose a fresh directory: ' + str(out))
    spec, files = spec_from(args.config)
    hashes = {p: sha(p) for p in files}
    if args.validate_only:
        print(json.dumps({'status': 'validated', 'plants': len(spec['plants']), 'files': len(hashes),
                          'detached': spec['detached'] if spec['detached'] == 'auto' else len(spec['detached']),
                          'soil': spec['soil'] is not None}))
        return 0
    out.mkdir(parents=True, exist_ok=False)
    meter = pc.StepMeter()
    record = {'version': VERSION, 'status': 'running', 'spec': spec, 'command': sys.argv,
              'working_directory': str(Path.cwd()), 'repository': source_revision(),
              'started_utc': datetime.now(timezone.utc).isoformat(), 'input_sha256': hashes,
              'code_sha256': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
              'versions': {k: importlib.metadata.version(k) for k in ['numpy', 'scipy', 'Pillow']}}
    write(out / 'input-manifest.json', record)
    try:
        result, more = run(spec, meter)
        hashes.update(more)
        result.update(input_sha256=hashes, code_sha256=record['code_sha256'], repository=record['repository'])
        write(out / 'orientation.json', result)
        if any(sha(p) != digest for p, digest in hashes.items()):
            raise ValueError('Source input changed during the run')
        record['status'] = result['status']
        record['input_sha256'] = hashes
        record['output_sha256'] = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'input-manifest.json'}
    except Exception as exc:
        record.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        record['ended_utc'] = datetime.now(timezone.utc).isoformat()
        record['steps'] = meter.steps
        record['peak_rss_kib_max_of_steps'] = meter.peak_kib()
        write(out / 'run-record.json', record)
    up = result['up']
    print('up: %s (%s)' % (up['method'], up['reason']))
    for n, d in result['detached'].items():
        print('%s: %s' % (n, d['assignment']))
    if result['status'] == 'refused':
        print('Refused: ' + str(out / 'orientation.json'), file=sys.stderr)
        return 3
    print('Complete: ' + str(out / 'orientation.json'))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print('orient-plants: ' + str(exc), file=sys.stderr)
        sys.exit(2)
