#!/usr/bin/env python3
"""Apply a frame calibration and color leaf regions in a 3DGS export."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tempfile

import numpy as np

import frame_calibration as fc
import plant_asset as pa
import transform_splat as ts
from analyze_organs import DEFAULTS, VERSION, sha
from leaf_lift import opacity_keep
from viewer import PALETTE

# Use the degree-zero basis constant from transform_splat's SH implementation.
C0 = float(ts.sh_basis(np.array([[0.0, 0.0, 1.0]]), 0)[0, 0])
LUMINANCE = np.array([0.2126, 0.7152, 0.0722], np.float64)
HEX64 = re.compile(r'^[0-9a-f]{64}$')
SIDECAR_SCHEMA = 'colored-splat'
SIDECAR_SCHEMA_VERSION = 1
PALETTE_SOURCE = 'pipeline/scripts/organ-analysis/viewer.py:PALETTE'


def read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def source_ply_hash(document):
    if not isinstance(document, dict):
        return None
    for key in ('source_ply_sha256', 'source_input_ply_sha256', 'input_ply_sha256'):
        value = document.get(key)
        if isinstance(value, str) and HEX64.fullmatch(value):
            return value
    hashes = document.get('input_sha256')
    spec = document.get('spec')
    ply = spec.get('inputs', {}).get('ply') if isinstance(spec, dict) else None
    if isinstance(hashes, dict) and isinstance(ply, str):
        value = hashes.get(ply)
        if isinstance(value, str) and HEX64.fullmatch(value):
            return value
    return None


def selected_regions_from_evidence(document):
    leaves = document.get('leaves') if isinstance(document, dict) else None
    if not isinstance(leaves, list) or not all(isinstance(row, dict) and 'association_support' in row for row in leaves):
        return None
    return {int(row.get('id', row.get('leaf_id'))) for row in leaves
            if row['association_support'] == 'supported'
            and isinstance(row.get('id', row.get('leaf_id')), int)
            and not isinstance(row.get('id', row.get('leaf_id')), bool)}


def selected_regions_from_leaf_json(document):
    clusters = document.get('clusters') if isinstance(document, dict) else None
    if not isinstance(clusters, list):
        return None
    return {int(row['leaf_id']) for row in clusters
            if isinstance(row, dict) and isinstance(row.get('leaf_id'), int)
            and not isinstance(row.get('leaf_id'), bool)}


def label_vector(value, source):
    labels = np.asarray(value)
    if labels.ndim != 1 or labels.dtype.kind not in 'iu':
        raise ValueError('Labels must be a one-dimensional integer array: ' + str(source))
    labels = labels.astype(np.int64, copy=False)
    unsupported = sorted(int(v) for v in np.unique(labels) if v < -2)
    if unsupported:
        raise ValueError('Unsupported negative label values: ' + ', '.join(map(str, unsupported)))
    return labels


def npz_arrays(path, names=None):
    with np.load(path, allow_pickle=False) as arrays:
        selected = arrays.files if names is None else [name for name in arrays.files if name in names]
        return {name: arrays[name] for name in selected}


def asset_root_for_file(path):
    for parent in (path.parent, *path.parents):
        if (parent / 'plant.json').is_file():
            return parent.resolve()
    return None


def load_asset_labels(asset_dir, explicit_file=None):
    asset_dir = Path(asset_dir).resolve()
    plant_path = asset_dir / 'plant.json'
    asset = pa.load_asset(asset_dir)
    pa.validate_asset(asset_dir, asset)
    labels_path = pa.inside(asset_dir, asset['point_labels']['path'])
    if explicit_file is not None and labels_path.resolve() != Path(explicit_file).resolve():
        raise ValueError('The supplied NPZ is not this plant asset point_labels file')
    arrays = npz_arrays(labels_path, {'label', 'X'})
    if 'label' not in arrays:
        raise ValueError('Plant asset point_labels NPZ has no label array')
    labels = label_vector(arrays['label'], labels_path)
    selected = set()
    for leaf in asset.get('leaves', []):
        cluster_id = leaf.get('source_cluster_id')
        if not isinstance(cluster_id, int) or isinstance(cluster_id, bool):
            raise ValueError('Plant asset leaf has an invalid source_cluster_id')
        selected.add(cluster_id)
    leaf_files = sorted((asset_dir / 'leaves').glob('*.npz'), key=lambda p: p.name)
    if len(leaf_files) != len(asset.get('leaves', [])):
        raise ValueError('Plant asset leaf files do not match plant.json')
    members = [{'path': p.relative_to(asset_dir).as_posix(), 'sha256': sha(p)} for p in leaf_files]
    directory_sha256 = digest_bytes(''.join(sorted(row['sha256'] for row in members)).encode('ascii'))
    source_hashes, evidence_paths = [], []
    for reference in asset.get('source_evidence', []):
        evidence_path = pa.inside(asset_dir, reference['path'])
        evidence_paths.append(evidence_path)
        if Path(reference['path']).name in ('organ-evidence-original.json', 'organ-evidence.json'):
            value = source_ply_hash(read_json(evidence_path))
            if value:
                source_hashes.append(value)
    if len(set(source_hashes)) > 1:
        raise ValueError('Plant asset evidence records conflicting source PLY hashes')
    provenance_paths = [plant_path, labels_path, *leaf_files, *evidence_paths,
                        pa.inside(asset_dir, asset['reviewed_graph']['path'])]
    provenance_paths = sorted({p.resolve() for p in provenance_paths}, key=str)
    return {
        'labels': labels, 'points': arrays.get('X'), 'selected_regions': selected,
        'selection_basis': 'plant.json leaves roster', 'source_ply_sha256': source_hashes[0] if source_hashes else None,
        'kind': 'plant_asset_directory', 'path': asset_dir, 'sha256': directory_sha256,
        'hash_method': 'sha256 of sorted leaf NPZ SHA-256 hex digests', 'members': members,
        'provenance_paths': provenance_paths,
    }


def verify_record_outputs(record, labels_path, sidecars):
    if record.get('status') != 'complete':
        raise ValueError('Label run-record is not complete')
    output_hashes = record.get('output_sha256', {})
    for path in [labels_path, *sidecars]:
        if path.name == 'run-record.json':
            continue
        expected = output_hashes.get(path.name)
        if expected is None:
            raise ValueError('Label run-record has no output hash for ' + path.name)
        if sha(path) != expected:
            raise ValueError('Label run-record hash mismatch: ' + path.name)


def load_npz_labels(labels_path):
    labels_path = Path(labels_path).resolve()
    arrays = npz_arrays(labels_path, {'label', 'X', 'source_ply_sha256',
                                      'source_input_ply_sha256', 'input_ply_sha256'})
    if 'label' not in arrays:
        raise ValueError('Labels NPZ has no label array: ' + str(labels_path))
    labels = label_vector(arrays['label'], labels_path)
    source_hashes = []
    for key in ('source_ply_sha256', 'source_input_ply_sha256', 'input_ply_sha256'):
        if key in arrays and np.asarray(arrays[key]).size == 1:
            value = np.asarray(arrays[key]).reshape(()).item()
            if isinstance(value, bytes):
                value = value.decode('ascii')
            if isinstance(value, str) and HEX64.fullmatch(value):
                source_hashes.append(value)
                break
    names = ('organ-evidence.json', 'organ-evidence-original.json', 'leaf_ids2.json',
             'input-manifest.json', 'run-record.json')
    sidecars = [labels_path.parent / name for name in names if (labels_path.parent / name).is_file()]
    documents = [(path, read_json(path)) for path in sidecars]
    selected, selection_basis = None, None
    for path, document in documents:
        value = source_ply_hash(document)
        if value:
            source_hashes.append(value)
        if selected is None:
            selected = selected_regions_from_evidence(document)
            if selected is not None:
                selection_basis = 'organ-evidence supported-cluster roster'
        if selected is None:
            selected = selected_regions_from_leaf_json(document)
            if selected is not None:
                selection_basis = 'leaf_ids2.json supported-cluster summary'
    evidence_path = next((p for p in sidecars if p.name == 'organ-evidence.json'), None)
    if evidence_path:
        document = read_json(evidence_path)
        point_labels = document.get('point_labels', {})
        if isinstance(point_labels, dict):
            expected = point_labels.get('sha256')
            if expected and expected != sha(labels_path):
                raise ValueError('organ-evidence point-label hash does not match the labels NPZ')
            named_file = point_labels.get('file')
            if named_file and Path(named_file).name != labels_path.name:
                raise ValueError('organ-evidence points at a different labels file')
    run_record_path = next((p for p in sidecars if p.name == 'run-record.json'), None)
    if run_record_path:
        verify_record_outputs(read_json(run_record_path), labels_path, sidecars)
    if len(set(source_hashes)) > 1:
        raise ValueError('Label source records conflicting source PLY hashes')
    provenance_paths = sorted({labels_path.resolve(), *(p.resolve() for p in sidecars)}, key=str)
    return {
        'labels': labels, 'points': arrays.get('X'), 'selected_regions': selected,
        'selection_basis': selection_basis, 'source_ply_sha256': source_hashes[0] if source_hashes else None,
        'kind': 'leaf_ids2_npz', 'path': labels_path, 'sha256': sha(labels_path),
        'hash_method': 'sha256 of the labels NPZ', 'members': [], 'provenance_paths': provenance_paths,
    }


def load_labels(path):
    """Load a valid plant-asset directory or a leaf_ids2.npz with companion evidence."""
    path = Path(path).resolve()
    if path.is_dir():
        if not (path / 'plant.json').is_file():
            raise ValueError('Labels directory must contain plant.json: ' + str(path))
        return load_asset_labels(path)
    if not path.is_file():
        raise ValueError('Labels path does not exist: ' + str(path))
    root = asset_root_for_file(path)
    if root is not None:
        return load_asset_labels(root, explicit_file=path)
    return load_npz_labels(path)


def validate_color_splat_data(data):
    required = {'x', 'y', 'z', 'opacity', 'scale_0', 'scale_1', 'scale_2',
                'rot_0', 'rot_1', 'rot_2', 'rot_3', 'f_dc_0', 'f_dc_1', 'f_dc_2'}
    required.update(name for name in data.dtype.names if name.startswith('f_rest_'))
    missing = sorted(required - set(data.dtype.names))
    if missing:
        raise ValueError('Missing splat properties: ' + ', '.join(missing))
    for name in required:
        if not np.isfinite(data[name]).all():
            raise ValueError('Non-finite splat property: ' + name)


def property_signature(lines):
    return [line for line in lines if line.startswith(('format ', 'element ', 'property '))]


def same_bytes(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def color_properties(data):
    return {name for name in data.dtype.names if name.startswith(('f_dc_', 'f_rest_'))}


def check_analysis(calibrated, colored, input_header, output_header):
    if len(calibrated) != len(colored):
        raise ValueError('Self-check failed: colored vertex count changed')
    if property_signature(input_header) != property_signature(output_header):
        raise ValueError('Self-check failed: PLY element/property header changed')
    allowed = color_properties(calibrated)
    for name in calibrated.dtype.names:
        if name not in allowed and not same_bytes(calibrated[name], colored[name]):
            raise ValueError('Self-check failed: non-color property changed: ' + name)
    for name in ('x', 'y', 'z', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3', 'opacity'):
        if not same_bytes(calibrated[name], colored[name]):
            raise ValueError('Self-check failed: geometry/opacity changed: ' + name)


def recolor(calibrated, labels, raw_rows, regions, variant):
    output = calibrated.copy()
    raw_labels = np.full(len(calibrated), -1, np.int64)
    raw_labels[raw_rows] = labels
    region_mask = np.isin(raw_labels, sorted(regions))
    dc = np.stack([calibrated['f_dc_%d' % c] for c in range(3)], axis=1).astype(np.float64)
    changed = dc.copy()
    rgb_palette = np.array([[int(color[i:i + 2], 16) for i in (1, 3, 5)] for color in PALETTE], np.float64) / 255.0
    if region_mask.any():
        ids = raw_labels[region_mask]
        changed[region_mask] = (rgb_palette[ids % len(PALETTE)] - 0.5) / C0
    zero_rest = region_mask.copy()
    if variant == 'B':
        rgb = np.clip(0.5 + C0 * dc, 0.0, 1.0)
        gray = rgb @ LUMINANCE
        changed[~region_mask] = ((gray[~region_mask] - 0.5) / C0)[:, None]
        zero_rest[:] = True
    for channel in range(3):
        output['f_dc_%d' % channel] = changed[:, channel]
    for name in output.dtype.names:
        if name.startswith('f_rest_') and zero_rest.any():
            output[name][zero_rest] = 0
    return output


def color_header(lines, sidecar_name):
    result = []
    for line in lines:
        result.append(line)
        if line.startswith('format '):
            result.append('comment jsps-dt4ag %s color_splat: sidecar %s' % (VERSION, sidecar_name))
    return result


def display_header(lines, viewer, analysis_name, sidecar_name):
    result = []
    for line in lines:
        result.append(line)
        if line.startswith('format '):
            result.append('comment DISPLAY ONLY for %s: rotated analysis file %s; sidecar %s'
                          % (viewer, analysis_name, sidecar_name))
    return result


def count_records(labels, selected_regions, regions, below_cutoff):
    residual = None if selected_regions is None else int(
        np.count_nonzero((labels >= 0) & ~np.isin(labels, sorted(selected_regions))))
    region_counts = {str(region): int(np.count_nonzero(labels == region)) for region in sorted(regions)}
    return region_counts, {'non_leaf': int(np.count_nonzero(labels == -2)),
                           'unlabeled': int(np.count_nonzero(labels == -1)),
                           'residual': residual, 'below_opacity_cutoff': int(below_cutoff)}


def sidecar_base(input_path, input_sha256, calibration_path, calibration_sha256, calibration,
                 label_source, cutoff, include_residual, variant, region_counts, unlabeled, transform_record):
    return {
        'schema': SIDECAR_SCHEMA, 'schema_version': SIDECAR_SCHEMA_VERSION,
        'generator': {'name': fc.GENERATOR, 'version': VERSION, 'command': 'color_splat.py'},
        'input': {'path': str(input_path), 'sha256': input_sha256},
        'calibration': {'path': str(calibration_path), 'sha256': calibration_sha256,
                        'capture_id': calibration['capture_id'], 'frame': calibration['target_frame']},
        'labels': {
            'path': str(label_source['path']), 'sha256': label_source['sha256'],
            'kind': label_source['kind'], 'hash_method': label_source['hash_method'],
            'source_ply_sha256': label_source['source_ply_sha256'],
            'source_ply_binding': 'matched', 'selection_basis': label_source['selection_basis'],
            'selected_regions': sorted(label_source['selected_regions'])
            if label_source['selected_regions'] is not None else None,
            'included_regions': sorted(regions_for_source(label_source, include_residual)),
            'members': label_source['members'],
            'provenance_files': [{'path': str(path), 'sha256': sha(path)}
                                 for path in label_source['provenance_paths']],
        },
        'palette': {'source': PALETTE_SOURCE, 'colors': list(PALETTE)},
        'settings': {'opacity_cutoff': cutoff, 'include_residual': include_residual,
                     'variant': variant, 'calibration_transform': transform_record},
        'region_counts': region_counts, 'unlabeled_breakdown': unlabeled,
    }


def regions_for_source(label_source, include_residual):
    labels = label_source['labels']
    if include_residual:
        return {int(value) for value in np.unique(labels) if value >= 0}
    return set(label_source['selected_regions'] or ())


def legend_documents(input_path, calibration_path, label_source, cutoff, include_residual,
                     regions, region_counts, unlabeled, variants):
    entries = [{'region_id': region, 'color': PALETTE[region % len(PALETTE)],
                'filtered_gaussians': region_counts.get(str(region), 0)} for region in sorted(regions)]
    legend = {
        'schema': 'leaf-color-legend', 'schema_version': 1,
        'generator': {'name': fc.GENERATOR, 'version': VERSION},
        'input': str(input_path), 'calibration': str(calibration_path),
        'labels': {'path': str(label_source['path']), 'sha256': label_source['sha256']},
        'palette_source': PALETTE_SOURCE, 'palette': list(PALETTE),
        'opacity_cutoff': cutoff, 'include_residual': include_residual,
        'variants': variants, 'regions': entries, 'unlabeled_breakdown': unlabeled,
    }
    lines = ['# Leaf-region color legend', '',
             'Colors match the organ-analysis viewer. Counts use opacity-filtered rows.',
             'A colors selected regions and preserves other appearance. B also grays other rows.', '',
             '| Region | Color | Filtered Gaussians |', '|---:|:---:|---:|']
    lines.extend('| %d | %s | %d |' % (row['region_id'], row['color'], row['filtered_gaussians'])
                 for row in entries)
    lines.extend(['', 'Unlabeled breakdown:', '', '- Non-leaf: %s' % unlabeled['non_leaf'],
                  '- Unlabeled: %s' % unlabeled['unlabeled'],
                  '- Residual: %s' % ('unknown' if unlabeled['residual'] is None else unlabeled['residual']),
                  '- Below opacity cutoff: %s raw rows' % unlabeled['below_opacity_cutoff'],
                  '', 'Per-output `.color.json` sidecars contain provenance and self-checks.', ''])
    return legend, '\n'.join(lines)


def verify_sources(expected_hashes):
    for path, expected in expected_hashes.items():
        if sha(path) != expected:
            raise ValueError('An input changed during color_splat: ' + str(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True, type=Path, help='raw 3DGS export PLY (read only)')
    parser.add_argument('--calibration', required=True, type=Path, help='current frame-calibration JSON')
    parser.add_argument('--labels', required=True, type=Path, help='plant-asset directory or leaf_ids2.npz')
    parser.add_argument('--out-dir', required=True, type=Path, help='new output directory; existing paths are refused')
    parser.add_argument('--variant', choices=['A', 'B', 'both'], default='both')
    parser.add_argument('--viewer', choices=sorted(ts.VIEWERS), help='also write display-only copies for this viewer')
    parser.add_argument('--include-residual', action='store_true',
                        help='color positive region labels outside the selected-region roster')
    parser.add_argument('--opacity-cutoff', type=float, default=DEFAULTS['opacity_cutoff'])
    args = parser.parse_args(argv)
    if not math.isfinite(args.opacity_cutoff) or not 0.0 < args.opacity_cutoff < 1.0:
        raise ValueError('--opacity-cutoff must be finite and in (0,1)')

    input_path, calibration_path, out_dir = args.input.resolve(), args.calibration.resolve(), args.out_dir.resolve()
    if not input_path.is_file():
        raise ValueError('Input PLY does not exist: ' + str(input_path))
    if out_dir.exists():
        raise ValueError('Output directory already exists; choose a fresh directory: ' + str(out_dir))
    if not out_dir.parent.is_dir():
        raise ValueError('Output parent directory does not exist: ' + str(out_dir.parent))
    calibration = fc.load(calibration_path)
    if args.viewer:
        required_up = ts.VIEWERS[args.viewer]['requires_up']
        up = calibration['target_frame'].get('up', fc.DEFAULT_UP)
        if up != required_up:
            raise ValueError('--viewer %s needs a calibration with target up %s; this one has %s'
                             % (args.viewer, required_up, up))

    label_source = load_labels(args.labels)
    input_sha256, calibration_sha256 = sha(input_path), sha(calibration_path)
    if not label_source['source_ply_sha256']:
        raise ValueError('Label source has no recorded input PLY hash')
    if label_source['source_ply_sha256'] != input_sha256:
        raise ValueError('Label source PLY hash differs from --input')
    if label_source['selected_regions'] is None and not args.include_residual \
            and np.any(label_source['labels'] >= 0):
        raise ValueError('Cannot distinguish selected regions from residual labels; provide evidence or use --include-residual')

    header, data = ts.read_ply(input_path)
    ts.layout(data)
    validate_color_splat_data(data)
    keep = opacity_keep(data, args.opacity_cutoff)
    raw_rows = np.flatnonzero(keep)
    labels = label_source['labels']
    if len(labels) != len(raw_rows):
        raise ValueError('Label length %d != opacity-filtered rows %d' % (len(labels), len(raw_rows)))
    raw_xyz = np.stack([data['x'], data['y'], data['z']], axis=1).astype(np.float64)
    if label_source['points'] is not None:
        points = np.asarray(label_source['points'], np.float64)
        if points.shape != (len(labels), 3) or not np.isfinite(points).all():
            raise ValueError('Label source X must be a finite (N,3) filtered point array')
        if not np.array_equal(points, raw_xyz[raw_rows]):
            raise ValueError('Label coordinates do not match the opacity-filtered input PLY rows')
    regions = ({int(value) for value in np.unique(labels) if value >= 0} if args.include_residual
               else set(label_source['selected_regions'] or ()))
    present = {int(value) for value in np.unique(labels) if value >= 0}
    if not regions <= present:
        raise ValueError('Selected-region metadata names a region absent from the labels')
    variants = ['A', 'B'] if args.variant == 'both' else [args.variant]
    region_counts, unlabeled = count_records(labels, label_source['selected_regions'], regions, int((~keep).sum()))
    R, origin, scale = ts.parameters(calibration)
    calibrated, transform_record = ts.transform(data, R, origin, scale)
    expected_hashes = {input_path: input_sha256, calibration_path: calibration_sha256}
    for path in label_source['provenance_paths']:
        expected_hashes[path] = sha(path)

    targets = []
    for variant in variants:
        targets += ['%s_leafcolor-%s.ply' % (input_path.stem, variant),
                    '%s_leafcolor-%s.color.json' % (input_path.stem, variant)]
        if args.viewer:
            targets += ['%s_leafcolor-%s-%s.ply' % (input_path.stem, variant, args.viewer),
                        '%s_leafcolor-%s-%s.color.json' % (input_path.stem, variant, args.viewer)]
    targets += ['legend.md', 'legend.json']
    if len(targets) != len(set(targets)):
        raise ValueError('Output name collision')

    with tempfile.TemporaryDirectory(prefix='.color-splat-', dir=str(out_dir.parent)) as temporary:
        temp_dir = Path(temporary)
        written = []
        for variant in variants:
            name = '%s_leafcolor-%s.ply' % (input_path.stem, variant)
            side_name = '%s_leafcolor-%s.color.json' % (input_path.stem, variant)
            transformed_header = ts.updated_header(header, calibration, False, True, side_name)
            analysis_header = color_header(transformed_header, side_name)
            colored = recolor(calibrated, labels, raw_rows, regions, variant)
            check_analysis(calibrated, colored, header, analysis_header)
            analysis_path = temp_dir / name
            ts.write_ply(analysis_path, analysis_header, colored)
            analysis_sha = sha(analysis_path)
            record = sidecar_base(input_path, input_sha256, calibration_path, calibration_sha256,
                                  calibration, label_source, args.opacity_cutoff, args.include_residual,
                                  variant, region_counts, unlabeled, transform_record)
            record.update({
                'input': {'path': str(input_path), 'sha256': input_sha256, 'vertices': int(len(data))},
                'output': {'path': str(out_dir / name), 'sha256': analysis_sha, 'vertices': int(len(colored))},
                'display_only': False,
                'self_checks': {'vertex_count_and_header_properties_unchanged': True,
                                'only_color_properties_differ_from_calibrated_analysis': True,
                                'positions_scales_rotations_opacity_bitwise_equal': True},
            })
            (temp_dir / side_name).write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n',
                                              encoding='utf-8')
            written.append((name, analysis_sha))

            if args.viewer:
                display_name = '%s_leafcolor-%s-%s.ply' % (input_path.stem, variant, args.viewer)
                display_side_name = '%s_leafcolor-%s-%s.color.json' % (input_path.stem, variant, args.viewer)
                display, _ = ts.viewer_copy(colored, args.viewer)
                display_ply_header = display_header(analysis_header, args.viewer, name, display_side_name)
                if len(display) != len(colored) or property_signature(display_ply_header) != property_signature(header):
                    raise ValueError('Self-check failed: display vertex count or PLY properties changed')
                xyz = np.stack([colored['x'], colored['y'], colored['z']], axis=1).astype(np.float64)
                want_xyz = xyz @ np.asarray(ts.VIEWERS[args.viewer]['rotation'], np.float64).T
                want_xyz = want_xyz.astype(display.dtype['x'])
                got_xyz = np.stack([display['x'], display['y'], display['z']], axis=1)
                if not np.array_equal(got_xyz, want_xyz):
                    raise ValueError('Self-check failed: display positions do not match viewer rotation')
                display_path = temp_dir / display_name
                ts.write_ply(display_path, display_ply_header, display)
                display_sha = sha(display_path)
                display_record = sidecar_base(input_path, input_sha256, calibration_path,
                                              calibration_sha256, calibration, label_source,
                                              args.opacity_cutoff, args.include_residual, variant,
                                              region_counts, unlabeled, transform_record)
                display_record.update({
                    'output': {'path': str(out_dir / display_name), 'sha256': display_sha,
                               'vertices': int(len(display))},
                    'display_only': True, 'viewer': args.viewer,
                    'viewer_rotation': {'matrix': ts.VIEWERS[args.viewer]['rotation'],
                                        'description': ts.VIEWERS[args.viewer]['rotation_description'],
                                        'map': 'p_display = R @ p_analysis (no origin, no scale)'},
                    'analysis_file': {'path': str(out_dir / name), 'sha256': analysis_sha,
                                      'sidecar': side_name},
                    'self_checks': {'vertex_count_and_header_properties_unchanged': True,
                                    'positions_equal_viewer_rotation': True},
                })
                (temp_dir / display_side_name).write_text(
                    json.dumps(display_record, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
                written.append((display_name, display_sha))

        legend, legend_md = legend_documents(input_path, calibration_path, label_source,
                                             args.opacity_cutoff, args.include_residual,
                                             regions, region_counts, unlabeled, variants)
        (temp_dir / 'legend.json').write_text(json.dumps(legend, indent=2, sort_keys=True, allow_nan=False) + '\n',
                                              encoding='utf-8')
        (temp_dir / 'legend.md').write_text(legend_md, encoding='utf-8')
        verify_sources(expected_hashes)
        temp_dir.rename(out_dir)

    for name, output_hash in sorted(written):
        print('Wrote %s (%s)' % (out_dir / name, output_hash))
    print('Wrote %s and %s' % (out_dir / 'legend.md', out_dir / 'legend.json'))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, TypeError, pa.PlantAssetError) as exc:
        print('color-splat: ' + str(exc), file=sys.stderr)
        sys.exit(2)
