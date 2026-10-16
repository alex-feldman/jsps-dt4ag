#!/usr/bin/env python3
"""Cluster Gaussian-splat centers into candidate plants; write explicitly selected per-plant derivative PLYs. See README.md."""
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

CONFIG_KEYS = {'capture_id', 'inputs', 'parameters', 'source_ply_sha256', 'expected_cluster_sizes', 'derivatives'}
DERIVATIVE_KEYS = {'name', 'basis', 'clusters', 'rows', 'expand_unclustered_within'}
NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$')


def _number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _count(value):
    return type(value) is int and value >= 1


def spec_from(path):
    """Read and validate the explicit JSON config. Relative paths resolve against it."""
    path = Path(path).resolve()
    spec = read(path)
    if not isinstance(spec, dict) or set(spec) - CONFIG_KEYS:
        raise ValueError('Unknown config keys: ' + str(sorted(set(spec) - CONFIG_KEYS) if isinstance(spec, dict) else spec))
    if not isinstance(spec.get('capture_id'), str) or not spec['capture_id'].strip():
        raise ValueError('Expected a nonempty capture_id')
    inputs = spec.get('inputs')
    if not isinstance(inputs, dict) or set(inputs) != {'ply'} or not isinstance(inputs['ply'], str):
        raise ValueError('inputs must contain exactly "ply"')
    ply = (path.parent / inputs['ply']).resolve()
    if not ply.is_file():
        raise ValueError('Missing input ply: ' + str(ply))
    params = dict(pc.DEFAULTS)
    given = spec.get('parameters', {})
    if not isinstance(given, dict) or set(given) - set(params):
        raise ValueError('Unknown clustering parameter')
    params.update(given)
    if not _number(params['opacity_cutoff']) or not 0 < params['opacity_cutoff'] < 1:
        raise ValueError('Expected fraction in (0,1): opacity_cutoff')
    if not _number(params['radius_mult']) or params['radius_mult'] <= 0:
        raise ValueError('Expected positive radius_mult')
    for key in ['min_cluster_size', 'sweep_min_size', 'sweep_top', 'knn_fallback', 'max_gap_clusters']:
        if not _count(params[key]):
            raise ValueError('Expected positive integer ' + key)
    if (not isinstance(params['sweep_mults'], list) or not params['sweep_mults']
            or any(not _number(v) or v <= 0 for v in params['sweep_mults'])):
        raise ValueError('sweep_mults must be a nonempty list of positive numbers')
    if not isinstance(params['all_rows'], bool):
        raise ValueError('all_rows must be boolean')
    expected = spec.get('expected_cluster_sizes', {})
    if not isinstance(expected, dict) or any(not re.fullmatch(r'c\d+', k) or not _count(v) for k, v in expected.items()):
        raise ValueError('expected_cluster_sizes maps "c<id>" to a positive integer')
    derivatives = spec.get('derivatives', [])
    if not isinstance(derivatives, list):
        raise ValueError('derivatives must be a list')
    names, files = set(), []
    for d in derivatives:
        if not isinstance(d, dict) or set(d) - DERIVATIVE_KEYS:
            raise ValueError('Unknown derivative keys')
        if not isinstance(d.get('name'), str) or not NAME.match(d['name']) or d['name'] in names:
            raise ValueError('Each derivative needs a unique plain name (letters, digits, . _ -)')
        names.add(d['name'])
        if not isinstance(d.get('basis'), str) or not d['basis'].strip():
            raise ValueError('Derivative %s needs a basis: who or what evidence selected it' % d['name'])
        if 'clusters' not in d and 'rows' not in d:
            raise ValueError('Derivative %s selects nothing: give clusters and/or rows' % d['name'])
        clusters = d.get('clusters', [])
        if (not isinstance(clusters, list) or any(type(k) is not int or k < 0 for k in clusters)
                or len(set(clusters)) != len(clusters)):
            raise ValueError('Derivative %s clusters must be distinct nonnegative integers' % d['name'])
        if 'rows' in d:
            if isinstance(d['rows'], str):
                d['rows'] = str((path.parent / d['rows']).resolve())
                files.append(d['rows'])
            elif not isinstance(d['rows'], list) or any(type(k) is not int for k in d['rows']):
                raise ValueError('Derivative %s rows must be a list of integers or a .npy path' % d['name'])
        if 'expand_unclustered_within' in d and (not _number(d['expand_unclustered_within']) or d['expand_unclustered_within'] <= 0):
            raise ValueError('Derivative %s expand_unclustered_within must be positive' % d['name'])
    if derivatives and not isinstance(spec.get('source_ply_sha256'), str):
        raise ValueError('Derivatives require source_ply_sha256, so cluster ids cannot be applied to another model')
    spec['inputs'], spec['parameters'] = {'ply': str(ply)}, params
    return spec, files


def load_rows(value, count):
    if isinstance(value, str):
        arr = np.load(value, allow_pickle=False)
    else:
        arr = np.array(value, dtype=np.int64)
    return pc.check_rows(arr, count)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='Explicit input/parameter/selection JSON; see plant-clusters-example.json')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; existing paths are refused')
    parser.add_argument('--validate-only', action='store_true', help='Validate and hash inputs without creating output')
    args = parser.parse_args(argv)
    out = args.output.resolve()
    if out.exists():
        raise ValueError('Output already exists; choose a fresh directory: ' + str(out))
    spec, row_files = spec_from(args.config)
    ply = spec['inputs']['ply']
    hashes = {p: sha(p) for p in [ply] + row_files}
    if spec.get('source_ply_sha256') not in (None, hashes[ply]):
        raise ValueError('source_ply_sha256 does not match the input PLY; cluster ids belong to another model')
    meter = pc.StepMeter()
    with meter.step('read_ply'):
        data, XA, kept = pc.load_centers(ply, spec['parameters']['opacity_cutoff'])
        pc.ply_layout(ply, data.dtype)
    rows_in = {d['name']: load_rows(d['rows'], len(data)) for d in spec.get('derivatives', []) if 'rows' in d}
    if args.validate_only:
        print(json.dumps({'status': 'validated', 'rows': int(len(data)), 'opacity_kept_rows': int(len(kept)),
                          'derivatives': len(spec.get('derivatives', []))}))
        return 0
    out.mkdir(parents=True, exist_ok=False)
    record = {'version': VERSION, 'status': 'running', 'spec': spec, 'command': sys.argv,
              'working_directory': str(Path.cwd()), 'repository': source_revision(),
              'started_utc': datetime.now(timezone.utc).isoformat(), 'input_sha256': hashes,
              'code_sha256': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
              'versions': {k: importlib.metadata.version(k) for k in ['numpy', 'scipy']}}
    write(out / 'input-manifest.json', record)
    try:
        report, arrays, state = pc.analyze(XA, kept, spec['parameters'], meter)
        sizes = {k: v['size'] for k, v in report['clusters'].items()}
        wrong = {k: [v, sizes.get(k)] for k, v in spec.get('expected_cluster_sizes', {}).items() if sizes.get(k) != v}
        if wrong:
            raise ValueError('Cluster sizes differ from expected_cluster_sizes (expected, found): ' + json.dumps(wrong))
        report.update(capture_id=spec['capture_id'], source_ply=ply, source_ply_sha256=hashes[ply],
                      selection='none: clusters, sizes, gaps and join radii are reported; plant identity is an explicit input')
        write(out / 'clusters.json', report)
        with open(out / 'labels.npz', 'xb') as f:
            np.savez_compressed(f, **arrays)
        if spec.get('derivatives'):
            with meter.step('derivatives'):
                faint_info = pc.nearest_kept(XA, kept)
                outputs, chosen = [], {}
                for d in spec['derivatives']:
                    rows, result = pc.select_rows(state, kept, faint_info, XA, d, rows_in.get(d['name']))
                    digest = pc.write_derivative(ply, rows, out / (d['name'] + '.ply'), data.dtype)
                    with open(out / (d['name'] + '_rows.npy'), 'xb') as f:
                        np.save(f, rows)
                    result.update(name=d['name'], basis=d['basis'], ply=d['name'] + '.ply', sha256=digest,
                                  row_indices=d['name'] + '_rows.npy', row_indices_sha256=sha(out / (d['name'] + '_rows.npy')))
                    if d['name'] in rows_in:
                        result['rows_source'] = d['rows'] if isinstance(d['rows'], str) else 'inline list'
                    outputs.append(result)
                    chosen[d['name']] = rows
                names = list(chosen)
                overlaps = {'%s|%s' % (a, b): int(len(np.intersect1d(chosen[a], chosen[b])))
                            for i, a in enumerate(names) for b in names[i + 1:]}
            write(out / 'selection.json', {
                'capture_id': spec['capture_id'], 'source_ply': ply, 'source_ply_sha256': hashes[ply],
                'source_rows': int(len(data)), 'parameters': spec['parameters'],
                'median_nn_spacing_d1': state['d1'], 'radius': state['radius'], 'cluster_sizes': sizes,
                'status': 'derivatives written from an explicit selection; each basis records the identity decision, '
                          'which this command does not make',
                'faint_rule': 'a row with sigmoid(opacity) <= opacity_cutoff is attached when its nearest '
                              'opacity-kept center lies within radius and is selected',
                'row_copy': 'rows copied byte for byte by index from a read-only memmap; header identical except '
                            'the vertex count; re-read and compared',
                'derivatives': outputs, 'row_overlap_between_derivatives': overlaps})
        if any(sha(p) != digest for p, digest in hashes.items()):
            raise ValueError('Source input changed during clustering')
        record['status'] = 'complete'
        record['output_sha256'] = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'input-manifest.json'}
    except Exception as exc:
        record.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        record['ended_utc'] = datetime.now(timezone.utc).isoformat()
        record['steps'] = meter.steps
        record['peak_rss_kib_max_of_steps'] = meter.peak_kib()
        write(out / 'run-record.json', record)
    named = report['clusters']
    print('%d named clusters at r = %.6g (%g x d1): %s' % (len(named), state['radius'], spec['parameters']['radius_mult'],
          ', '.join('%s %d' % (k, v['size']) for k, v in named.items())))
    print('Complete: ' + str(out / 'clusters.json'))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print('cluster-plants: ' + str(exc), file=sys.stderr)
        sys.exit(2)
