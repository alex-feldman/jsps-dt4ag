#!/usr/bin/env python3
"""Render an organ-analysis viewer from existing outputs, optionally with reviewed leaf groups.

Reads a completed leaf-stage, skeleton-stage or branch-review folder and writes only a new
viewer folder. Source labels, evidence, graph and measurements are read, never changed.
"""
import argparse
from pathlib import Path
import sys

from analyze_organs import read, write, sha, source_revision, VERSION
from viewer import check_leaf_groups, write_viewer


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True,
                   help='Existing output folder holding organ-evidence.json and leaf_ids2.npz')
    p.add_argument('--output', type=Path, required=True, help='Fresh viewer folder')
    p.add_argument('--candidates', action='store_true',
                   help='Also show branch-candidates.json from the source folder (branch-review output)')
    p.add_argument('--leaf-groups', type=Path, help='Human-reviewed leaf-group record (organ-leaf-groups v1)')
    args = p.parse_args(argv)
    source, out = args.source.resolve(), args.output.resolve()
    if out.exists() or source == out or source in out.parents:
        raise ValueError('Choose a fresh output directory outside the source folder')
    evidence, labels = source/'organ-evidence.json', source/'leaf_ids2.npz'
    bundle = read(evidence)
    inputs = [evidence, labels]
    candidates = None
    if args.candidates:
        inputs.append(source/'branch-candidates.json')
        candidates = read(inputs[-1])
    groups = None
    if args.leaf_groups:
        inputs.append(args.leaf_groups.resolve())
        groups = check_leaf_groups(read(inputs[-1]), bundle, labels)
    before = {str(path): sha(path) for path in inputs}
    out.mkdir(parents=True, exist_ok=False)
    write_viewer(out, bundle, candidates=candidates, labels_path=labels, leaf_groups=groups)
    after = {str(path): sha(path) for path in inputs}
    if before != after:
        raise ValueError('An input changed while rendering')
    write(out/'viewer-record.json', {
        'schema': 'organ-viewer-render', 'schema_version': 1, 'tool_version': VERSION,
        'capture_id': bundle['capture_id'], 'repository': source_revision(), 'command': sys.argv,
        'viewer_code_sha256': sha(Path(__file__).with_name('viewer.py')),
        'inputs': [{'path': path, 'sha256': digest} for path, digest in before.items()],
        'leaf_groups': None if groups is None else {'path': str(args.leaf_groups.resolve()),
                                                    'sha256': before[str(args.leaf_groups.resolve())]},
        'clusters': [leaf['id'] for leaf in bundle['leaves']],
        'viewer': {'path': 'viewer.html', 'sha256': sha(out/'viewer.html')},
        'source_unchanged': True})
    print('Viewer: '+str(out/'viewer.html'))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError) as exc:
        print('render-viewer: '+str(exc), file=sys.stderr)
        sys.exit(2)
