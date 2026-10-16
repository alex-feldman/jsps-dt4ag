#!/usr/bin/env python3
"""Export one leaf of a plant-asset package, with its petiole and attachment, to a fresh folder.

``--reload`` checks an existing export folder or a viewer JSON download instead.
The source asset is validated and only read. See README.md ("Plant asset package").
"""
import argparse
import json
from pathlib import Path
import sys

import plant_asset as pa


def summary(document, arrays):
    petiole = document['petiole']
    return {'leaf': document['leaf']['id'], 'role': document['leaf']['role'], 'points': int(len(arrays['X'])),
            'petiole': None if petiole is None else petiole['id'], 'petiole_status': document['petiole_status'],
            'petiole_geometry_status': None if petiole is None else petiole['geometry_status'],
            'attachment': None if document['attachment'] is None else document['attachment']['id'],
            'units': document['units'], 'export_route': document['export_route']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path, help='plant-asset folder containing plant.json')
    parser.add_argument('--leaf', help='Leaf ID such as welzo-M1:leaf:2, or its source cluster integer')
    parser.add_argument('--output', type=Path, help='New export folder; existing paths are refused')
    parser.add_argument('--reload', type=Path, help='Check an export folder or viewer-downloaded JSON and print it')
    args = parser.parse_args(argv)
    if args.reload is not None:
        if args.asset or args.leaf or args.output:
            parser.error('--reload takes no other arguments')
        print(json.dumps(summary(*pa.load_leaf_export(args.reload))))
        return 0
    if not (args.asset and args.leaf is not None and args.output):
        parser.error('--asset, --leaf and --output are required for an export')
    pa.export_leaf(args.asset, args.leaf, args.output)
    print(json.dumps(summary(*pa.load_leaf_export(args.output))))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError) as exc:
        print('export-leaf: ' + str(exc), file=sys.stderr)
        sys.exit(2)
