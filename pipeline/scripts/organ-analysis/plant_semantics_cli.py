#!/usr/bin/env python3
"""Build, validate or query an experimental source-bound partial semantic package."""
import argparse
import json
import sys
from pathlib import Path
import plant_semantics as ps


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    build = sub.add_parser('build')
    build.add_argument('--source', type=Path, required=True)
    build.add_argument('--output', type=Path, required=True)
    build.add_argument('--registry', type=Path)
    check = sub.add_parser('validate')
    check.add_argument('package', type=Path)
    query = sub.add_parser('query')
    query.add_argument('package', type=Path)
    query.add_argument('mode', choices=['candidate-groups', 'candidate-leaves', 'topology'])
    query.add_argument('--id')
    args = parser.parse_args(argv)
    if args.action == 'build':
        result = ps.build(args.source, args.output, args.registry)
    elif args.action == 'validate':
        result = ps.validate(args.package)
    else:
        result = ps.query(args.package, args.mode, args.id)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print('plant-semantics: ' + str(exc), file=sys.stderr)
        sys.exit(2)
