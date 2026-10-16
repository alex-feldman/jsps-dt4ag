#!/usr/bin/env python3
"""Check a plant-asset folder against the versioned contract; exit 2 with the first violation."""
import argparse
import json
from pathlib import Path
import sys

import plant_asset as pa


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('asset', type=Path, help='plant-asset folder containing plant.json')
    args = parser.parse_args(argv)
    print(json.dumps(pa.validate_asset(args.asset)))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print('validate-plant-asset: ' + str(exc), file=sys.stderr)
        sys.exit(2)
