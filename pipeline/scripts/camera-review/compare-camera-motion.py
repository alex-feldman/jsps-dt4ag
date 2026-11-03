"""Compare adjacent camera motions in two COLMAP models; see README.md."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import struct

SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def index_names(names):
    result = {}
    for name in names:
        key = Path(name).with_suffix('').as_posix()
        if key in result:
            raise ValueError(f'ambiguous relative image stem: {key}')
        result[key] = name
    return result


def multiply(a, b):
    w, x, y, z = a
    W, X, Y, Z = b
    return (w*W-x*X-y*Y-z*Z, w*X+x*W+y*Z-z*Y,
            w*Y-x*Z+y*W+z*X, w*Z+x*Y-y*X+z*W)


def conjugate(q):
    return (q[0], -q[1], -q[2], -q[3])


def unit(q):
    n = math.sqrt(sum(x*x for x in q))
    if not math.isfinite(n) or n == 0:
        raise ValueError('invalid quaternion')
    return tuple(x/n for x in q)


def motion(a, b):
    return unit(multiply(b, conjugate(a)))


def rotation_difference(a, b):
    q = unit(multiply(a, conjugate(b)))
    return math.degrees(2 * math.atan2(math.sqrt(sum(x*x for x in q[1:])), abs(q[0])))


def center(pose):
    q, t = pose
    return tuple(-x for x in multiply(multiply(conjugate(q), (0, *t)), q)[1:])


def read_images(path):
    result = {}
    with path.open('rb') as f:
        def read(fmt):
            size = struct.calcsize('<' + fmt)
            data = f.read(size)
            if len(data) != size:
                raise ValueError(f'truncated image model: {path}')
            return struct.unpack('<' + fmt, data)
        n, = read('Q')
        for _ in range(n):
            v = read('i7di')
            name = bytearray()
            while True:
                ch = f.read(1)
                if ch == b'\0':
                    break
                if not ch:
                    raise ValueError(f'truncated image name: {path}')
                name.extend(ch)
            name = name.decode('utf-8')
            if name in result:
                raise ValueError(f'duplicate model image: {name}')
            q, t = unit(v[1:5]), v[5:8]
            if not all(math.isfinite(x) for x in t):
                raise ValueError(f'invalid translation: {name}')
            result[name] = (q, t)
            count, = read('Q')
            if len(f.read(count * 24)) != count * 24:
                raise ValueError(f'truncated observations: {name}')
    return result


def compare(expected, first, second):
    indexes = [index_names(x) for x in (expected, first, second)]
    groups = {}
    for name in sorted(expected):
        groups.setdefault(Path(name).parent.as_posix(), []).append(name)
    rows, refusals = [], []
    for folder, names in groups.items():
        for a, b in zip(names, names[1:]):
            keys = [Path(n).with_suffix('').as_posix() for n in (a, b)]
            if any(k not in idx for idx in indexes[1:] for k in keys):
                refusals.append({'folder': folder, 'image_a': a, 'image_b': b, 'reason': 'missing registration'})
                continue
            poses = [[model[idx[k]] for k in keys] for model, idx in zip((first, second), indexes[1:])]
            lengths = [math.dist(center(p[0]), center(p[1])) for p in poses]
            rows.append({'folder': folder, 'image_a': a, 'image_b': b,
                         'rotation_disagreement_deg': rotation_difference(*(motion(p[0][0], p[1][0]) for p in poses)),
                         'baseline_a': lengths[0], 'baseline_b': lengths[1]})
    for folder in groups:
        group = [r for r in rows if r['folder'] == folder]
        for side in ('a', 'b'):
            median = statistics.median(r['baseline_' + side] for r in group) if group else 0
            for r in group:
                length = r['baseline_' + side]
                r['normalized_' + side] = length / median if length > 0 and median > 0 else None
                if r['normalized_' + side] is None:
                    refusals.append({'folder': folder, 'image_a': r['image_a'], 'image_b': r['image_b'],
                                     'reason': f'zero baseline or folder median in model {side}'})
        for rank, r in enumerate(sorted(group, key=lambda x: x['rotation_disagreement_deg'], reverse=True), 1):
            r['folder_rank'] = rank
    return rows, refusals, {'missing_a': sorted(indexes[0].keys() - indexes[1].keys()),
                            'missing_b': sorted(indexes[0].keys() - indexes[2].keys()),
                            'unexpected_a': sorted(indexes[1].keys() - indexes[0].keys()),
                            'unexpected_b': sorted(indexes[2].keys() - indexes[0].keys())}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--workspace-a', type=Path, required=True)
    p.add_argument('--workspace-b', type=Path, required=True)
    p.add_argument('--model-a', default='0')
    p.add_argument('--model-b', default='0')
    p.add_argument('--out', type=Path, required=True, help='fresh output directory')
    args = p.parse_args(argv)
    root = args.capture / 'images'
    if not root.is_dir():
        p.error('capture must contain images/')
    names = sorted(x.relative_to(root).as_posix() for x in root.rglob('*') if x.is_file() and x.suffix.lower() in SUFFIXES)
    if not names:
        p.error('capture has no expected images')
    first = read_images(args.workspace_a / 'sparse' / args.model_a / 'images.bin')
    second = read_images(args.workspace_b / 'sparse' / args.model_b / 'images.bin')
    if not first or not second:
        p.error('both models must contain registered images')
    rows, refusals, names_report = compare(names, first, second)
    args.out.mkdir(parents=True, exist_ok=False)
    def write_csv(name, data, fields):
        with (args.out / name).open('w', newline='', encoding='utf8') as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(data)
    write_csv('edges.csv', rows, ['folder', 'folder_rank', 'image_a', 'image_b', 'rotation_disagreement_deg', 'baseline_a', 'baseline_b', 'normalized_a', 'normalized_b'])
    write_csv('refusals.csv', refusals, ['folder', 'image_a', 'image_b', 'reason'])
    meta = {'capture': str(args.capture), 'workspace_a': str(args.workspace_a), 'workspace_b': str(args.workspace_b),
            'model_a': args.model_a, 'model_b': args.model_b, 'expected_count': len(names), 'eligible_edges': len(rows),
            'ordering': 'lexical expected relative names within each folder; chronology assumed',
            'limits': 'No ground truth or acceptance decision. Raw baselines have arbitrary independent model scales.', **names_report}
    (args.out / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf8')
    print(f"{len(rows)} eligible edges; {len(refusals)} refusals; results: {args.out}")


if __name__ == '__main__':
    main()
