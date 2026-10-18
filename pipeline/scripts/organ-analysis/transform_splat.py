#!/usr/bin/env python3
"""Transform a 3DGS splat export PLY by a frame calibration, without retraining.

Every Gaussian is mapped by p' = s * R @ (p - o):
- centers x, y, z by the full map; normals nx, ny, nz (when present) by R;
- quaternions rot_0..rot_3 (w, x, y, z, the gsplat order) by q' = q_R * q, which keeps their norm;
- log scales scale_0..scale_2 plus log(s) when the scale is applied;
- spherical-harmonic f_rest coefficients rotated band by band in the real SH basis gsplat evaluates
  (gsplat _eval_sh_bases_fast), with nerfstudio's channel-major export layout f_rest_{c*K + k}
  (k = 0..K-1 is basis index k+1, c = 0..2 the color channel);
- f_dc, opacity and every other property unchanged.
Header element/property order is kept. The output is always a new file, with a JSON sidecar
recording input, output and calibration hashes and the settings. See README.md.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

from analyze_organs import VERSION, sha
import frame_calibration as fc

SIDECAR_SCHEMA = 'splat-transform'
SIDECAR_SCHEMA_VERSION = 1
PLY_TYPES = {'float': '<f4', 'float32': '<f4', 'double': '<f8', 'float64': '<f8', 'uchar': 'u1', 'uint8': 'u1',
             'char': 'i1', 'int8': 'i1', 'ushort': '<u2', 'uint16': '<u2', 'short': '<i2', 'int16': '<i2',
             'int': '<i4', 'int32': '<i4', 'uint': '<u4', 'uint32': '<u4'}
SH_FIT_DIRECTIONS = 4096
SH_FIT_SEED = 20261018
SH_RESIDUAL_TOL = 1e-10


# ----------------------------------------------------------------------------- PLY

def read_ply(path):
    """(header lines as str without newline, structured vertex array). Binary little endian,
    one vertex element of scalar properties only."""
    with open(path, 'rb') as f:
        lines = []
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError('Truncated PLY header')
            line = raw.decode('ascii').rstrip('\r\n')
            lines.append(line)
            if line.strip() == 'end_header':
                break
        if lines[0] != 'ply':
            raise ValueError('Not a PLY file')
        props, count, fmt, elements = [], None, None, 0
        for line in lines[1:]:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'format':
                fmt = parts[1]
            elif parts[0] == 'element':
                elements += 1
                if parts[1] != 'vertex':
                    raise ValueError('Only vertex-only splat PLY files are supported')
                count = int(parts[2])
            elif parts[0] == 'property':
                if len(parts) != 3 or parts[1] not in PLY_TYPES:
                    raise ValueError('Unsupported PLY property: ' + line)
                props.append((parts[2], PLY_TYPES[parts[1]]))
        if fmt != 'binary_little_endian' or elements != 1 or not count or not props:
            raise ValueError('Expected one nonempty binary_little_endian vertex element')
        data = np.fromfile(f, dtype=np.dtype(props), count=count)
        if len(data) != count or f.read(1):
            raise ValueError('PLY body length does not match the header')
    return lines, data


def write_ply(path, lines, data):
    with open(path, 'xb') as f:
        f.write(('\n'.join(lines) + '\n').encode('ascii'))
        f.write(data.tobytes())


# ----------------------------------------------------------------------------- SH basis (gsplat)

def sh_basis(dirs, degree):
    """Real SH basis exactly as gsplat _eval_sh_bases_fast (Sloan 2013) at unit directions (N,3);
    returns (N, (degree+1)^2). Degrees 0 to 4."""
    d = np.asarray(dirs, np.float64)
    n = (degree + 1) ** 2
    out = np.empty((len(d), n))
    out[:, 0] = 0.2820947917738781
    if degree < 1:
        return out
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    a = -0.48860251190292
    out[:, 2] = -a * z
    out[:, 3] = a * x
    out[:, 1] = a * y
    if degree < 2:
        return out
    z2 = z * z
    b = -1.092548430592079 * z
    a = 0.5462742152960395
    c1 = x * x - y * y
    s1 = 2 * x * y
    out[:, 6] = 0.9461746957575601 * z2 - 0.3153915652525201
    out[:, 7] = b * x
    out[:, 5] = b * y
    out[:, 8] = a * c1
    out[:, 4] = a * s1
    if degree < 3:
        return out
    c = -2.285228997322329 * z2 + 0.4570457994644658
    b = 1.445305721320277 * z
    a = -0.5900435899266435
    c2 = x * c1 - y * s1
    s2 = x * s1 + y * c1
    out[:, 12] = z * (1.865881662950577 * z2 - 1.119528997770346)
    out[:, 13] = c * x
    out[:, 11] = c * y
    out[:, 14] = b * c1
    out[:, 10] = b * s1
    out[:, 15] = a * c2
    out[:, 9] = a * s2
    if degree < 4:
        return out
    dd = z * (-4.683325804901025 * z2 + 2.007139630671868)
    c = 3.31161143515146 * z2 - 0.47308734787878
    b = -1.770130769779931 * z
    a = 0.6258357354491763
    c3 = x * c2 - y * s2
    s3 = x * s2 + y * c2
    out[:, 20] = 1.984313483298443 * z2 * (1.865881662950577 * z2 - 1.119528997770346) + \
        -1.006230589874905 * (0.9461746957575601 * z2 - 0.3153915652525201)
    out[:, 21] = dd * x
    out[:, 19] = dd * y
    out[:, 22] = c * c1
    out[:, 18] = c * s1
    out[:, 23] = b * c2
    out[:, 17] = b * s2
    out[:, 24] = a * c3
    out[:, 16] = a * s3
    return out


def sample_directions(n=SH_FIT_DIRECTIONS, seed=SH_FIT_SEED):
    v = np.random.default_rng(seed).normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def sh_rotation(R, degree, n=SH_FIT_DIRECTIONS, seed=SH_FIT_SEED):
    """Per-band matrices D_l with B_l(R d) @ D_l @ c = B_l(d) @ c, i.e. new coefficients
    c' = D_l c give the rotated color function f'(R d) = f(d). Fitted by least squares over n
    random unit directions with the exact basis; returns (list of D_l for l = 1..degree, fit record)."""
    R = np.asarray(R, np.float64)
    d = sample_directions(n, seed)
    B, BR = sh_basis(d, degree), sh_basis(d @ R.T, degree)
    mats, rec = [], {'directions': n, 'seed': seed, 'bands': []}
    for l in range(1, degree + 1):
        sl = slice(l * l, (l + 1) * (l + 1))
        D, *_ = np.linalg.lstsq(BR[:, sl], B[:, sl], rcond=None)
        resid = float(np.abs(BR[:, sl] @ D - B[:, sl]).max())
        orth = float(np.abs(D @ D.T - np.eye(2 * l + 1)).max())
        if resid > SH_RESIDUAL_TOL:
            raise ValueError('SH band %d rotation fit residual %.3g exceeds %g' % (l, resid, SH_RESIDUAL_TOL))
        mats.append(D)
        rec['bands'].append({'band': l, 'max_fit_residual': resid, 'max_orthogonality_deviation': orth})
    return mats, rec


# ----------------------------------------------------------------------------- quaternions

def rotmat_to_quat(R):
    """Unit quaternion (w, x, y, z), w >= 0, of a rotation matrix (Shepperd's method)."""
    R = np.asarray(R, np.float64)
    t = np.trace(R)
    if t > 0:
        w = math.sqrt(1 + t) / 2
        q = [w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        r = math.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / (2 * r)
        q[1 + i] = r / 2
        q[1 + j] = (R[j, i] + R[i, j]) / (2 * r)
        q[1 + k] = (R[k, i] + R[i, k]) / (2 * r)
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def quat_multiply(a, b):
    """Hamilton product a * b, both (..., 4) in (w, x, y, z)."""
    aw, ax, ay, az = np.moveaxis(np.asarray(a, np.float64), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, np.float64), -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], -1)


def quat_to_rotmat(q):
    """gsplat _quat_to_rotmat: normalize, then (w, x, y, z) to (..., 3, 3)."""
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x ** 2 + y ** 2)], -1).reshape(q.shape[:-1] + (3, 3))


# ----------------------------------------------------------------------------- transform

def _names(data, prefix):
    names = [n for n in data.dtype.names if n.startswith(prefix) and n[len(prefix):].isdigit()]
    return sorted(names, key=lambda n: int(n[len(prefix):]))


def layout(data):
    """Check the splat properties this transform needs and return the SH layout."""
    names = set(data.dtype.names)
    need = {'x', 'y', 'z', 'rot_0', 'rot_1', 'rot_2', 'rot_3', 'scale_0', 'scale_1', 'scale_2'}
    if not need <= names:
        raise ValueError('Missing splat properties: ' + ', '.join(sorted(need - names)))
    rest = _names(data, 'f_rest_')
    if [int(n[7:]) for n in rest] != list(range(len(rest))):
        raise ValueError('f_rest properties must be f_rest_0..f_rest_{n-1}')
    if len(rest) % 3:
        raise ValueError('f_rest count %d is not a multiple of 3 channels' % len(rest))
    K = len(rest) // 3
    degree = int(round(math.sqrt(K + 1))) - 1
    if (degree + 1) ** 2 != K + 1 or degree > 4:
        raise ValueError('f_rest count %d does not match an SH degree up to 4' % len(rest))
    return {'sh_degree': degree, 'coefficients_per_channel': K, 'f_rest_layout': 'channel-major f_rest_{c*K+k}',
            'has_normals': {'nx', 'ny', 'nz'} <= names}


def transform(data, R, o, s, sh_fit=None):
    """New structured array with every Gaussian mapped by p' = s R (p - o). Arithmetic in float64,
    cast back to each property's stored dtype. Returns (array, record)."""
    R = np.asarray(R, np.float64)
    o = np.asarray(o, np.float64)
    s = float(s)
    if R.shape != (3, 3) or np.abs(R @ R.T - np.eye(3)).max() > 1e-9 or abs(np.linalg.det(R) - 1) > 1e-9:
        raise ValueError('R must be a proper rotation')
    if not (math.isfinite(s) and s > 0):
        raise ValueError('scale must be finite and > 0')
    info = layout(data)
    out = data.copy()
    P = np.stack([data['x'], data['y'], data['z']], 1).astype(np.float64)
    Q = s * ((P - o) @ R.T)
    for i, k in enumerate('xyz'):
        out[k] = Q[:, i]
    if info['has_normals']:
        N = np.stack([data['nx'], data['ny'], data['nz']], 1).astype(np.float64) @ R.T
        for i, k in enumerate(('nx', 'ny', 'nz')):
            out[k] = N[:, i]
    q = np.stack([data['rot_%d' % i] for i in range(4)], 1).astype(np.float64)
    qR = rotmat_to_quat(R)
    q2 = quat_multiply(qR[None], q)
    for i in range(4):
        out['rot_%d' % i] = q2[:, i]
    for i in range(3):
        out['scale_%d' % i] = data['scale_%d' % i].astype(np.float64) + math.log(s)
    rec = dict(info, quaternion_order='wxyz (gsplat _quat_to_rotmat)', quaternion_rule="q' = q_R * q (Hamilton)",
               q_R=[float(v) for v in qR], log_scale_offset=math.log(s))
    if info['sh_degree'] > 0:
        mats, fit = sh_fit if sh_fit is not None else sh_rotation(R, info['sh_degree'])
        K = info['coefficients_per_channel']
        for c in range(3):
            C = np.stack([data['f_rest_%d' % (c * K + k)] for k in range(K)], 1).astype(np.float64)
            C2 = C.copy()
            for l, D in enumerate(mats, start=1):
                sl = slice(l * l - 1, (l + 1) * (l + 1) - 1)
                C2[:, sl] = C[:, sl] @ D.T
            for k in range(K):
                out['f_rest_%d' % (c * K + k)] = C2[:, k]
        rec['sh_rotation_fit'] = fit
    return out, rec


def transformed_properties(data):
    info = layout(data)
    names = ['x', 'y', 'z'] + (['nx', 'ny', 'nz'] if info['has_normals'] else [])
    names += ['rot_%d' % i for i in range(4)] + ['scale_%d' % i for i in range(3)] + _names(data, 'f_rest_')
    return set(names)


def parameters(cal, use_scale=True, inverse=False):
    """(R, o, s) of the calibration map, or of its inverse (p = (1/s) R^T p' + o)."""
    R = np.array(cal['rotation']['matrix'], np.float64) if cal.get('rotation') else np.eye(3)
    o = np.array(cal['origin']['point_source_frame'], np.float64) if cal.get('origin') else np.zeros(3)
    s = float(cal['scale']['meters_per_scene_unit']) if (use_scale and cal.get('scale')) else 1.0
    if inverse:
        return R.T, -s * (R @ o), 1.0 / s
    return R, o, s


def updated_header(lines, cal, inverse, use_scale, sidecar_name):
    """Same element and property lines; the nerfstudio 'Vertical Axis' comment is updated to the
    target up axis and one provenance comment is added after the format line."""
    up = cal['target_frame'].get('up', fc.DEFAULT_UP)[-1]
    out = []
    for line in lines:
        if line.startswith('comment Vertical Axis:') and not inverse:
            line = 'comment Vertical Axis: ' + up
        out.append(line)
        if line.startswith('format '):
            out.append('comment jsps-dt4ag %s transform_splat: %s frame, %s; sidecar %s' % (
                VERSION, 'dataparser' if inverse else 'plant', 'meters' if (use_scale and cal.get('scale')) else
                'scene units', sidecar_name))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--calibration', required=True, type=Path)
    ap.add_argument('--input', required=True, type=Path, help='splat export PLY (read only)')
    ap.add_argument('--out', required=True, type=Path, help='new PLY; the sidecar is OUT with .transform.json')
    ap.add_argument('--no-scale', action='store_true', help='rotation and origin only; keep scene units')
    ap.add_argument('--inverse', action='store_true', help='map plant-frame splats back to the source frame')
    args = ap.parse_args(argv)
    cal = fc.load(args.calibration)
    src, out = args.input.resolve(), args.out.resolve()
    side = out.with_name(out.name[:-4] + '.transform.json' if out.suffix == '.ply' else out.name + '.transform.json')
    if out.suffix != '.ply':
        raise ValueError('Output must end in .ply')
    if out == src:
        raise ValueError('transform_splat never writes in place')
    for p in (out, side):
        if p.exists():
            raise ValueError('Output already exists; choose a new path: ' + str(p))
    if not out.parent.is_dir():
        raise ValueError('Output directory does not exist: ' + str(out.parent))
    use_scale = not args.no_scale
    R, o, s = parameters(cal, use_scale, args.inverse)
    in_sha = sha(src)
    lines, data = read_ply(src)
    new, rec = transform(data, R, o, s)
    header = updated_header(lines, cal, args.inverse, use_scale, side.name)
    write_ply(out, header, new)
    if sha(src) != in_sha:
        raise ValueError('Input changed during the run')
    units = cal['target_frame']['units'] if use_scale else fc.SCENE_UNITS
    sidecar = {'schema': SIDECAR_SCHEMA, 'schema_version': SIDECAR_SCHEMA_VERSION,
               'generator': {'name': fc.GENERATOR, 'version': VERSION, 'command': 'transform_splat.py'},
               'input': {'path': str(src), 'sha256': in_sha, 'vertices': int(len(data))},
               'output': {'path': str(out), 'sha256': sha(out), 'vertices': int(len(new))},
               'calibration': {'path': str(args.calibration.resolve()), 'sha256': sha(args.calibration),
                               'capture_id': cal['capture_id']},
               'settings': {'use_scale': use_scale, 'inverse': args.inverse, 'R': [[float(v) for v in r] for r in R],
                            'o': [float(v) for v in o], 's': s,
                            'map': "p' = s * R @ (p - o)"},
               'frame': ({'name': 'dataparser', 'units': fc.SCENE_UNITS} if args.inverse else
                         {'name': cal['target_frame']['name'], 'up': cal['target_frame'].get('up', fc.DEFAULT_UP),
                          'units': units}),
               'header_changes': [l for l in header if l not in lines],
               'properties': list(data.dtype.names), 'transform': rec,
               'unchanged_properties': [n for n in data.dtype.names if n not in transformed_properties(data)]}
    with side.open('x') as f:
        f.write(json.dumps(sidecar, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print('Wrote %s (%d Gaussians, %s frame, %s) and %s' % (out, len(new), sidecar['frame']['name'], sidecar['frame']['units'], side))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print('transform-splat: ' + str(exc), file=sys.stderr)
        sys.exit(2)
