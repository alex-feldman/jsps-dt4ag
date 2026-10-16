"""Scene up direction, per-plant vertical axes and detached-cluster assignment by stacking.

Algorithmic only: every selection follows from the recorded settings below, and nothing is
chosen by looking at images. The stage reports and records; it never edits a derivative.
See DESIGN.md for the method and its provenance, README.md for the command
(orient_plants.py). Distances are in the export's scale-free scene units (dataparser frame).
"""
import json
import math
import struct
from pathlib import Path

import numpy as np

from geometry import camera_to_world, dataparser_frame

# Experimental defaults, first set on 2026-10-17 from the Welzo M1 captures (October 2 soil
# pilot and October 15 separation). They are not universal tuning; a run records the values used.
DEFAULTS = {
    'opacity_cutoff': 0.3,         # sigmoid(opacity) must exceed this (organ-analysis rule)
    'min_cameras': 8,              # camera ring: fewer registered cameras is refused
    'max_planarity': 0.25,         # camera ring: RMS distance from the plane / ring radius
    'min_sign_cos': 0.1,           # camera ring: |cos| of mean image-up to the normal, below which the sign is ambiguous
    'soil_min_hits': 2,            # soil: mask hits at a point's own track observations
    'soil_hit_share': 0.6,         # soil: hits / available tracked views
    'soil_local_radius': 0.3,      # soil: distance from the base estimate
    'ransac_seed': 42,
    'ransac_trials': 1000,
    'plane_tol': 0.004,            # RANSAC inlier distance
    'min_plane_points': 30,        # soil candidates (after coordinate deduplication)
    'min_inliers': 30,
    'max_up_angle_deg': 10.0,      # A versus B agreement threshold
    'slice': 0.02,                 # axis slice thickness along up
    'min_slice_points': 30,
    'radius_percentile': 95.0,
    'detached_max_share': 0.05,    # automatic detached clusters: size below this share of the largest named cluster
    'assign_max_ratio': 1.0,       # r/R at most this for the assigned plant
    'other_min_ratio': 2.0,        # r/R at least this for every other plant
    'height_margin_slices': 1,     # height-range margin, in slices
}
INTEGER_KEYS = {'min_cameras', 'soil_min_hits', 'ransac_seed', 'ransac_trials', 'min_plane_points', 'min_inliers',
                'min_slice_points', 'height_margin_slices'}


def check_parameters(params):
    for key, value in params.items():
        if key in INTEGER_KEYS:
            if type(value) is not int or value < (0 if key in ('ransac_seed', 'height_margin_slices') else 1):
                raise ValueError('Expected nonnegative or positive integer ' + key)
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Expected positive number ' + key)
    for key in ['opacity_cutoff', 'soil_hit_share', 'detached_max_share', 'min_sign_cos']:
        if not params[key] < 1:
            raise ValueError('Expected fraction in (0,1): ' + key)
    if not params['radius_percentile'] <= 100:
        raise ValueError('radius_percentile must be at most 100')
    if params['soil_min_hits'] < 1 or params['min_plane_points'] < 3 or params['min_inliers'] < 3:
        raise ValueError('Plane fits need at least three points')
    if params['other_min_ratio'] < params['assign_max_ratio']:
        raise ValueError('other_min_ratio must be at least assign_max_ratio')
    return params


def unit(v):
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError('Zero or nonfinite vector')
    return v / n


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(unit(a), unit(b)), -1.0, 1.0))))


def floats(v):
    return [float(x) for x in np.asarray(v).ravel()]


# ----------------------------------------------------------------------------- up, method A

def camera_poses(transforms, dataparser):
    """(frame stems, centers (N,3), image-up vectors (N,3)) of every transforms.json frame in the
    dataparser frame. Image up is minus the OpenCV camera y axis."""
    t, d = json.loads(Path(transforms).read_text()), json.loads(Path(dataparser).read_text())
    T, scale = dataparser_frame(t, d)
    stems, C, U = [], [], []
    for fr in t['frames']:
        c2w = camera_to_world(T, scale, fr)
        if not np.isfinite(c2w).all():
            raise ValueError('Nonfinite camera pose')
        stems.append(Path(fr['file_path']).stem)
        C.append(c2w[:3, 3])
        U.append(-c2w[:3, 1])
    return stems, np.array(C).reshape(-1, 3), np.array(U).reshape(-1, 3)


def camera_ring_up(C, U, params):
    """Method A: normal of the SVD plane through the camera centers, signed by the mean image-up.
    Returns a record whose status is 'ok' or 'refused' with a reason; never a guess."""
    rec = {'method': 'camera_ring', 'cameras': int(len(C)), 'normal': None}
    if len(C) < params['min_cameras']:
        return dict(rec, status='refused', reason='%d cameras; at least min_cameras = %d required'
                    % (len(C), params['min_cameras']))
    center = C.mean(0)
    _, sv, vt = np.linalg.svd(C - center, full_matrices=False)
    n = vt[2]
    dist = (C - center) @ n
    inplane = (C - center) - dist[:, None] * n
    ring_radius = float(np.linalg.norm(inplane, axis=1).mean())
    rms = float(np.sqrt(np.mean(dist ** 2)))
    planarity = rms / ring_radius if ring_radius > 0 else math.inf
    mean_up = U.mean(0)
    mean_up_norm = float(np.linalg.norm(mean_up))
    cos = float(np.dot(n, mean_up) / mean_up_norm) if mean_up_norm > 0 else 0.0
    rec.update(centroid=floats(center), singular_values=floats(sv), ring_radius=ring_radius,
               rms_distance_from_plane=rms, max_distance_from_plane=float(np.abs(dist).max()),
               planarity=planarity, mean_image_up=floats(mean_up), mean_image_up_length=mean_up_norm,
               sign_cos=abs(cos))
    if not planarity <= params['max_planarity']:
        return dict(rec, status='refused', reason='cameras are not ring-like: planarity %.4g > max_planarity %.4g'
                    % (planarity, params['max_planarity']))
    if abs(cos) < params['min_sign_cos']:
        return dict(rec, status='refused', reason='sign ambiguous: mean image-up is nearly in the ring plane '
                    '(|cos| %.4g < min_sign_cos %.4g)' % (abs(cos), params['min_sign_cos']))
    n = n if cos > 0 else -n
    rec.update(status='ok', normal=floats(n), angle_to_mean_image_up_deg=angle_deg(n, mean_up),
               angle_to_dataparser_z_deg=angle_deg(n, [0, 0, 1]))
    return rec


# ----------------------------------------------------------------------------- up, method B

def read_points3d_binary(path):
    """COLMAP points3D.bin: {id: (xyz, image_ids, point2D_idxs)}."""
    buf = Path(path).read_bytes()
    off, out = 8, {}
    (count,) = struct.unpack_from('<Q', buf, 0)
    for _ in range(count):
        pid, x, y, z = struct.unpack_from('<Qddd', buf, off)
        off += 8 + 24 + 3 + 8
        (n,) = struct.unpack_from('<Q', buf, off)
        off += 8
        track = np.frombuffer(buf, dtype='<i4', count=2 * n, offset=off).reshape(n, 2)
        off += 8 * n
        out[pid] = (np.array([x, y, z]), track[:, 0].copy(), track[:, 1].copy())
    if off != len(buf):
        raise ValueError('Unexpected trailing bytes in ' + str(path))
    return out


def read_images_binary(path):
    """COLMAP images.bin: {image_id: (name, xys (M,2))}."""
    buf = Path(path).read_bytes()
    off, out = 8, {}
    (count,) = struct.unpack_from('<Q', buf, 0)
    for _ in range(count):
        (iid,) = struct.unpack_from('<i', buf, off)
        off += 4 + 32 + 24 + 4
        end = buf.index(b'\0', off)
        name = buf[off:end].decode()
        off = end + 1
        (n,) = struct.unpack_from('<Q', buf, off)
        off += 8
        rec = np.frombuffer(buf, dtype=np.dtype([('x', '<f8'), ('y', '<f8'), ('id', '<i8')]), count=n, offset=off)
        off += 24 * n
        out[iid] = (name, np.stack([rec['x'], rec['y']], 1))
    if off != len(buf):
        raise ValueError('Unexpected trailing bytes in ' + str(path))
    return out


def colmap_to_dataparser(xyz, d):
    """Raw COLMAP points (no applied_transform) to the dataparser frame: the saved transform
    already includes applied_transform, then scale multiplies the result including translation."""
    T = np.array(d['transform'], np.float64)
    return (xyz @ T[:3, :3].T + T[:3, 3]) * float(d['scale'])


def soil_votes(points, images, name_by_image_id, soil_masks):
    """Soil votes at each point's own feature-track observations, as in the October 2 pilot.
    soil_masks maps a source name to a boolean mask (plant pixels already removed). A view
    without a soil mask is unknown, not a negative. Each view votes at most once per point."""
    ids = list(points)
    hits, seen = np.zeros(len(ids), int), np.zeros(len(ids), int)
    for row, pid in enumerate(ids):
        _, image_ids, idxs = points[pid]
        voted = set()
        for iid, j in zip(image_ids.tolist(), idxs.tolist()):
            name = name_by_image_id.get(iid)
            if name not in soil_masks or name in voted:
                continue
            m = soil_masks[name]
            x, y = images[iid][1][j]
            x, y = int(x), int(y)
            if not (0 <= x < m.shape[1] and 0 <= y < m.shape[0]):
                continue
            voted.add(name)
            seen[row] += 1
            hits[row] += int(m[y, x])
    return np.array(ids, dtype=np.int64), hits, seen


def ransac_plane(P, params):
    """Deterministic three-point RANSAC (the pilot's loop), then SVD refinement of the largest
    consensus. Returns (normal, center, inlier mask) or (None, None, inliers) below min_inliers."""
    inliers = np.zeros(len(P), bool)
    rng = np.random.default_rng(params['ransac_seed'])
    for _ in range(params['ransac_trials']):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-10:
            continue
        n /= norm
        good = np.abs((P - a) @ n) < params['plane_tol']
        if good.sum() > inliers.sum():
            inliers = good
    if inliers.sum() < params['min_inliers']:
        return None, None, inliers
    center = P[inliers].mean(0)
    _, _, vt = np.linalg.svd(P[inliers] - center, full_matrices=False)
    return vt[-1], center, inliers


def soil_plane_up(SX, hits, seen, base, toward, params):
    """Method B from mapped COLMAP points SX with soil votes: select, restrict to soil_local_radius
    of base, deduplicate coordinates for fitting, RANSAC, SVD; sign toward the point `toward`."""
    soil = (hits >= params['soil_min_hits']) & (hits / np.maximum(seen, 1) >= params['soil_hit_share'])
    local = soil & (np.linalg.norm(SX - base, axis=1) < params['soil_local_radius'])
    P = np.unique(SX[local], axis=0)
    rec = {'method': 'soil_plane', 'normal': None, 'sparse_points': int(len(SX)),
           'soil_supported_points': int(soil.sum()), 'local_points': int(local.sum()),
           'candidate_points_deduplicated': int(len(P)), 'base': floats(base), 'toward': floats(toward)}
    if len(P) < params['min_plane_points']:
        return dict(rec, status='refused', reason='%d candidate points < min_plane_points %d'
                    % (len(P), params['min_plane_points'])), soil, local
    normal, center, inl = ransac_plane(P, params)
    rec['inliers'] = int(inl.sum())
    if normal is None:
        return dict(rec, status='refused', reason='%d inliers < min_inliers %d'
                    % (inl.sum(), params['min_inliers'])), soil, local
    side = float((toward - base) @ normal)
    if side == 0:
        return dict(rec, status='refused', reason='sign undefined: plant reference lies in the plane direction'), soil, local
    if side < 0:
        normal = -normal
    resid = (P[inl] - center) @ normal
    offset = float((base - center) @ normal)
    rec.update(status='ok', normal=floats(normal), point=floats(center), inlier_rms=float(np.sqrt(np.mean(resid ** 2))),
               base_offset_above_plane=offset, angle_to_dataparser_z_deg=angle_deg(normal, [0, 0, 1]))
    return rec, soil, local


def choose_up(a, b, prefer_soil, params):
    """Choice rule: B when it exists and agrees with A within max_up_angle_deg; on disagreement
    A unless prefer_soil. Every case is recorded; nothing is picked silently."""
    ok_a = a is not None and a['status'] == 'ok'
    ok_b = b is not None and b['status'] == 'ok'
    rec = {'angle_a_b_deg': angle_deg(a['normal'], b['normal']) if ok_a and ok_b else None,
           'up_disagreement': False, 'prefer_soil': bool(prefer_soil), 'threshold_deg': params['max_up_angle_deg']}
    if ok_a and ok_b:
        if rec['angle_a_b_deg'] <= params['max_up_angle_deg']:
            return dict(rec, method='soil_plane', up=b['normal'], reason='A and B agree within the threshold; B used')
        rec['up_disagreement'] = True
        if prefer_soil:
            return dict(rec, method='soil_plane', up=b['normal'], reason='up_disagreement; prefer_soil is set, so B used')
        return dict(rec, method='camera_ring', up=a['normal'], reason='up_disagreement; prefer_soil is not set, so A used')
    if ok_a:
        why = 'no soil inputs' if b is None else 'soil plane %s: %s' % (b['status'], b.get('reason'))
        return dict(rec, method='camera_ring', up=a['normal'], reason='A used; ' + why)
    return dict(rec, method=None, up=None, reason='no up: camera ring %s (%s)' % (a['status'], a.get('reason')))


# ----------------------------------------------------------------------------- plant axis

def horizontal_basis(up):
    """Deterministic orthonormal (e1, e2) perpendicular to up: e1 from the dataparser axis least
    aligned with up."""
    up = unit(up)
    axis = np.eye(3)[int(np.argmin(np.abs(up)))]
    e1 = unit(axis - (axis @ up) * up)
    return e1, np.cross(up, e1)


def plant_axis(X, up, params):
    """Slice centers along up every `slice` from the lowest center; slices with at least
    min_slice_points give a median horizontal position and a radius_percentile radius."""
    up = unit(up)
    e1, e2 = horizontal_basis(up)
    h = X @ up
    H = np.stack([X @ e1, X @ e2], 1)
    s = params['slice']
    rec = {'points': int(len(X)), 'point_height_min': float(h.min()) if len(X) else None,
           'point_height_max': float(h.max()) if len(X) else None, 'slices': []}
    if not len(X):
        return dict(rec, status='insufficient', reason='no points')
    k = np.floor((h - h.min()) / s).astype(np.int64)
    for i in range(int(k.max()) + 1):
        sel = k == i
        n = int(sel.sum())
        if n < params['min_slice_points']:
            continue
        m = np.median(H[sel], axis=0)
        r = float(np.percentile(np.linalg.norm(H[sel] - m, axis=1), params['radius_percentile']))
        lo = float(h.min() + i * s)
        hc = lo + s / 2
        rec['slices'].append({'index': i, 'height_low': lo, 'height_center': hc, 'points': n,
                              'horizontal': floats(m), 'axis_point': floats(m[0] * e1 + m[1] * e2 + hc * up),
                              'radius': r})
    sl = rec['slices']
    if not sl:
        return dict(rec, status='insufficient', reason='no slice has min_slice_points')
    rec.update(status='ok', height_range=[sl[0]['height_low'], sl[-1]['height_low'] + s],
               points_in_slices=int(sum(x['points'] for x in sl)), gaps=int(sl[-1]['index'] - sl[0]['index'] + 1 - len(sl)),
               polyline=[x['axis_point'] for x in sl], radius_profile=[[x['height_center'], x['radius']] for x in sl])
    if len(sl) >= 2:
        hc = np.array([x['height_center'] for x in sl])
        M = np.array([x['horizontal'] for x in sl])
        A = np.stack([hc, np.ones_like(hc)], 1)
        coef, *_ = np.linalg.lstsq(A, M, rcond=None)
        dev = np.linalg.norm(M - A @ coef, axis=1)
        rec['straightness'] = {'max_horizontal_deviation': float(dev.max()), 'rms_horizontal_deviation': float(np.sqrt(np.mean(dev ** 2))),
                               'line_tilt_from_up_deg': float(np.degrees(np.arctan(np.linalg.norm(coef[0])))),
                               'definition': 'horizontal residual of the slice medians from a least-squares line in height'}
    else:
        rec['straightness'] = None
    return rec


def axis_at(axis, height):
    """Horizontal position and radius at a height: linear interpolation between slice centers,
    or the nearest slice (flagged) outside them."""
    sl = axis['slices']
    hc = np.array([x['height_center'] for x in sl])
    M = np.array([x['horizontal'] for x in sl])
    R = np.array([x['radius'] for x in sl])
    if height <= hc[0] or height >= hc[-1]:
        i = 0 if height <= hc[0] else len(sl) - 1
        return M[i], float(R[i]), bool(height != hc[i]), 'nearest_slice'
    return (np.array([np.interp(height, hc, M[:, 0]), np.interp(height, hc, M[:, 1])]),
            float(np.interp(height, hc, R)), False, 'interpolated')


def assign_detached(Y, axes, up, params):
    """Assign a detached cluster Y to plant P when r_P/R_P <= assign_max_ratio, its median height
    is within P's height range +- height_margin_slices slices, and r_Q/R_Q >= other_min_ratio for
    every other plant Q; otherwise unassigned. Every number is returned."""
    up = unit(up)
    e1, e2 = horizontal_basis(up)
    h = float(np.median(Y @ up))
    c = np.array([np.median(Y @ e1), np.median(Y @ e2)])
    margin = params['height_margin_slices'] * params['slice']
    rec = {'points': int(len(Y)), 'median_height': h, 'median_horizontal': floats(c), 'plants': {}}
    missing = [name for name, ax in axes.items() if ax['status'] != 'ok']
    for name, ax in axes.items():
        if ax['status'] != 'ok':
            rec['plants'][name] = {'status': 'no_axis'}
            continue
        m, R, outside, how = axis_at(ax, h)
        r = float(np.linalg.norm(c - m))
        lo, hi = ax['height_range']
        rec['plants'][name] = {'axis_horizontal': floats(m), 'r': r, 'R': R, 'ratio': r / R if R > 0 else math.inf,
                               'axis_lookup': how, 'outside_slice_centers': outside,
                               'height_range': [lo, hi], 'within_height_range_with_margin': bool(lo - margin <= h <= hi + margin)}
    if missing:
        return dict(rec, assignment='unassigned', reason='plants without an axis: ' + ', '.join(missing))
    good = [n for n, p in rec['plants'].items() if p['ratio'] <= params['assign_max_ratio'] and p['within_height_range_with_margin']]
    for name in good:
        others = [p['ratio'] for n, p in rec['plants'].items() if n != name]
        if all(o >= params['other_min_ratio'] for o in others):
            return dict(rec, assignment=name, reason='r/R %.4g <= %g within the height range; every other plant r/R >= %g'
                        % (rec['plants'][name]['ratio'], params['assign_max_ratio'], params['other_min_ratio']))
    if not good:
        return dict(rec, assignment='unassigned', reason='no plant has r/R <= %g within its height range' % params['assign_max_ratio'])
    return dict(rec, assignment='unassigned', reason='candidate(s) %s, but another plant has r/R < %g'
                % (', '.join(good), params['other_min_ratio']))
