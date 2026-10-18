"""Metric scale (meters per dataparser scene unit) from clicked tape-measure graduations.

Uses every "tape_{N}cm" label of an image-clicks file. Each label becomes a 3D point with
click_points (multi-view when clicked in 2+ images, single-view-local-surface otherwise). For
every pair of resolved marks the known distance is |Na - Nb| cm and the scene distance is their
dataparser-frame separation. The estimate is the baseline-weighted median of the pair estimates;
see README.md ("Clicked image points and ruler scale") and DESIGN.md for the rules.

python ruler_scale.py --clicks F --colmap DIR --dataparser F --capture-id ID --output OUT.json
       [--exclude LABEL ...] [--parameters P.json]
Exit 0 written; 2 invalid input; 3 fewer than two resolved marks (a refusal record is written
to OUT with suffix .refused.json, never to OUT).
"""
import argparse
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import click_points as cp

TAPE_RE = re.compile(r'^tape_(?P<n>\d+(?:\.\d+)?)cm$')
METHOD = ('baseline-weighted median of all pair estimates (weight = known distance); '
          'std = max(1.4826 x weighted MAD of the pair estimates, propagated std of the longest pair)')


def tape_marks(labels):
    """{label: N in cm} for tape_{N}cm labels; refuses two labels with the same N."""
    out, by_n = {}, {}
    for label in labels:
        m = TAPE_RE.match(label)
        if not m:
            continue
        n = float(m.group('n'))
        if n in by_n:
            raise ValueError('Two tape labels for %g cm: %s and %s' % (n, by_n[n], label))
        by_n[n] = label
        out[label] = n
    return out


def weighted_median(values, weights):
    order = np.argsort(values)
    v, w = np.asarray(values, float)[order], np.asarray(weights, float)[order]
    c = np.cumsum(w)
    half = c[-1] / 2
    k = int(np.searchsorted(c, half))
    if math.isclose(c[k], half, rel_tol=1e-12) and k + 1 < len(v):
        return float((v[k] + v[k + 1]) / 2)
    return float(v[k])


def line_fit(P):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    return c, vt[0]


def estimate(marks, points):
    """marks {label: N cm}; points {label: solve_label record with xyz_dataparser}.
    Returns the estimate dict (scale-estimate fields plus checks)."""
    labels = sorted((l for l in marks if l in points), key=lambda l: marks[l])
    pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            pa, pb = points[a], points[b]
            known = abs(marks[a] - marks[b]) / 100.0
            v = pb['xyz_dataparser'] - pa['xyz_dataparser']
            d = float(np.linalg.norm(v))
            u = v / d if d > 0 else np.zeros(3)
            sd = float(np.sqrt(max(u @ (pa['cov_dataparser'] + pb['cov_dataparser']) @ u, 0)))
            pairs.append({'a': a, 'b': b, 'known_m': known, 'scene_distance': d,
                          'meters_per_scene_unit': known / d if d > 0 else math.inf,
                          'scene_distance_sigma': sd})
    if not pairs:
        raise ValueError('fewer than two resolved tape marks')
    est = np.array([p['meters_per_scene_unit'] for p in pairs])
    w = np.array([p['known_m'] for p in pairs])
    m = weighted_median(est, w)
    spread = 1.4826 * weighted_median(np.abs(est - m), w) if len(pairs) >= 2 else 0.0
    longest = max(pairs, key=lambda p: (p['known_m'], p['a']))
    prop = longest['meters_per_scene_unit'] * longest['scene_distance_sigma'] / longest['scene_distance']
    for p in pairs:
        p['propagated_std'] = p['meters_per_scene_unit'] * p['scene_distance_sigma'] / p['scene_distance']
    rec = {'meters_per_scene_unit': m, 'std_meters_per_scene_unit': max(spread, prop), 'n_pairs': len(pairs),
           'pairs': pairs, 'estimators': {
               'weighted_median': m, 'weighted_mad_std': spread, 'longest_pair': {'a': longest['a'], 'b': longest['b'],
                                                                                 'meters_per_scene_unit': longest['meters_per_scene_unit'],
                                                                                 'propagated_std': prop},
               'unweighted_median': float(np.median(est)), 'min': float(est.min()), 'max': float(est.max())}}
    rec['collinearity'] = collinearity(labels, marks, points, m)
    return rec


def collinearity(labels, marks, points, mpsu):
    """Check: perpendicular deviation of the 3D marks from their best-fit line, and the spacing
    along it against the graduations (a line fit of position versus N)."""
    if len(labels) < 3:
        return {'status': 'not_applicable', 'reason': 'needs at least three resolved marks', 'marks': len(labels)}
    P = np.array([points[l]['xyz_dataparser'] for l in labels])
    N = np.array([marks[l] for l in labels])
    c, u = line_fit(P)
    s = (P - c) @ u
    perp = np.linalg.norm(P - c - s[:, None] * u, axis=1)
    A = np.stack([N, np.ones_like(N)], 1)
    coef, *_ = np.linalg.lstsq(A, s, rcond=None)
    spacing = s - A @ coef
    per_cm = abs(float(coef[0]))
    loo = []
    for k in range(len(labels)):
        if len(labels) < 4:
            loo.append((None, None))
            continue
        keep = np.arange(len(labels)) != k
        ck, uk = line_fit(P[keep])
        sk = (P[keep] - ck) @ uk
        ak, *_ = np.linalg.lstsq(np.stack([N[keep], np.ones(int(keep.sum()))], 1), sk, rcond=None)
        q = P[k] - ck
        along = float(q @ uk)
        loo.append((float(np.linalg.norm(q - along * uk) * mpsu * 1000),
                    float(abs(along - (ak[0] * N[k] + ak[1])) * mpsu * 1000)))
    return {'status': 'ok', 'marks': len(labels),
            'max_perpendicular_deviation_scene': float(perp.max()),
            'max_perpendicular_deviation_mm': float(perp.max() * mpsu * 1000),
            'max_spacing_residual_scene': float(np.abs(spacing).max()),
            'max_spacing_residual_mm': float(np.abs(spacing).max() * mpsu * 1000),
            'line_fit_meters_per_scene_unit': 0.01 / per_cm if per_cm > 0 else None,
            'per_mark': [{'label': l, 'N_cm': float(n), 'perpendicular_deviation_mm': float(p * mpsu * 1000),
                          'spacing_residual_mm': float(r * mpsu * 1000),
                          'leave_one_out_perpendicular_mm': lo[0], 'leave_one_out_spacing_residual_mm': lo[1]}
                         for l, n, p, r, lo in zip(labels, N, perp, spacing, loo)],
            'definition': ('SVD line through the marks; spacing residual from a least-squares fit of position along the line '
                           'versus N; leave-one-out values refit both without the mark (needs 4 marks; the sign of '
                           'the along-line direction is arbitrary, so the leave-one-out spacing residual is absolute); '
                           'mm use the estimate')}


def run(args):
    params = cp.load_params(args.parameters)
    out = Path(args.output)
    refused_path = out.with_name(out.stem + '.refused.json')
    if out.exists() or refused_path.exists():
        raise ValueError('Output exists: ' + str(out if out.exists() else refused_path))
    doc = cp.load_clicks(args.clicks)
    obs = cp.observations(doc)
    marks = tape_marks(obs)
    unknown = set(args.exclude or []) - set(marks)
    if unknown:
        raise ValueError('--exclude names labels that are not tape marks in the clicks: ' + ', '.join(sorted(unknown)))
    model = cp.Model(args.colmap, args.dataparser, args.sparse, args.transforms)
    model.check_click_images(doc)
    excluded = sorted(args.exclude or [])
    offset_m = float(args.surface_offset_m)
    if not math.isfinite(offset_m) or offset_m < 0:
        raise ValueError('--surface-offset-m must be a nonnegative number')
    if offset_m and not args.surface_offset_basis:
        raise ValueError('--surface-offset-m needs --surface-offset-basis (who stated it and how)')

    def solve(offset_colmap):
        solved, refused = {}, []
        for label in sorted(marks, key=marks.get):
            rec = cp.solve_label(model, obs[label], params, surface_offset_colmap=offset_colmap)
            if 'xyz_dataparser' not in rec:
                refused.append(dict(cp.point_record(label, rec)))
            elif label not in excluded:
                solved[label] = rec
        return solved, refused

    # A metric offset needs the scale: start at zero offset, convert offset_m with the current
    # estimate (meters -> dataparser -> COLMAP units), re-solve, until the estimate stops changing.
    offset_colmap, iterations = 0.0, []
    solved, refused = solve(0.0)
    for it in range(args.max_offset_iterations if offset_m else 1):
        if len(solved) < 2:
            break
        m = estimate(marks, solved)['meters_per_scene_unit']
        iterations.append({'iteration': it, 'offset_colmap': offset_colmap, 'offset_dataparser': offset_colmap * model.scale,
                           'meters_per_scene_unit': m})
        if not offset_m:
            break
        new = offset_m / m / model.scale
        if it and abs(new - offset_colmap) <= 1e-12 * max(new, 1e-30):
            iterations[-1]['converged'] = True
            break
        offset_colmap = new
        solved, refused = solve(offset_colmap)
    offset_rec = {'surface_offset_m': offset_m, 'basis': args.surface_offset_basis, 'final_offset_colmap': offset_colmap,
                  'final_offset_dataparser': offset_colmap * model.scale, 'iterations': iterations,
                  'converged': (not offset_m) or bool(iterations and iterations[-1].get('converged')),
                  'rule': ('single-view local planes are moved along their normal toward the camera by the offset before '
                           'ray intersection; offset_scene = offset_m / meters_per_scene_unit, iterated to a fixed point; '
                           'multi-view marks are unaffected')}
    base = {'schema': 'scale-estimate', 'schema_version': 1, 'capture_id': args.capture_id, 'source_frame': 'dataparser'}
    inputs = model.input_records()
    decisions = [{'path': str(Path(args.clicks).resolve()), 'sha256': cp.sha256(args.clicks), 'by': 'Alex'}]
    extra = {'created': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'parameters': params,
             'checks': model.checks, 'tape_labels': {l: marks[l] for l in sorted(marks, key=marks.get)},
             'excluded_labels': excluded, 'refused_points': refused, 'surface_offset': offset_rec,
             'code': cp.code_records(), 'source_revision': cp.source_revision(), 'command': sys.argv}
    if len(solved) < 2:
        cp.write_new(refused_path, cp.jsonable(dict(base, status='refused', reason='%d resolved tape marks < 2' % len(solved),
                                                    inputs=inputs, decisions=decisions, **extra)))
        print('refused: %d resolved tape marks (need 2); wrote %s' % (len(solved), refused_path), file=sys.stderr)
        return 3
    e = estimate(marks, solved)
    points = []
    for label in sorted(solved, key=marks.get):
        r = cp.point_record(label, solved[label])
        points.append({'label': label, 'xyz_dataparser': r['xyz_dataparser'], 'method': r['method'],
                       'diagnostics': dict(r['diagnostics'], status=r['status'], flags=r['flags'],
                                           xyz_colmap=r['xyz_colmap'])})
    if offset_m and not offset_rec['converged']:
        raise ValueError('surface offset iteration did not converge in %d iterations' % args.max_offset_iterations)
    summary = mark_summary(model, obs, solved, e['meters_per_scene_unit'])
    result = dict(base, meters_per_scene_unit=e['meters_per_scene_unit'],
                  std_meters_per_scene_unit=e['std_meters_per_scene_unit'], n_pairs=e['n_pairs'], method=METHOD,
                  pairs=e['pairs'], points=points, inputs=inputs, decisions=decisions,
                  estimators=e['estimators'], collinearity=e['collinearity'], mark_summary=summary, **extra)
    cp.write_new(out, cp.jsonable(result))
    print('meters_per_scene_unit %.6g +- %.2g from %d pairs of %d marks (%d refused, %d excluded)'
          % (e['meters_per_scene_unit'], e['std_meters_per_scene_unit'], e['n_pairs'], len(solved), len(refused), len(excluded)))
    for p in points:
        print('  %-14s %-26s %-8s %s' % (p['label'], p['method'], p['diagnostics']['status'], ','.join(p['diagnostics']['flags'])))
    for s in summary:
        print('  %-10s support %s r=%s px, plane rms %s mm, camera distance %.4f m' % (
            s['label'], s.get('support_count'), s.get('support_radius_px'), _fmt(s.get('plane_residual_rms_mm')),
            s['camera_distance_m']))
    col = e['collinearity']
    if col['status'] == 'ok':
        print('collinearity: max perpendicular %.2f mm, max spacing residual %.2f mm'
              % (col['max_perpendicular_deviation_mm'], col['max_spacing_residual_mm']))
    return 0


def _fmt(v):
    return 'n/a' if v is None else '%.2f' % v


def mark_summary(model, obs, solved, mpsu):
    """Per mark, in meters with the estimate: support, plane residual, camera distance, and where
    the supporting COLMAP points lie relative to the bare-surface ray hit (the signal for a tape
    that is suspended above, rather than resting on, the surface)."""
    k = mpsu * model.scale * 1000  # mm per COLMAP unit
    out = []
    for label, rec in sorted(solved.items(), key=lambda kv: kv[0]):
        cam = model.camera(obs[label][0][0])
        dist_m = float(np.linalg.norm(rec['xyz_colmap'] - cam.center) * mpsu * model.scale)
        d = rec['diagnostics']
        s = {'label': label, 'method': rec['method'], 'status': rec['status'], 'flags': rec.get('flags', []),
             'image': obs[label][0][0], 'camera_distance_m': dist_m}
        if rec['method'] == 'single-view-local-surface':
            sp = d.get('support_points', [])
            s.update(submethod=d['submethod'], support_count=d['support_count'], support_radius_px=d['support_radius_px'],
                     plane_inliers=d.get('plane_inliers'), plane_inlier_share=d.get('plane_inlier_share'),
                     ray_incidence_deg=d.get('ray_incidence_deg'),
                     plane_residual_rms_mm=None if d.get('plane_residual_rms_colmap') is None else d['plane_residual_rms_colmap'] * k,
                     along_ray_sigma_mm=d['along_ray_sigma_colmap'] * k,
                     surface_offset_mm=d.get('surface_offset_colmap', 0.0) * k,
                     support_point_ids=d['support_point_ids'])
            sd = [p['signed_distance_from_plane_toward_camera_colmap'] * k for p in sp if 'signed_distance_from_plane_toward_camera_colmap' in p]
            dz = [p.get('depth_minus_surface_hit_depth_colmap', p.get('depth_minus_median_depth_colmap')) * k for p in sp]
            if sd:
                s['support_signed_distance_from_plane_mm'] = {'min': min(sd), 'median': float(np.median(sd)), 'max': max(sd)}
            if dz:
                s['support_depth_minus_surface_hit_mm'] = {'min': min(dz), 'median': float(np.median(dz)), 'max': max(dz)}
        else:
            s.update(max_residual_px=d['max_residual_px'], triangulation_angle_max_deg=d['triangulation_angle_max_deg'])
        out.append(s)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cp.add_model_args(ap)
    ap.add_argument('--exclude', action='append', help='tape label to leave out (Alex\'s explicit decision; recorded)')
    ap.add_argument('--surface-offset-m', type=float, default=0.0,
                    help='height of the tape above the supporting surface in meters (single-view marks)')
    ap.add_argument('--surface-offset-basis', help='who stated the offset and how (recorded)')
    ap.add_argument('--max-offset-iterations', type=int, default=50)
    args = ap.parse_args(argv)
    try:
        return run(args)
    except ValueError as e:
        print('error: ' + str(e), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
