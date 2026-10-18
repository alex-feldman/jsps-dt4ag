#!/usr/bin/env python3
"""Build, extend, validate and apply frame-calibration records (dataparser frame to plant frame).

Subcommands: orient (rotation and origin from an orient_plants.py orientation.json), scale (metric
scale from a scale-estimate JSON), validate, apply (transform points into a new file). Every
choice is algorithmic or an explicit, recorded human decision (orient --origin-point /
--origin-from-asset with --origin-decided-by Alex); nothing is selected by looking at images.
See README.md.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

from analyze_organs import sha
import frame_calibration as fc

ORIENTATION_SCHEMA = 'plant-orientation'
ORIENTATION_SCHEMA_VERSION = 1


def _angle_deg(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(math.degrees(math.acos(max(-1.0, min(1.0, c)))))


def _vec(value, what):
    a = np.asarray(value, np.float64) if value is not None else None
    if a is None or a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError('orientation.json %s must be three finite numbers' % what)
    return a


def origin_from_asset(asset_path, dataparser_sha):
    """(point, record) for main_axis.polyline[0] of a plant asset's plant.json, after checking that the
    asset's coordinates are in the frame of DATAPARSER_SHA: one of its hash-listed source_evidence JSON
    files must record camera_inputs.dataparser_transforms equal to it. Refuses otherwise."""
    asset_path = Path(asset_path).resolve()
    with asset_path.open() as f:
        asset = json.load(f)
    axis = asset.get('main_axis')
    if not isinstance(axis, dict) or not isinstance(axis.get('polyline'), list) or not axis['polyline']:
        raise ValueError('%s has no main_axis.polyline' % asset_path)
    point = _vec(axis['polyline'][0], 'main_axis.polyline[0] of the asset')
    if asset.get('units') not in (None, fc.SCENE_UNITS):
        raise ValueError('Asset units must be %s, not %r' % (fc.SCENE_UNITS, asset.get('units')))
    frame = None
    for ref in asset.get('source_evidence') or []:
        if not isinstance(ref, dict) or not str(ref.get('path', '')).endswith('.json'):
            continue
        ev = (asset_path.parent / ref['path']).resolve()
        if not ev.is_file():
            continue
        ev_sha = sha(ev)
        if ev_sha != ref.get('sha256'):
            raise ValueError('Asset evidence hash mismatch: %s' % ev)
        with ev.open() as f:
            evidence = json.load(f)
        cams = evidence.get('camera_inputs') if isinstance(evidence, dict) else None
        if isinstance(cams, dict) and cams.get('dataparser_transforms'):
            frame = {'evidence_path': str(ev), 'evidence_sha256': ev_sha,
                     'evidence_dataparser_transforms_sha256': cams['dataparser_transforms'],
                     'evidence_source_ply_sha256': evidence.get('source_ply_sha256'),
                     'evidence_capture_id': evidence.get('capture_id')}
            break
    if frame is None:
        raise ValueError('Cannot establish the asset frame: no hash-listed source_evidence JSON in %s records '
                         'camera_inputs.dataparser_transforms; give the point with --origin-point instead' % asset_path)
    if frame['evidence_dataparser_transforms_sha256'] != dataparser_sha:
        raise ValueError('Asset evidence records dataparser_transforms %s, not %s: different frame'
                         % (frame['evidence_dataparser_transforms_sha256'], dataparser_sha))
    rec = {'asset_path': str(asset_path), 'asset_sha256': sha(asset_path), 'asset_plant_id': asset.get('plant_id'),
           'asset_capture_id': asset.get('capture_id'), 'main_axis_id': axis.get('id'),
           'main_axis_point': 'polyline[0]', 'frame_check': frame}
    return point, rec


def _parse_point(text):
    try:
        values = [float(v) for v in text.split(',')]
    except ValueError:
        values = []
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError('--origin-point expects X,Y,Z (three finite numbers in the source frame), got %r' % text)
    return np.array(values)


def calibration_from_orientation(orientation, orientation_path, dataparser_path, capture_id, bind_files=None,
                                 x_reference=(1.0, 0.0, 0.0), up_axis='+y', explicit_origin=None):
    """Rotation and origin from a complete plant-orientation record.

    Up: up_soil_plane.normal when up_soil_plane.status is ok (method soil_plane), otherwise
    up_camera_mean_image_up.normal when its status is ok (fallback, recorded as such); otherwise
    refused. R maps up to +z, or to +y with up_axis '+y' (the default: the z-up rotation followed
    by [[1,0,0],[0,0,1],[0,-1,0]]); plant +x is the source x axis projected onto the horizontal plane
    (fc.horizontal_x; refused within 1 degree of up). Origin (soil case): up_soil_plane.base (the
    final soil iteration's base estimate, the base plant's lowest-slice axis point) projected along
    up_soil_plane.normal onto the plane through up_soil_plane.point (the final fit's inlier centroid).
    Origin (fallback): the lowest axis point (axis.polyline[0]) of the only plant, unprojected; with
    several plants the origin stays null. The camera-versus-soil angle is a diagnostic only.

    EXPLICIT_ORIGIN (optional) replaces the default origin with a human-decided source-frame point,
    projected along the soil normal onto the soil plane (refused without an ok soil plane). It is a dict
    with 'point' (3,), 'decided_by' ('Alex'), 'decision' (statement with date), 'source' (description),
    and optionally 'record' (asset provenance) and 'inputs' (extra [{'path','sha256'}])."""
    orientation_path = Path(orientation_path).resolve()
    dataparser_path = Path(dataparser_path).resolve()
    if orientation.get('schema') != ORIENTATION_SCHEMA or orientation.get('schema_version') != ORIENTATION_SCHEMA_VERSION:
        raise ValueError('Expected a %s schema_version %d record' % (ORIENTATION_SCHEMA, ORIENTATION_SCHEMA_VERSION))
    if orientation.get('status') != 'complete':
        raise ValueError('orientation.json status is %r, not complete' % orientation.get('status'))
    if orientation.get('capture_id') != capture_id:
        raise ValueError('--capture-id %r differs from orientation.json capture_id %r' % (capture_id, orientation.get('capture_id')))
    if orientation.get('units') != fc.SCENE_UNITS:
        raise ValueError('orientation.json units must be ' + fc.SCENE_UNITS)
    dp_sha = sha(dataparser_path)
    recorded = {p: h for p, h in (orientation.get('input_sha256') or {}).items() if p.endswith('dataparser_transforms.json')}
    if recorded and dp_sha not in recorded.values():
        raise ValueError('--dataparser-transforms sha256 %s matches none recorded by the orientation run: %s'
                         % (dp_sha, json.dumps(recorded, sort_keys=True)))
    o_sha = sha(orientation_path)
    binding = {}
    for name, path in sorted((bind_files or {}).items()):
        key = name if name.endswith('_sha256') else name + '_sha256'
        binding[key] = sha(path)
    cal = fc.new(capture_id, dp_sha, binding, up=up_axis)

    soil = orientation.get('up_soil_plane')
    cam = orientation.get('up_camera_mean_image_up')
    choice = orientation.get('up') or {}
    soil_ok = isinstance(soil, dict) and soil.get('status') == 'ok'
    cam_ok = isinstance(cam, dict) and cam.get('status') == 'ok'
    inputs = [{'path': str(orientation_path), 'sha256': o_sha}, {'path': str(dataparser_path), 'sha256': dp_sha}]
    if soil_ok:
        up = _vec(soil.get('normal'), 'up_soil_plane.normal')
        up_method = 'soil_plane'
    elif cam_ok:
        up = _vec(cam.get('normal'), 'up_camera_mean_image_up.normal')
        up_method = 'camera_mean_image_up (fallback: soil plane %s)' % ('absent' if soil is None else soil.get('status'))
    else:
        raise ValueError('orientation.json has neither an ok soil plane nor an ok camera mean image-up; no up')
    up_unit = up / np.linalg.norm(up)
    R, xdiag = fc.horizontal_x(up_unit, x_reference, target_up=up_axis)
    diag = dict(xdiag)
    diag.update({
        'up_source': up_method,
        'up_source_frame': [float(v) for v in up_unit],
        'up_angle_to_dataparser_z_deg': _angle_deg(up_unit, [0, 0, 1]),
        'orientation_choice_method': choice.get('method'),
        'orientation_choice_status': choice.get('status'),
        'orientation_capture_id': orientation.get('capture_id'),
        'orientation_version': orientation.get('version'),
        'orientation_recorded_dataparser_sha256': recorded,
    })
    if soil_ok and cam_ok:
        diag['camera_vs_soil_angle_deg'] = _angle_deg(soil['normal'], cam['normal'])
        diag['orientation_angle_b_c_deg'] = choice.get('angle_b_c_deg')
        diag['orientation_up_disagreement'] = choice.get('up_disagreement')
        diag['disagreement_threshold_deg'] = choice.get('disagreement_threshold_deg')
        diag['camera_vs_soil_note'] = 'reported only; never acted on'
        diag['camera_mean_image_up_angular_rms_deg'] = cam.get('angular_rms_deg')
    if soil_ok:
        for k in ('inliers', 'inlier_rms', 'converged', 'base_offset_above_plane', 'base_plant',
                  'candidate_points_deduplicated', 'sign_rule'):
            diag['soil_' + k] = soil.get(k)
    if soil_ok and choice.get('method') not in (None, 'soil_plane'):
        diag['note'] = 'orientation.json chose %r, but its soil plane is ok; the soil normal is used' % choice.get('method')
    method = ('up from orientation.json %s; R maps up to %s; plant +x is the source-frame x axis projected '
              'onto the horizontal plane' % ('up_soil_plane.normal' if soil_ok else 'up_camera_mean_image_up.normal', up_axis))

    origin = origin_method = definition = None
    odiag = None
    origin_inputs = [inputs[0]]
    if explicit_origin is not None:
        if explicit_origin.get('decided_by') != 'Alex':
            raise ValueError('An explicit origin must be decided by Alex (--origin-decided-by Alex)')
        statement = str(explicit_origin.get('decision') or '').strip()
        if not statement:
            raise ValueError('An explicit origin needs --origin-decision (the decision statement, with its date)')
        if not soil_ok:
            raise ValueError('An explicit origin is projected onto the soil plane, and this orientation has no ok '
                             'soil plane')
        raw = _vec(explicit_origin.get('point'), 'explicit origin point')
        point = _vec(soil.get('point'), 'up_soil_plane.point')
        height = float((raw - point) @ up_unit)
        origin = raw - height * up_unit
        default_base = _vec(soil.get('base'), 'up_soil_plane.base')
        default_origin = default_base - float((default_base - point) @ up_unit) * up_unit
        shift = origin - default_origin
        origin_method = ('explicit human decision by %s (%s): %s, projected along up_soil_plane.normal onto the '
                         'soil plane through up_soil_plane.point' % (explicit_origin['decided_by'], statement,
                                                                      explicit_origin.get('source')))
        definition = ('human-decided point (%s) projected along the soil normal onto the fitted soil plane: %s'
                      % (statement, explicit_origin.get('source')))
        odiag = {'decision': {'by': explicit_origin['decided_by'], 'statement': statement},
                 'source': explicit_origin.get('source'),
                 'unprojected_point_source_frame': [float(v) for v in raw],
                 'projection_distance_along_up': height,
                 'projection_note': 'signed height of the unprojected point above the soil plane, removed along the up normal',
                 'plane_point_source_frame': [float(v) for v in point],
                 'default_origin_source_frame': [float(v) for v in default_origin],
                 'shift_from_default_origin_source_frame': [float(v) for v in shift],
                 'shift_from_default_origin_scene_units': float(np.linalg.norm(shift))}
        if explicit_origin.get('record') is not None:
            odiag['asset'] = explicit_origin['record']
        origin_inputs = origin_inputs + list(explicit_origin.get('inputs') or [])
    elif soil_ok:
        base = _vec(soil.get('base'), 'up_soil_plane.base')
        point = _vec(soil.get('point'), 'up_soil_plane.point')
        offset = float((base - point) @ up_unit)
        origin = base - offset * up_unit
        origin_method = ('up_soil_plane.base projected along up_soil_plane.normal onto the soil plane through '
                         'up_soil_plane.point')
        definition = ('soil-plane base point: the base plant\'s lowest-slice axis point (orientation up_soil_plane.base, '
                      'final soil iteration) projected along the soil normal onto the fitted soil plane '
                      '(through up_soil_plane.point, the final fit\'s inlier centroid)')
        odiag = {'base_source_frame': [float(v) for v in base], 'plane_point_source_frame': [float(v) for v in point],
                 'base_offset_above_plane_recomputed': offset,
                 'base_offset_above_plane_recorded': soil.get('base_offset_above_plane'),
                 'base_plant': soil.get('base_plant')}
    else:
        plants = {n: p for n, p in (orientation.get('plants') or {}).items()
                  if isinstance(p, dict) and isinstance(p.get('axis'), dict) and p['axis'].get('status') == 'ok'}
        if len(orientation.get('plants') or {}) == 1 and len(plants) == 1:
            name, p = next(iter(plants.items()))
            origin = _vec(p['axis']['polyline'][0], 'plants.%s.axis.polyline[0]' % name)
            origin_method = 'plants.%s.axis.polyline[0] (lowest-slice axis point; no soil plane to project onto)' % name
            definition = ('lowest-slice axis point of plant %s under the fallback camera mean image-up; no soil plane, '
                          'so the origin is not on a ground reference' % name)
            odiag = {'plant': name, 'note': 'fallback origin: not projected onto any plane'}
        else:
            definition = 'unset: no soil plane and not exactly one plant with an axis'
    fc.set_rotation(cal, R, origin, method, 'algorithmic', inputs, diag,
                    origin_method=origin_method, origin_inputs=origin_inputs if origin is not None else None,
                    origin_definition=definition, origin_diagnostics=odiag, up_source=up_unit)
    return cal


def _new_path(path):
    path = Path(path).resolve()
    if path.exists():
        raise ValueError('Output already exists; choose a new path: ' + str(path))
    if not path.parent.is_dir():
        raise ValueError('Output directory does not exist: ' + str(path.parent))
    return path


def _bind(values):
    out = {}
    for v in values or []:
        name, sep, path = v.partition('=')
        if not sep or not name or not path:
            raise ValueError('--bind-file expects NAME=PATH, got ' + v)
        if not Path(path).is_file():
            raise ValueError('Missing --bind-file ' + path)
        if name in out:
            raise ValueError('Duplicate --bind-file name ' + name)
        out[name] = path
    return out


def cmd_orient(args):
    out = _new_path(args.out)
    with Path(args.orientation).open() as f:
        orientation = json.load(f)
    explicit = None
    if args.origin_point is not None or args.origin_from_asset is not None:
        if args.origin_point is not None and args.origin_from_asset is not None:
            raise ValueError('Give --origin-point or --origin-from-asset, not both')
        if args.origin_decided_by is None or args.origin_decision is None:
            raise ValueError('An explicit origin needs --origin-decided-by Alex and --origin-decision TEXT')
        explicit = {'decided_by': args.origin_decided_by, 'decision': args.origin_decision}
        if args.origin_point is not None:
            explicit.update(point=_parse_point(args.origin_point), source='source-frame point given with --origin-point')
        else:
            point, rec = origin_from_asset(args.origin_from_asset, sha(args.dataparser_transforms))
            explicit.update(point=point, record=rec,
                            source='main_axis.polyline[0] (main-axis root) of plant asset %s' % rec['asset_path'],
                            inputs=[{'path': rec['asset_path'], 'sha256': rec['asset_sha256']}])
    elif args.origin_decided_by is not None or args.origin_decision is not None:
        raise ValueError('--origin-decided-by/--origin-decision need --origin-point or --origin-from-asset')
    cal = calibration_from_orientation(orientation, args.orientation, args.dataparser_transforms, args.capture_id,
                                       _bind(args.bind_file), up_axis='+' + args.up, explicit_origin=explicit)
    fc.save(cal, out)
    d = cal['rotation']['diagnostics']
    print('up: %s -> target %s; angle to dataparser z %.4f deg' % (d['up_source'], cal['target_frame']['up'],
                                                                 d['up_angle_to_dataparser_z_deg']))
    if 'camera_vs_soil_angle_deg' in d:
        print('camera vs soil angle: %.4f deg (reported only)' % d['camera_vs_soil_angle_deg'])
    print('origin: %s' % (cal['origin']['point_source_frame'] if cal['origin'] else None))
    od = (cal['origin'] or {}).get('diagnostics') or {}
    if 'projection_distance_along_up' in od:
        print('explicit origin: projected %.6f along up onto the soil plane; %.6f scene units from the default origin'
              % (od['projection_distance_along_up'], od['shift_from_default_origin_scene_units']))
    print('Wrote ' + str(out))
    return 0


def cmd_scale(args):
    out = _new_path(args.out)
    cal = fc.load(args.calibration)
    with Path(args.estimate).open() as f:
        estimate = json.load(f)
    fc.set_scale_from_estimate(cal, estimate, args.estimate)
    fc.save(cal, out)
    print('meters_per_scene_unit %.9g (std %.3g, %d pairs)' % (cal['scale']['meters_per_scene_unit'],
          cal['scale']['uncertainty']['std_meters_per_scene_unit'], cal['scale']['uncertainty']['n_pairs']))
    print('Wrote ' + str(out))
    return 0


def cmd_validate(args):
    with Path(args.calibration).open() as f:
        cal = json.load(f)
    errors = fc.validate(cal)
    if errors:
        for e in errors:
            print('invalid: ' + e)
        return 1
    print('valid: %s (rotation %s, origin %s, scale %s, units %s)' % (
        args.calibration, 'set' if cal['rotation'] else 'null', 'set' if cal['origin'] else 'null',
        'set' if cal['scale'] else 'null', cal['target_frame']['units']))
    return 0


def cmd_apply(args):
    cal = fc.load(args.calibration)
    src = Path(args.input).resolve()
    out = _new_path(args.out)
    if out == src:
        raise ValueError('apply never writes in place')
    if src.suffix == '.npy':
        if out.suffix != '.npy':
            raise ValueError('An .npy input needs an .npy output')
        P = np.load(src, allow_pickle=False)
        Y = fc.apply(cal, P)
        with out.open('xb') as f:
            np.save(f, Y)
    elif src.suffix == '.npz':
        if out.suffix != '.npz':
            raise ValueError('An .npz input needs an .npz output')
        with np.load(src, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        if args.array not in arrays:
            raise ValueError('%s has no array %s (has %s)' % (src, args.array, ', '.join(sorted(arrays))))
        arrays[args.array] = fc.apply(cal, arrays[args.array])
        with out.open('xb') as f:
            np.savez(f, **arrays)
        Y = arrays[args.array]
    else:
        raise ValueError('Expected an .npy or .npz input')
    print('Transformed %d points into %s (%s); wrote %s' % (len(Y), cal['target_frame']['name'],
                                                           cal['target_frame']['units'], out))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('orient', help='rotation and origin from an orientation.json')
    p.add_argument('--orientation', required=True, type=Path)
    p.add_argument('--dataparser-transforms', required=True, type=Path)
    p.add_argument('--capture-id', required=True)
    p.add_argument('--bind-file', action='append', metavar='NAME=PATH',
                   help='extra source-frame binding hash, recorded as NAME_sha256 (repeatable)')
    p.add_argument('--up', choices=['y', 'z'], default='y',
                   help='target up axis: y (default) or z; recorded as target_frame.up')
    p.add_argument('--origin-point', metavar='X,Y,Z',
                   help='explicit origin, source (dataparser) frame; projected onto the soil plane along up')
    p.add_argument('--origin-from-asset', type=Path, metavar='PLANT_JSON',
                   help="explicit origin at a plant asset's main_axis.polyline[0], after checking the asset's "
                        'evidence names the same dataparser transforms; projected onto the soil plane along up')
    p.add_argument('--origin-decided-by', choices=['Alex'], help='who decided the explicit origin (required with it)')
    p.add_argument('--origin-decision', metavar='TEXT',
                   help='the decision statement with its date, recorded verbatim (required with an explicit origin)')
    p.add_argument('--out', required=True, type=Path)
    p.set_defaults(func=cmd_orient)
    p = sub.add_parser('scale', help='add the metric scale from a scale-estimate JSON')
    p.add_argument('--calibration', required=True, type=Path)
    p.add_argument('--estimate', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.set_defaults(func=cmd_scale)
    p = sub.add_parser('validate', help='check a calibration file')
    p.add_argument('calibration', type=Path)
    p.set_defaults(func=cmd_validate)
    p = sub.add_parser('apply', help='transform an (N,3) .npy, or one array of an .npz, into a new file')
    p.add_argument('--calibration', required=True, type=Path)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--array', default='X', help='array name inside an .npz input (default X)')
    p.add_argument('--out', required=True, type=Path)
    p.set_defaults(func=cmd_apply)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print('calibrate-frame: ' + str(exc), file=sys.stderr)
        sys.exit(2)
