"""Frame calibration record: the mapping from the export's dataparser frame to a plant frame.

    p_plant = s * R @ (p_source - o)

R is a 3x3 rotation, o the origin in the source frame and s meters_per_scene_unit (1.0 while
scale is unset; target units then stay scale-free scene units). Each of the three parts is set
separately, with its method, inputs (path plus SHA-256) and diagnostics; transform_4x4 is derived
and recomputed on every save. The record is a deterministic JSON sidecar: sorted keys, two-space
indentation, final newline, no timestamps or output-location fields. README.md owns the field
table and CLI (calibrate_frame.py); DESIGN.md explains the frame convention.
"""
import json
import math
from pathlib import Path
import re

import numpy as np

from analyze_organs import VERSION, sha

SCHEMA = 'frame-calibration'
SCHEMA_VERSION = 1
GENERATOR = 'jsps-dt4ag'
SOURCE_FRAME = 'dataparser'
SOURCE_DESCRIPTION = 'nerfstudio dataparser frame of the exported splat'
SCENE_UNITS = 'scale_free_scene_units'
TARGET_FRAME = 'plant'
ROTATION_STATUSES = ('algorithmic', 'human-decision')
ORTHONORMAL_TOL = 1e-9
TRANSFORM_TOL = 1e-12
ESTIMATE_SCHEMA = 'scale-estimate'
ESTIMATE_SCHEMA_VERSION = 1
HEX64 = re.compile(r'^[0-9a-f]{64}$')


def _floats(a):
    return [float(x) for x in np.asarray(a, np.float64).ravel()]


def _matrix(a):
    a = np.asarray(a, np.float64)
    return [[float(x) for x in row] for row in a]


def new(capture_id, dataparser_transforms_sha256, extra_binding=None, origin_definition='unset'):
    """An empty calibration bound to one dataparser transform: no rotation, origin or scale,
    so transform_4x4 is the identity. extra_binding adds optional {name: sha256} binding hashes."""
    binding = {'dataparser_transforms_sha256': dataparser_transforms_sha256}
    for k, v in sorted((extra_binding or {}).items()):
        if k in binding:
            raise ValueError('Duplicate binding key: ' + k)
        binding[k] = v
    cal = {'schema': SCHEMA, 'schema_version': SCHEMA_VERSION,
           'generator': {'name': GENERATOR, 'version': VERSION},
           'capture_id': capture_id,
           'source_frame': {'name': SOURCE_FRAME, 'description': SOURCE_DESCRIPTION, 'units': SCENE_UNITS,
                            'binding': binding},
           'target_frame': {'name': TARGET_FRAME, 'up': '+z', 'units': SCENE_UNITS,
                            'origin_definition': origin_definition},
           'rotation': None, 'origin': None, 'scale': None}
    cal['transform_4x4'] = _matrix(compose(cal))
    return cal


def _parts(cal):
    R = np.array(cal['rotation']['matrix'], np.float64) if cal.get('rotation') else np.eye(3)
    o = np.array(cal['origin']['point_source_frame'], np.float64) if cal.get('origin') else np.zeros(3)
    s = float(cal['scale']['meters_per_scene_unit']) if cal.get('scale') else 1.0
    return R, o, s


def compose(cal):
    """The 4x4 homogeneous matrix of p_plant = s * R @ (p_source - o). Unset parts are identity."""
    R, o, s = _parts(cal)
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = -s * (R @ o)
    return T


def apply(cal, points):
    """Map an (N,3) array of source-frame points into the plant frame (float64, a new array)."""
    P = np.asarray(points, np.float64)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError('Expected an (N,3) point array, got shape %s' % (P.shape,))
    R, o, s = _parts(cal)
    return s * ((P - o) @ R.T)


def horizontal_x(up, reference=(1.0, 0.0, 0.0), min_norm=math.sin(math.radians(1.0))):
    """Rotation R with R @ up = +z and plant +x along the projection of `reference` (default the
    source x axis) onto the plane perpendicular to up. Refuses when that projection is shorter
    than min_norm (reference within asin(min_norm) of the up line). Returns (R, diagnostics)."""
    u = np.asarray(up, np.float64)
    n = float(np.linalg.norm(u))
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError('Up vector is zero or nonfinite')
    u = u / n
    ref = np.asarray(reference, np.float64)
    ref = ref / np.linalg.norm(ref)
    h = ref - (ref @ u) * u
    hn = float(np.linalg.norm(h))
    if hn < min_norm:
        raise ValueError('Degenerate x rule: the reference axis lies within %.3g degrees of up (projection length %.3g < %.3g)'
                         % (math.degrees(math.asin(min(hn, 1.0))), hn, min_norm))
    x = h / hn
    y = np.cross(u, x)
    R = np.stack([x, y, u])
    return R, {'x_rule': 'plant +x is the source-frame reference axis projected onto the plane perpendicular to up; '
                         'plant +y = up x plant +x (right-handed)',
               'x_reference_source_frame': _floats(ref), 'x_projection_length': hn,
               'x_projection_min_length': float(min_norm),
               'angle_reference_to_plant_x_deg': float(math.degrees(math.acos(max(-1.0, min(1.0, float(ref @ x)))))),
               'up_input_norm': n}


def _refresh(cal):
    cal['target_frame']['units'] = 'm' if cal.get('scale') else SCENE_UNITS
    cal['transform_4x4'] = _matrix(compose(cal))
    return cal


def _check_inputs(inputs, where, errors, by=False):
    if not isinstance(inputs, list):
        errors.append(where + ' must be a list')
        return
    for i, item in enumerate(inputs):
        w = '%s[%d]' % (where, i)
        if not isinstance(item, dict):
            errors.append(w + ' must be an object')
            continue
        if not isinstance(item.get('path'), str) or not item['path'].strip():
            errors.append(w + '.path must be a nonempty string')
        if not isinstance(item.get('sha256'), str) or not HEX64.match(item['sha256']):
            errors.append(w + '.sha256 must be 64 lowercase hex characters')
        if by and item.get('by') != 'Alex':
            errors.append(w + '.by must be "Alex"')


def _check_component_common(comp, name, errors):
    if not isinstance(comp.get('method'), str) or not comp['method'].strip():
        errors.append(name + '.method must be a nonempty string')
    if not isinstance(comp.get('inputs'), list) or not comp['inputs']:
        errors.append(name + '.inputs must be a nonempty list')
    else:
        _check_inputs(comp['inputs'], name + '.inputs', errors)


def _finite_array(value, shape):
    try:
        a = np.array(value, np.float64)
    except (TypeError, ValueError):
        return None
    if a.shape != shape or not np.isfinite(a).all():
        return None
    if any(isinstance(x, bool) for x in np.asarray(value, dtype=object).ravel()):
        return None
    return a


def _positive_finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def validate(cal):
    """Return a list of error strings; empty means valid."""
    errors = []
    if not isinstance(cal, dict):
        return ['calibration must be a JSON object']
    if cal.get('schema') != SCHEMA:
        errors.append('schema must be "%s"' % SCHEMA)
    if cal.get('schema_version') != SCHEMA_VERSION:
        return errors + ['unsupported schema_version %r (this software reads %d)' % (cal.get('schema_version'), SCHEMA_VERSION)]
    gen = cal.get('generator')
    if not isinstance(gen, dict) or gen.get('name') != GENERATOR or not isinstance(gen.get('version'), str) or not gen['version']:
        errors.append('generator must be {"name": "%s", "version": <string>}' % GENERATOR)
    if not isinstance(cal.get('capture_id'), str) or not cal['capture_id'].strip():
        errors.append('capture_id must be a nonempty string')
    src = cal.get('source_frame')
    if not isinstance(src, dict):
        errors.append('source_frame must be an object')
    else:
        if src.get('name') != SOURCE_FRAME:
            errors.append('source_frame.name must be "%s"' % SOURCE_FRAME)
        if not isinstance(src.get('description'), str):
            errors.append('source_frame.description must be a string')
        if src.get('units') != SCENE_UNITS:
            errors.append('source_frame.units must be "%s"' % SCENE_UNITS)
        b = src.get('binding')
        if not isinstance(b, dict) or 'dataparser_transforms_sha256' not in b:
            errors.append('source_frame.binding must hold dataparser_transforms_sha256')
        else:
            for k, v in b.items():
                if not isinstance(v, str) or not HEX64.match(v):
                    errors.append('source_frame.binding.%s must be 64 lowercase hex characters' % k)
    tgt = cal.get('target_frame')
    if not isinstance(tgt, dict):
        errors.append('target_frame must be an object')
        tgt = {}
    else:
        if tgt.get('name') != TARGET_FRAME:
            errors.append('target_frame.name must be "%s"' % TARGET_FRAME)
        if tgt.get('up') != '+z':
            errors.append('target_frame.up must be "+z"')
        if not isinstance(tgt.get('origin_definition'), str) or not tgt['origin_definition'].strip():
            errors.append('target_frame.origin_definition must be a nonempty string')
    for key in ('rotation', 'origin', 'scale'):
        if key not in cal:
            errors.append(key + ' must be present (null when unset)')
        elif cal[key] is not None and not isinstance(cal[key], dict):
            errors.append(key + ' must be null or an object')
    parts_ok = True
    rot = cal.get('rotation')
    if isinstance(rot, dict):
        _check_component_common(rot, 'rotation', errors)
        R = _finite_array(rot.get('matrix'), (3, 3))
        if R is None:
            errors.append('rotation.matrix must be a finite 3x3 list of lists')
            parts_ok = False
        else:
            dev = float(np.abs(R @ R.T - np.eye(3)).max())
            if dev > ORTHONORMAL_TOL:
                errors.append('rotation.matrix is not orthonormal (max |R R^T - I| = %.3g > %g)' % (dev, ORTHONORMAL_TOL))
            det = float(np.linalg.det(R))
            if abs(det - 1.0) > ORTHONORMAL_TOL:
                errors.append('rotation.matrix determinant %.12g is not +1' % det)
        if rot.get('status') not in ROTATION_STATUSES:
            errors.append('rotation.status must be one of %s' % (ROTATION_STATUSES,))
        if not isinstance(rot.get('diagnostics'), dict):
            errors.append('rotation.diagnostics must be an object')
    elif rot is not None:
        parts_ok = False
    org = cal.get('origin')
    if isinstance(org, dict):
        _check_component_common(org, 'origin', errors)
        if _finite_array(org.get('point_source_frame'), (3,)) is None:
            errors.append('origin.point_source_frame must be three finite numbers')
            parts_ok = False
    elif org is not None:
        parts_ok = False
    sc = cal.get('scale')
    if isinstance(sc, dict):
        _check_component_common(sc, 'scale', errors)
        if not _positive_finite(sc.get('meters_per_scene_unit')):
            errors.append('scale.meters_per_scene_unit must be a finite number > 0')
            parts_ok = False
        u = sc.get('uncertainty')
        if not isinstance(u, dict):
            errors.append('scale.uncertainty must be an object')
        else:
            std = u.get('std_meters_per_scene_unit')
            if not (isinstance(std, (int, float)) and not isinstance(std, bool) and math.isfinite(std) and std >= 0):
                errors.append('scale.uncertainty.std_meters_per_scene_unit must be a finite number >= 0')
            if type(u.get('n_pairs')) is not int or u['n_pairs'] < 1:
                errors.append('scale.uncertainty.n_pairs must be a positive integer')
            if not isinstance(u.get('method'), str) or not u['method'].strip():
                errors.append('scale.uncertainty.method must be a nonempty string')
        _check_inputs(sc.get('decisions'), 'scale.decisions', errors, by=True)
        if not isinstance(sc.get('diagnostics'), dict):
            errors.append('scale.diagnostics must be an object')
    elif sc is not None:
        parts_ok = False
    want_units = 'm' if isinstance(sc, dict) else SCENE_UNITS
    if tgt.get('units') != want_units:
        errors.append('target_frame.units must be "%s" when scale is %s' % (want_units, 'set' if isinstance(sc, dict) else 'null'))
    T = _finite_array(cal.get('transform_4x4'), (4, 4))
    if T is None:
        errors.append('transform_4x4 must be a finite 4x4 list of lists')
    elif parts_ok:
        dev = float(np.abs(T - compose(cal)).max())
        if dev > TRANSFORM_TOL:
            errors.append('transform_4x4 differs from the composition of rotation, origin and scale (max %.3g > %g)'
                          % (dev, TRANSFORM_TOL))
    return errors


def dumps(cal):
    return json.dumps(cal, sort_keys=True, indent=2, allow_nan=False, ensure_ascii=False) + '\n'


def save(cal, path):
    """Recompute transform_4x4 and target units, validate, and write a new file (existing refused)."""
    _refresh(cal)
    errors = validate(cal)
    if errors:
        raise ValueError('Invalid frame calibration: ' + '; '.join(errors))
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        f.write(dumps(cal))
    return path


def load(path):
    """Read and validate; refuses unknown schema versions and any validation error."""
    with Path(path).open(encoding='utf-8') as f:
        cal = json.load(f)
    errors = validate(cal)
    if errors:
        raise ValueError('Invalid frame calibration %s: %s' % (path, '; '.join(errors)))
    return cal


def _inputs(inputs):
    return [{'path': str(i['path']), 'sha256': str(i['sha256'])} for i in inputs]


def set_rotation(cal, R, origin, method, status, inputs, diagnostics, origin_method=None, origin_inputs=None,
                 origin_definition=None, origin_diagnostics=None):
    """Set the rotation (and, unless origin is None, the origin). The origin takes its own method
    and inputs when given, otherwise the rotation's."""
    R = np.asarray(R, np.float64)
    if R.shape != (3, 3) or not np.isfinite(R).all():
        raise ValueError('Rotation must be a finite 3x3 matrix')
    previous = {k: cal.get(k) for k in ('rotation', 'origin')}
    previous_definition = cal['target_frame'].get('origin_definition')
    cal['rotation'] = {'matrix': _matrix(R), 'method': method, 'status': status, 'inputs': _inputs(inputs),
                       'diagnostics': diagnostics}
    if origin is None:
        cal['origin'] = None
    else:
        cal['origin'] = {'point_source_frame': _floats(origin), 'method': origin_method or method,
                         'inputs': _inputs(origin_inputs if origin_inputs is not None else inputs)}
        if origin_diagnostics is not None:
            cal['origin']['diagnostics'] = origin_diagnostics
    if origin_definition is not None:
        cal['target_frame']['origin_definition'] = origin_definition
    _refresh(cal)
    errors = validate(cal)
    if errors:
        cal.update(previous)
        cal['target_frame']['origin_definition'] = previous_definition
        _refresh(cal)
        raise ValueError('Invalid rotation/origin: ' + '; '.join(errors))
    return cal


def set_scale(cal, meters_per_scene_unit, std_meters_per_scene_unit, n_pairs, uncertainty_method, method,
              decisions, inputs, diagnostics):
    """Set the metric scale; target units become meters."""
    previous = cal.get('scale')
    cal['scale'] = {'meters_per_scene_unit': float(meters_per_scene_unit),
                    'uncertainty': {'std_meters_per_scene_unit': float(std_meters_per_scene_unit),
                                    'n_pairs': n_pairs, 'method': uncertainty_method},
                    'method': method,
                    'decisions': [{'path': str(d['path']), 'sha256': str(d['sha256']), 'by': d.get('by')} for d in decisions],
                    'inputs': _inputs(inputs), 'diagnostics': diagnostics}
    _refresh(cal)
    errors = validate(cal)
    if errors:
        cal['scale'] = previous
        _refresh(cal)
        raise ValueError('Invalid scale: ' + '; '.join(errors))
    return cal


def check_estimate(estimate, cal):
    """Errors that make a scale-estimate unusable for this calibration."""
    errors = []
    if not isinstance(estimate, dict):
        return ['scale estimate must be a JSON object']
    if estimate.get('schema') != ESTIMATE_SCHEMA:
        errors.append('scale estimate schema must be "%s"' % ESTIMATE_SCHEMA)
    if estimate.get('schema_version') != ESTIMATE_SCHEMA_VERSION:
        errors.append('unsupported scale estimate schema_version %r' % estimate.get('schema_version'))
    if estimate.get('capture_id') != cal.get('capture_id'):
        errors.append('scale estimate capture_id %r differs from calibration capture_id %r'
                      % (estimate.get('capture_id'), cal.get('capture_id')))
    if estimate.get('source_frame') != cal['source_frame']['name']:
        errors.append('scale estimate source_frame %r differs from calibration source frame %r'
                      % (estimate.get('source_frame'), cal['source_frame']['name']))
    for key in ('pairs', 'points', 'inputs', 'decisions'):
        if not isinstance(estimate.get(key), list):
            errors.append('scale estimate %s must be a list' % key)
    if isinstance(estimate.get('inputs'), list):
        bound = cal['source_frame']['binding']['dataparser_transforms_sha256']
        for item in estimate['inputs']:
            if isinstance(item, dict) and str(item.get('path', '')).endswith('dataparser_transforms.json') \
                    and item.get('sha256') != bound:
                errors.append('scale estimate input %s has sha256 %s, but the calibration is bound to dataparser '
                              'transforms %s' % (item.get('path'), item.get('sha256'), bound))
    return errors


def set_scale_from_estimate(cal, estimate, estimate_path):
    """Copy a scale-estimate (schema 1) into the scale component. Refuses a mismatched capture_id
    or source frame, and an estimate input naming a different dataparser_transforms.json. The
    estimate's own path and SHA-256 are appended to the scale inputs."""
    errors = check_estimate(estimate, cal)
    if errors:
        raise ValueError('Scale estimate refused: ' + '; '.join(errors))
    path = str(Path(estimate_path).resolve())
    inputs = list(estimate['inputs']) + [{'path': path, 'sha256': sha(path)}]
    diagnostics = {'scale_estimate_schema_version': estimate['schema_version'],
                   'pairs_listed': len(estimate['pairs']), 'points_listed': len(estimate['points']),
                   'source': 'copied from the scale-estimate file listed last in inputs'}
    for k in ('diagnostics', 'outlier_rule', 'settings'):
        if k in estimate:
            diagnostics['estimate_' + k] = estimate[k]
    return set_scale(cal, estimate.get('meters_per_scene_unit'), estimate.get('std_meters_per_scene_unit'),
                     estimate.get('n_pairs'),
                     estimate.get('uncertainty_method') or 'std_meters_per_scene_unit as reported by the scale estimate (%s)'
                     % estimate.get('method'),
                     estimate.get('method'), estimate['decisions'], inputs, diagnostics)
