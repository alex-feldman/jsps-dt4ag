"""Human-clicked image points to 3D points in the export's dataparser frame.

Clicks are Alex's explicit decisions (an `image-clicks` file from the click tool). Everything
after the clicks is algorithmic with the recorded settings in DEFAULTS: no image pixels are read
and nothing is chosen by looking at photographs. See README.md ("Clicked image points and ruler
scale") for commands and formats, DESIGN.md for the methods.

Two methods:
- multi-view: a label clicked in two or more images is triangulated (linear DLT on undistorted
  normalized coordinates, then nonlinear least squares on pixel reprojection error with the
  camera's own distortion). Per-view residuals and the triangulation angle are reported.
- single-view-local-surface: a label clicked in exactly one image is placed where its ray meets
  a robust local plane through the COLMAP 3D points observed in that same image whose keypoints
  lie near the click (median depth when support is small; refused when support is smaller still).

Points are computed in the COLMAP model frame and mapped to the dataparser frame with
plant_orientation.colmap_to_dataparser (the saved transform already includes applied_transform;
procedure 6a). Distances are scale-free scene units unless a scale-estimate is supplied.

CLI: python click_points.py triangulate --clicks F --colmap DIR --dataparser F --output OUT.json
     [--plant-asset DIR] [--scale SCALE.json]
"""
import argparse
import hashlib
import json
import math
import re
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from plant_orientation import camera_poses, colmap_to_dataparser, read_points3d_binary

SCHEMA = 'image-clicks'
SCHEMA_VERSION = 1
POINTS_SCHEMA = 'image-click-points'
POINTS_SCHEMA_VERSION = 1
PIXEL_CONVENTION = ('x right, y down, origin at the top-left corner of the top-left pixel; '
                    'pixel centers at i+0.5 (COLMAP convention)')

# Experimental defaults, first set 2026-10-18 for the October 2 Welzo M1 model (4096 x 3072,
# f about 2975 px). Not universal tuning; every output records the values it used.
DEFAULTS = {
    # multi-view
    'max_reprojection_px': 4.0,        # flag when any view's residual exceeds this (COLMAP's default filter; this model's own keypoints reach 3.98 px)
    'min_triangulation_angle_deg': 2.0,  # flag a point when the largest ray-pair angle is below this
    'click_sigma_px': 1.0,             # assumed click noise for the reported 3D covariance
    # single-view local surface
    'support_radius_px': 40.0,         # first support radius: COLMAP keypoints in the clicked image within this of the click
    'max_support_radius_px': 160.0,    # the radius doubles up to this until min_support_plane points are found
    'min_support': 4,                  # refuse below this many support points
    'min_support_plane': 8,            # plane fit at or above this; median depth between the two
    'min_plane_inliers': 6,            # a plane with fewer inliers falls back to median depth (flagged)
    'plane_tol_rel_depth': 0.005,      # RANSAC inlier distance, as a fraction of the support's median depth
    'ransac_trials': 500,
    'ransac_seed': 0,
    'min_inlier_share': 0.6,           # flag 'mixed_surfaces' below this inlier share
    'max_incidence_deg': 80.0,         # ray-to-normal angle above this: plane too grazing, use median depth (flagged)
    'max_depth_spread_rel': 0.02,      # median-depth only: flag when the support's robust depth spread exceeds this fraction of depth
}
INTEGER_KEYS = {'min_support', 'min_support_plane', 'min_plane_inliers', 'ransac_trials', 'ransac_seed'}

# COLMAP camera model ids and parameter counts this module supports.
CAMERA_MODELS = {0: ('SIMPLE_PINHOLE', 3), 1: ('PINHOLE', 4), 2: ('SIMPLE_RADIAL', 4), 3: ('RADIAL', 5),
                 4: ('OPENCV', 8)}
ALL_MODEL_PARAMS = {5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}  # read, then refused (fisheye etc.)


def check_parameters(params):
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise ValueError('Unknown parameters: ' + ', '.join(sorted(unknown)))
    for key, value in params.items():
        if key in INTEGER_KEYS:
            if type(value) is not int or value < (0 if key == 'ransac_seed' else 1):
                raise ValueError('Expected integer ' + key)
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Expected positive number ' + key)
    if params['min_support'] < 1 or params['min_support_plane'] < max(3, params['min_support']):
        raise ValueError('Need min_support >= 1 and min_support_plane >= max(3, min_support)')
    if not 4 <= params['min_plane_inliers'] <= params['min_support_plane']:
        raise ValueError('Need 4 <= min_plane_inliers <= min_support_plane')
    if params['max_support_radius_px'] < params['support_radius_px']:
        raise ValueError('max_support_radius_px must be at least support_radius_px')
    if not params['min_inlier_share'] <= 1 or not params['max_incidence_deg'] < 90:
        raise ValueError('min_inlier_share must be at most 1 and max_incidence_deg below 90')
    return params


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def floats(v):
    return [float(x) for x in np.asarray(v, np.float64).ravel()]


# ----------------------------------------------------------------------------- clicks file

def _number(v):
    return not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)


def validate_clicks(doc, source='clicks'):
    """Validate a parsed image-clicks v1 document. Returns it unchanged; raises ValueError with
    every problem found. Bounds: 0 <= x <= width, 0 <= y <= height (pixel-corner convention)."""
    errors = []
    if not isinstance(doc, dict):
        raise ValueError(source + ': not a JSON object')
    if doc.get('schema') != SCHEMA or doc.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('%s: expected schema %r version %d, got %r version %r'
                         % (source, SCHEMA, SCHEMA_VERSION, doc.get('schema'), doc.get('schema_version')))
    if doc.get('pixel_convention') != PIXEL_CONVENTION:
        errors.append('pixel_convention must be exactly %r' % PIXEL_CONVENTION)
    for key in ('tool', 'tool_version', 'created'):
        if not isinstance(doc.get(key), str) or not doc[key].strip():
            errors.append('missing or empty ' + key)
    images, points = doc.get('images'), doc.get('points')
    if not isinstance(images, list) or not images:
        errors.append('images must be a nonempty list')
        images = []
    if not isinstance(points, list) or not points:
        errors.append('points must be a nonempty list')
        points = []
    size = {}
    for i, im in enumerate(images):
        if not isinstance(im, dict) or not isinstance(im.get('name'), str) or not im['name']:
            errors.append('images[%d]: missing name' % i)
            continue
        if im['name'] in size:
            errors.append('images[%d]: duplicate image %s' % (i, im['name']))
        w, h = im.get('width'), im.get('height')
        if type(w) is not int or type(h) is not int or w <= 0 or h <= 0:
            errors.append('images[%d]: width and height must be positive integers' % i)
            continue
        size[im['name']] = (w, h)
    seen = set()
    for i, p in enumerate(points):
        if not isinstance(p, dict):
            errors.append('points[%d]: not an object' % i)
            continue
        name, label = p.get('image'), p.get('label')
        if not isinstance(label, str) or not label.strip() or label != label.strip():
            errors.append('points[%d]: label must be nonempty text without surrounding spaces' % i)
            continue
        if name not in size:
            errors.append('points[%d] (%s): unknown image %r' % (i, label, name))
            continue
        if (name, label) in seen:
            errors.append('points[%d]: duplicate label %r in image %s' % (i, label, name))
        seen.add((name, label))
        x, y = p.get('x'), p.get('y')
        if not _number(x) or not _number(y):
            errors.append('points[%d] (%s): x and y must be finite numbers' % (i, label))
            continue
        w, h = size[name]
        if not (0 <= x <= w and 0 <= y <= h):
            errors.append('points[%d] (%s): pixel (%g, %g) outside %s (%d x %d)' % (i, label, x, y, name, w, h))
    if errors:
        raise ValueError(source + ': ' + '; '.join(errors))
    return doc


def load_clicks(path):
    return validate_clicks(json.loads(Path(path).read_text(encoding='utf-8')), str(path))


def observations(doc):
    """{label: [(image name, x, y), ...]} in file order."""
    out = {}
    for p in doc['points']:
        out.setdefault(p['label'], []).append((p['image'], float(p['x']), float(p['y'])))
    return out


# ----------------------------------------------------------------------------- COLMAP model

def read_cameras_binary(path):
    """COLMAP cameras.bin: {camera_id: {'model', 'width', 'height', 'params'}}."""
    buf = Path(path).read_bytes()
    (count,) = struct.unpack_from('<Q', buf, 0)
    off, out = 8, {}
    for _ in range(count):
        cid, model, w, h = struct.unpack_from('<iiQQ', buf, off)
        off += 24
        if model in CAMERA_MODELS:
            name, n = CAMERA_MODELS[model]
        elif model in ALL_MODEL_PARAMS:
            name, n = 'unsupported-%d' % model, ALL_MODEL_PARAMS[model]
        else:
            raise ValueError('Unknown COLMAP camera model id %d' % model)
        params = np.array(struct.unpack_from('<%dd' % n, buf, off))
        off += 8 * n
        out[cid] = {'model': name, 'model_id': model, 'width': int(w), 'height': int(h), 'params': params}
    if off != len(buf):
        raise ValueError('Unexpected trailing bytes in ' + str(path))
    return out


def qvec_to_rotation(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def read_images_poses(path):
    """COLMAP images.bin: {image_id: {'name', 'camera_id', 'R', 't', 'xys', 'point3D_ids'}} with
    world-to-camera x_cam = R X + t (COLMAP convention)."""
    buf = Path(path).read_bytes()
    (count,) = struct.unpack_from('<Q', buf, 0)
    off, out = 8, {}
    rec_t = np.dtype([('x', '<f8'), ('y', '<f8'), ('id', '<i8')])
    for _ in range(count):
        (iid,) = struct.unpack_from('<i', buf, off)
        q = np.array(struct.unpack_from('<4d', buf, off + 4))
        t = np.array(struct.unpack_from('<3d', buf, off + 36))
        (cid,) = struct.unpack_from('<i', buf, off + 60)
        off += 64
        end = buf.index(b'\0', off)
        name = buf[off:end].decode()
        off = end + 1
        (n,) = struct.unpack_from('<Q', buf, off)
        off += 8
        rec = np.frombuffer(buf, dtype=rec_t, count=n, offset=off)
        off += 24 * n
        out[iid] = {'name': name, 'camera_id': cid, 'R': qvec_to_rotation(q), 't': t,
                    'xys': np.stack([rec['x'], rec['y']], 1), 'point3D_ids': rec['id'].copy()}
    if off != len(buf):
        raise ValueError('Unexpected trailing bytes in ' + str(path))
    return out


class Camera:
    """One registered image: intrinsics, distortion and COLMAP-frame pose."""

    def __init__(self, image_id, name, cam, R, t):
        if cam['model_id'] not in CAMERA_MODELS:
            raise ValueError('Unsupported camera model %s for %s' % (cam['model'], name))
        self.image_id, self.name, self.model = image_id, name, cam['model']
        self.width, self.height = cam['width'], cam['height']
        p = cam['params']
        k1 = k2 = p1 = p2 = 0.0
        if self.model == 'SIMPLE_PINHOLE':
            fx = fy = p[0]; cx, cy = p[1:3]
        elif self.model == 'PINHOLE':
            fx, fy, cx, cy = p
        elif self.model == 'SIMPLE_RADIAL':
            fx = fy = p[0]; cx, cy, k1 = p[1:4]
        elif self.model == 'RADIAL':
            fx = fy = p[0]; cx, cy, k1, k2 = p[1:5]
        else:
            fx, fy, cx, cy, k1, k2, p1, p2 = p
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.k1, self.k2, self.p1, self.p2 = float(k1), float(k2), float(p1), float(p2)
        self.R, self.t = np.asarray(R, np.float64), np.asarray(t, np.float64)
        self.center = -self.R.T @ self.t

    def distort(self, xy):
        """Undistorted normalized (N,2) to distorted normalized (OpenCV radial-tangential)."""
        x, y = xy[:, 0], xy[:, 1]
        r2 = x * x + y * y
        radial = 1 + self.k1 * r2 + self.k2 * r2 * r2
        dx = 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
        dy = self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y
        return np.stack([x * radial + dx, y * radial + dy], 1)

    def undistort_pixels(self, uv):
        """Pixels (N,2) to undistorted normalized coordinates by fixed-point iteration, verified
        by redistorting (refuses when it does not converge to 1e-10 normalized)."""
        uv = np.atleast_2d(np.asarray(uv, np.float64))
        xd = np.stack([(uv[:, 0] - self.cx) / self.fx, (uv[:, 1] - self.cy) / self.fy], 1)
        xy = xd.copy()
        for _ in range(200):
            x, y = xy[:, 0], xy[:, 1]
            r2 = x * x + y * y
            radial = 1 + self.k1 * r2 + self.k2 * r2 * r2
            dx = 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
            dy = self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y
            new = np.stack([(xd[:, 0] - dx) / radial, (xd[:, 1] - dy) / radial], 1)
            if np.max(np.abs(new - xy)) < 1e-14:
                xy = new
                break
            xy = new
        if not np.isfinite(xy).all() or np.max(np.abs(self.distort(xy) - xd)) > 1e-10:
            raise ValueError('Undistortion did not converge in ' + self.name)
        return xy

    def project(self, X):
        """COLMAP-frame points (N,3) to pixels (N,2) and camera depths (N,)."""
        Xc = np.atleast_2d(X) @ self.R.T + self.t
        z = Xc[:, 2]
        d = self.distort(Xc[:, :2] / z[:, None])
        return np.stack([self.fx * d[:, 0] + self.cx, self.fy * d[:, 1] + self.cy], 1), z

    def ray(self, uv):
        """Unit world-frame ray direction(s) through pixel(s)."""
        xy = self.undistort_pixels(uv)
        d = np.concatenate([xy, np.ones((len(xy), 1))], 1) @ self.R
        return d / np.linalg.norm(d, axis=1, keepdims=True)


class Model:
    """A COLMAP sparse model bound to its transforms.json frame names and a dataparser transform."""

    def __init__(self, colmap_dir, dataparser, sparse=None, transforms=None):
        self.colmap_dir = Path(colmap_dir)
        self.sparse = Path(sparse) if sparse else self.colmap_dir / 'sparse' / '0'
        self.transforms_path = Path(transforms) if transforms else self.colmap_dir / 'transforms.json'
        self.dataparser_path = Path(dataparser)
        self.dataparser = json.loads(self.dataparser_path.read_text())
        self.transforms = json.loads(self.transforms_path.read_text())
        cams = read_cameras_binary(self.sparse / 'cameras.bin')
        imgs = read_images_poses(self.sparse / 'images.bin')
        self.points = read_points3d_binary(self.sparse / 'points3D.bin')
        self.cameras = {iid: Camera(iid, im['name'], cams[im['camera_id']], im['R'], im['t']) for iid, im in imgs.items()}
        self.images = imgs
        self.checks = {}
        self._bind_names()
        self._check_intrinsics(cams)
        self._check_frames()
        self._support = {}

    def _bind_names(self):
        """Click image names resolve through transforms.json (file_path basename -> colmap_im_id)
        and also by COLMAP's own image names. The sorted-name rule (procedure 6a) is checked."""
        self.by_name = {im['name']: iid for iid, im in self.images.items()}
        frames = {}
        for fr in self.transforms['frames']:
            if 'colmap_im_id' not in fr:
                raise ValueError('transforms.json frames must carry colmap_im_id')
            iid = int(fr['colmap_im_id'])
            if iid not in self.images:
                raise ValueError('transforms.json colmap_im_id %d is not in images.bin' % iid)
            frames[Path(fr['file_path']).name] = iid
        if len(set(frames.values())) != len(frames):
            raise ValueError('transforms.json maps two frames to one COLMAP image')
        clash = [n for n in frames if n in self.by_name and self.by_name[n] != frames[n]]
        if clash:
            raise ValueError('Frame names clash with COLMAP image names: ' + ', '.join(clash))
        self.by_name.update(frames)
        sorted_ids = [iid for _, iid in sorted((im['name'], iid) for iid, im in self.images.items())]
        sorted_frames = [iid for _, iid in sorted(frames.items())]
        self.checks['frame_map'] = {'source': 'transforms.json colmap_im_id', 'frames': len(frames),
                                    'agrees_with_sorted_colmap_names': sorted_ids == sorted_frames}
        self.frame_name = {iid: n for n, iid in frames.items()}

    def _check_intrinsics(self, cams):
        """transforms.json intrinsics must equal cameras.bin (else the directories are mispaired)."""
        t = self.transforms
        bad = []
        for cam in self.cameras.values():
            for key, v in (('fl_x', cam.fx), ('fl_y', cam.fy), ('cx', cam.cx), ('cy', cam.cy), ('k1', cam.k1),
                           ('k2', cam.k2), ('p1', cam.p1), ('p2', cam.p2), ('w', cam.width), ('h', cam.height)):
                if key in t and abs(float(t[key]) - v) > 1e-9 * max(1.0, abs(v)):
                    bad.append('%s %s: transforms %r, cameras.bin %r' % (cam.name, key, t[key], v))
        if bad:
            raise ValueError('Intrinsics differ between transforms.json and cameras.bin: ' + '; '.join(bad[:5]))
        models = sorted({c.model for c in self.cameras.values()})
        self.checks['intrinsics'] = {'cameras_bin_models': models, 'transforms_camera_model': t.get('camera_model'),
                                     'equal_to_transforms_json': True}

    def _check_frames(self):
        """Camera centers mapped COLMAP -> dataparser (colmap_to_dataparser) must equal the
        transforms.json route into the export frame (plant_orientation.camera_poses), frame by frame
        through the name mapping. Both routes read the same dataparser file, so this checks that
        transforms.json belongs to this sparse model, that the frame-to-image mapping is right and
        that the pose reading is right. It cannot tell whether the dataparser file is the run's
        (record its hash; with --plant-asset the asset's recorded hash is compared)."""
        stems, C_dp, _ = camera_poses(self.transforms_path, self.dataparser_path)
        diffs = []
        for stem, c in zip(stems, C_dp):
            iid = self.by_name[stem + '.png'] if stem + '.png' in self.by_name else None
            if iid is None:
                cand = [i for n, i in self.by_name.items() if Path(n).stem == stem]
                iid = cand[0] if cand else None
            if iid is None:
                raise ValueError('No COLMAP image for transforms.json frame ' + stem)
            diffs.append(np.linalg.norm(self.to_dataparser(self.cameras[iid].center[None])[0] - c))
        R = np.array(self.dataparser['transform'], np.float64)[:3, :3]
        rec = {'cameras': len(diffs), 'max_center_difference_dataparser': float(max(diffs)),
               'tolerance': 1e-6, 'dataparser_rotation_orthonormal_error': float(np.abs(R @ R.T - np.eye(3)).max()),
               'dataparser_scale': float(self.dataparser['scale'])}
        self.checks['frame_chain'] = rec
        if rec['max_center_difference_dataparser'] > rec['tolerance']:
            raise ValueError('COLMAP and transforms.json camera centers disagree in the dataparser frame by %.3g: '
                             'transforms.json does not belong to this sparse model, or its frame mapping is wrong'
                             % rec['max_center_difference_dataparser'])
        if rec['dataparser_rotation_orthonormal_error'] > 1e-5:
            raise ValueError('Dataparser transform is not a rotation times scale; distances would not scale uniformly')

    def to_dataparser(self, X):
        return colmap_to_dataparser(np.atleast_2d(np.asarray(X, np.float64)), self.dataparser)

    def covariance_to_dataparser(self, S):
        R = np.array(self.dataparser['transform'], np.float64)[:3, :3]
        s = float(self.dataparser['scale'])
        return s * s * R @ S @ R.T

    @property
    def scale(self):
        """Dataparser scene units per COLMAP unit."""
        return float(self.dataparser['scale'])

    def camera(self, name):
        if name not in self.by_name:
            raise ValueError('Image %r is not in the COLMAP model or transforms.json' % name)
        return self.cameras[self.by_name[name]]

    def support_table(self, image_id):
        """(point ids, keypoint pixels (N,2), xyz (N,3)) of COLMAP points whose track includes image_id."""
        if image_id not in self._support:
            ids, uv, X = [], [], []
            xys = self.images[image_id]['xys']
            for pid, (xyz, iids, idxs) in self.points.items():
                hit = np.flatnonzero(iids == image_id)
                for k in hit[:1]:
                    ids.append(pid), uv.append(xys[idxs[k]]), X.append(xyz)
            self._support[image_id] = (np.array(ids, np.int64), np.array(uv).reshape(-1, 2), np.array(X).reshape(-1, 3))
        return self._support[image_id]

    def check_click_images(self, doc):
        """Every clicks-file image must be in the model with matching width and height."""
        for im in doc['images']:
            cam = self.camera(im['name'])
            if (im['width'], im['height']) != (cam.width, cam.height):
                raise ValueError('%s: clicks say %d x %d, camera is %d x %d'
                                 % (im['name'], im['width'], im['height'], cam.width, cam.height))

    def input_records(self):
        paths = [self.sparse / 'cameras.bin', self.sparse / 'images.bin', self.sparse / 'points3D.bin',
                 self.transforms_path, self.dataparser_path]
        return [{'path': str(p.resolve()), 'sha256': sha256(p)} for p in paths]


# ----------------------------------------------------------------------------- multi-view

def triangulate_dlt(cams, xy):
    """Linear DLT from undistorted normalized coordinates xy (N,2) and cameras."""
    A = []
    for cam, (x, y) in zip(cams, xy):
        P = np.hstack([cam.R, cam.t[:, None]])
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.array(A))
    X = vt[-1]
    if abs(X[3]) < 1e-12:
        raise ValueError('DLT point at infinity')
    return X[:3] / X[3]


def _residuals(X, cams, uv):
    return np.concatenate([cam.project(X[None])[0][0] - p for cam, p in zip(cams, uv)])


def _jacobian(X, cams, uv, eps=None):
    eps = eps if eps is not None else 1e-6 * max(1.0, float(np.linalg.norm(X)))
    J = np.zeros((2 * len(cams), 3))
    for k in range(3):
        d = np.zeros(3); d[k] = eps
        J[:, k] = (_residuals(X + d, cams, uv) - _residuals(X - d, cams, uv)) / (2 * eps)
    return J


def refine_point(X0, cams, uv, iterations=50):
    """Levenberg-Marquardt on pixel reprojection error (numeric Jacobian). Returns (X, converged)."""
    X, lam = np.asarray(X0, np.float64).copy(), 1e-3
    r = _residuals(X, cams, uv)
    cost = float(r @ r)
    for _ in range(iterations):
        J = _jacobian(X, cams, uv)
        H, g = J.T @ J, J.T @ r
        while True:
            step = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), -g)
            Xn = X + step
            rn = _residuals(Xn, cams, uv)
            cn = float(rn @ rn)
            if np.isfinite(cn) and cn <= cost:
                lam = max(lam / 10, 1e-12)
                break
            lam *= 10
            if lam > 1e12:
                return X, True
        done = np.linalg.norm(step) < 1e-12 * max(1.0, np.linalg.norm(X)) or cost - cn < 1e-14 * max(cost, 1e-30)
        X, r, cost = Xn, rn, cn
        if done:
            return X, True
    return X, False


def triangulate(cams, uv, params):
    """Multi-view point from >= 2 cameras and their clicked pixels uv (N,2)."""
    uv = np.asarray(uv, np.float64)
    xy = np.concatenate([cam.undistort_pixels(p[None]) for cam, p in zip(cams, uv)])
    X0 = triangulate_dlt(cams, xy)
    X, converged = refine_point(X0, cams, uv)
    proj = [cam.project(X[None]) for cam in cams]
    depths = np.array([z[0] for _, z in proj])
    res = np.array([np.linalg.norm(p[0] - q) for (p, _), q in zip(proj, uv)])
    rays = np.array([(X - cam.center) / np.linalg.norm(X - cam.center) for cam in cams])
    angles = [math.degrees(math.acos(float(np.clip(rays[i] @ rays[j], -1, 1))))
              for i in range(len(cams)) for j in range(i + 1, len(cams))]
    J = _jacobian(X, cams, uv)
    try:
        cov = params['click_sigma_px'] ** 2 * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = np.full((3, 3), np.nan)
    flags = []
    if not (depths > 0).all():
        return {'status': 'refused', 'reason': 'point behind camera(s): ' + ', '.join(
            c.name for c, z in zip(cams, depths) if z <= 0), 'diagnostics': {'depths': floats(depths)}}
    if res.max() > params['max_reprojection_px']:
        flags.append('reprojection_residual_above_threshold')
    if max(angles) < params['min_triangulation_angle_deg']:
        flags.append('triangulation_angle_below_threshold')
    if not converged:
        flags.append('refinement_not_converged')
    diag = {'views': [{'image': c.name, 'image_id': int(c.image_id), 'click': floats(p),
                       'reprojected': floats(q[0]), 'residual_px': float(r), 'depth_colmap': float(z)}
                      for c, p, (q, _), r, z in zip(cams, uv, proj, res, depths)],
            'max_residual_px': float(res.max()), 'rms_residual_px': float(np.sqrt(np.mean(res ** 2))),
            'triangulation_angle_max_deg': float(max(angles)), 'triangulation_angle_min_deg': float(min(angles)),
            'dlt_to_refined_colmap': float(np.linalg.norm(X - X0)), 'refinement_converged': bool(converged),
            'covariance_colmap': [floats(row) for row in cov],
            'covariance_rule': 'click_sigma_px^2 (J^T J)^-1 at the refined point'}
    return {'status': 'flagged' if flags else 'ok', 'flags': flags, 'xyz_colmap': X, 'cov_colmap': cov, 'diagnostics': diag}


# ----------------------------------------------------------------------------- single-view

def ransac_plane(P, tol, trials, seed):
    """Deterministic three-point RANSAC then SVD refinement of the largest consensus."""
    best = np.zeros(len(P), bool)
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-15:
            continue
        good = np.abs((P - a) @ (n / norm)) < tol
        if good.sum() > best.sum():
            best = good
    if best.sum() < 3:
        return None, None, best
    center = P[best].mean(0)
    _, _, vt = np.linalg.svd(P[best] - center, full_matrices=False)
    n = vt[-1]
    # one reselection with the refined plane, then refit
    best2 = np.abs((P - center) @ n) < tol
    if best2.sum() >= 3:
        best = best2
        center = P[best].mean(0)
        _, _, vt = np.linalg.svd(P[best] - center, full_matrices=False)
        n = vt[-1]
    return n, center, best


def local_surface(model, cam, uv, params, exclude_point_ids=(), surface_offset_colmap=0.0):
    """Single-view point: the click's ray meets a robust plane through the COLMAP points observed
    in this image whose keypoints lie near the click. The radius starts at support_radius_px and
    doubles (never above max_support_radius_px) until min_support_plane points are found; the
    last radius tried is used either way. exclude_point_ids removes points (held-out tests).

    surface_offset_colmap (COLMAP units, >= 0) moves the fitted plane along its normal toward the
    camera before the ray is intersected: the clicked object lies that far above the supporting
    surface (for example a tape resting about 1 cm above soil). In the median-depth path the
    offset is subtracted from the depth along the ray direction (flagged). ruler_scale converts a
    metric offset with the current scale estimate and iterates."""
    uv = np.asarray(uv, np.float64)
    ids, kp, X = model.support_table(cam.image_id)
    dist = np.linalg.norm(kp - uv, axis=1)
    if len(exclude_point_ids):
        dist = np.where(np.isin(ids, np.asarray(list(exclude_point_ids), np.int64)), np.inf, dist)
    radii, r = [], float(params['support_radius_px'])
    while True:
        radii.append(r)
        if np.count_nonzero(dist <= r) >= params['min_support_plane'] or 2 * r > params['max_support_radius_px']:
            break
        r *= 2
    near = dist <= r
    S, sid = X[near], ids[near]
    C, d = cam.center, cam.ray(uv[None])[0]
    z = (S @ cam.R.T + cam.t)[:, 2] if len(S) else np.zeros(0)
    diag = {'image': cam.name, 'image_id': int(cam.image_id), 'click': floats(uv),
            'support_radius_px': r, 'radii_tried_px': radii,
            'support_counts_by_radius': [int(np.count_nonzero(dist <= q)) for q in radii],
            'support_count': int(len(S)), 'support_point_ids': [int(i) for i in sid]}
    if len(S) < params['min_support']:
        return {'status': 'refused', 'reason': '%d support points within %g px < min_support %d'
                % (len(S), r, params['min_support']), 'diagnostics': diag}
    zmed = float(np.median(z))
    spread = float(1.4826 * np.median(np.abs(z - zmed)))
    diag.update(support_depth_median_colmap=zmed, support_depth_min_colmap=float(z.min()),
                support_depth_max_colmap=float(z.max()), support_depth_robust_spread_colmap=spread,
                support_depth_spread_rel=spread / zmed)
    flags = []
    if len(S) >= params['min_support_plane']:
        tol = params['plane_tol_rel_depth'] * zmed
        n, P0, inl = ransac_plane(S, tol, params['ransac_trials'], params['ransac_seed'])
        diag['plane_tolerance_colmap'] = tol
        if n is not None and inl.sum() < params['min_plane_inliers']:
            diag['plane_inliers'] = int(inl.sum())
            flags.append('too_few_plane_inliers_used_median_depth')
        elif n is not None:
            cosi = abs(float(n @ d))
            incidence = math.degrees(math.acos(min(1.0, cosi)))
            resid = (S[inl] - P0) @ n
            rms = float(np.sqrt(np.sum(resid ** 2) / (int(inl.sum()) - 3)))
            share = float(inl.mean())
            diag.update(plane_normal_colmap=floats(n), plane_point_colmap=floats(P0), plane_inliers=int(inl.sum()),
                        plane_inlier_share=share, plane_residual_rms_colmap=rms, ray_incidence_deg=incidence)
            if share < params['min_inlier_share']:
                flags.append('mixed_surfaces')
            if incidence <= params['max_incidence_deg']:
                if float(n @ (C - P0)) < 0:
                    n = -n  # normal toward the camera
                s0 = float(n @ (P0 - C)) / float(n @ d)
                zp0 = float((cam.R @ (C + s0 * d) + cam.t)[2])
                P0s = P0 + surface_offset_colmap * n
                s = float(n @ (P0s - C)) / float(n @ d)
                Xp = C + s * d
                zp = float((cam.R @ Xp + cam.t)[2])
                if zp <= 0 or zp0 <= 0:
                    return {'status': 'refused', 'reason': 'plane intersection behind the camera', 'diagnostics': diag}
                zi = z[inl]
                if not (zi.min() <= zp0 <= zi.max()):
                    flags.append('intersection_outside_support_depth_range')
                signed = (S - P0) @ n
                diag.update(plane_normal_colmap=floats(n), surface_offset_colmap=float(surface_offset_colmap),
                            surface_hit_depth_colmap=zp0,
                            support_points=[{'point_id': int(i), 'keypoint_distance_px': float(kd), 'inlier': bool(f),
                                             'signed_distance_from_plane_toward_camera_colmap': float(sd),
                                             'depth_minus_surface_hit_depth_colmap': float(zz - zp0)}
                                            for i, kd, f, sd, zz in zip(sid, dist[near], inl, signed, z)],
                            support_points_rule='signed distance positive = on the camera side of the fitted plane; '
                                                'depth difference negative = nearer the camera than the bare-surface ray hit')
                sigma_ray = rms * math.sqrt(1 + 3 / int(inl.sum())) / max(cosi, 1e-12)
                diag.update(submethod='plane', depth_colmap=zp, along_ray_sigma_colmap=sigma_ray,
                            along_ray_sigma_rule='prediction sigma: plane residual RMS (n-3 dof) x sqrt(1 + 3/n) / |cos incidence|')
                return _single_result(model, cam, Xp, d, zp, sigma_ray, flags, diag, params)
            flags.append('grazing_plane_used_median_depth')
        else:
            flags.append('no_plane_used_median_depth')
    # median depth along the optical axis
    zray = float((cam.R @ d)[2])
    Xm = C + (zmed / zray - surface_offset_colmap) * d
    if surface_offset_colmap:
        flags.append('surface_offset_applied_along_ray_without_plane')
    diag.update(surface_offset_colmap=float(surface_offset_colmap),
                support_points=[{'point_id': int(i), 'keypoint_distance_px': float(kd),
                                 'depth_minus_median_depth_colmap': float(zz - zmed)} for i, kd, zz in zip(sid, dist[near], z)])
    sigma = spread / max(zray, 1e-12)
    if spread / zmed > params['max_depth_spread_rel']:
        flags.append('support_depth_spread_above_threshold')
    diag.update(submethod='median-depth', depth_colmap=float((cam.R @ Xm + cam.t)[2]), along_ray_sigma_colmap=sigma,
                along_ray_sigma_rule='robust depth spread (1.4826 MAD) of the support, along the ray')
    return _single_result(model, cam, Xm, d, zmed, sigma, flags, diag, params)


def _single_result(model, cam, X, d, depth, sigma_ray, flags, diag, params):
    lateral = params['click_sigma_px'] * depth / cam.fx
    P = np.eye(3) - np.outer(d, d)
    cov = sigma_ray ** 2 * np.outer(d, d) + lateral ** 2 * P
    diag['covariance_colmap'] = [floats(r) for r in cov]
    diag['covariance_rule'] = 'along-ray sigma^2 d d^T + (click_sigma_px depth / f)^2 (I - d d^T)'
    q = cam.project(X[None])[0][0]
    diag['reprojection_check_px'] = float(np.linalg.norm(q - np.array(diag['click'])))
    return {'status': 'flagged' if flags else 'ok', 'flags': flags, 'xyz_colmap': X, 'cov_colmap': cov, 'diagnostics': diag}


# ----------------------------------------------------------------------------- all labels

def solve_label(model, obs, params, exclude_point_ids=(), surface_offset_colmap=0.0):
    """One label: multi-view when seen in 2+ images, single-view-local-surface when in one."""
    cams = [model.camera(name) for name, _, _ in obs]
    uv = np.array([[x, y] for _, x, y in obs])
    if len(obs) >= 2:
        if len({c.image_id for c in cams}) != len(cams):
            raise ValueError('Two click images resolve to one COLMAP image')
        rec, method = triangulate(cams, uv, params), 'multi-view'
    else:
        rec, method = local_surface(model, cams[0], uv[0], params, exclude_point_ids,
                                    surface_offset_colmap), 'single-view-local-surface'
    rec['method'] = method
    if 'xyz_colmap' in rec:
        rec['xyz_dataparser'] = model.to_dataparser(rec['xyz_colmap'])[0]
        rec['cov_dataparser'] = model.covariance_to_dataparser(rec['cov_colmap'])
        rec['diagnostics']['covariance_dataparser'] = [floats(r) for r in rec['cov_dataparser']]
        rec['diagnostics']['sigma_dataparser'] = floats(np.sqrt(np.clip(np.diag(rec['cov_dataparser']), 0, None)))
    return rec


def solve_all(model, doc, params):
    model.check_click_images(doc)
    return {label: solve_label(model, obs, params) for label, obs in observations(doc).items()}


def point_record(label, rec):
    out = {'label': label, 'method': rec['method'], 'status': rec['status'], 'flags': rec.get('flags', []),
           'diagnostics': rec['diagnostics']}
    if 'xyz_colmap' in rec:
        out['xyz_colmap'] = floats(rec['xyz_colmap'])
        out['xyz_dataparser'] = floats(rec['xyz_dataparser'])
    else:
        out['reason'] = rec['reason']
    return out


# ----------------------------------------------------------------------------- leaves

LEAF_RE = re.compile(r'^(?P<leaf>.+)_(?P<end>base|tip)$')


def leaf_pairs(labels):
    """{leafname: {'base': label, 'tip': label}} for labels following {leaf}_base / {leaf}_tip."""
    out = {}
    for label in labels:
        m = LEAF_RE.match(label)
        if m and not label.startswith('tape_'):
            out.setdefault(m.group('leaf'), {})[m.group('end')] = label
    return out


def segment_distances(P, a, b):
    ab = b - a
    L2 = float(ab @ ab)
    t = np.zeros(len(P)) if L2 == 0 else np.clip((P - a) @ ab / L2, 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def load_plant_leaves(asset_dir):
    """Leaf point arrays from a plant-asset folder, hash-checked against plant.json."""
    asset_dir = Path(asset_dir)
    plant = json.loads((asset_dir / 'plant.json').read_text())
    if plant.get('schema') != 'plant-asset':
        raise ValueError('Not a plant-asset folder: ' + str(asset_dir))
    leaves, inputs = [], [{'path': str((asset_dir / 'plant.json').resolve()), 'sha256': sha256(asset_dir / 'plant.json')}]
    for leaf in plant['leaves']:
        path = asset_dir / leaf['point_geometry']['path']
        h = sha256(path)
        if h != leaf['point_geometry']['sha256']:
            raise ValueError('Leaf file hash differs from plant.json: ' + str(path))
        with np.load(path) as z:
            X = np.asarray(z['X'], np.float64)
        leaves.append({'id': leaf['id'], 'role': leaf.get('role'), 'X': X})
        inputs.append({'path': str(path.resolve()), 'sha256': h})
    return plant, leaves, inputs


def asset_dataparser_hash(asset_dir):
    """The dataparser_transforms.json hash the asset's original organ evidence recorded, or None."""
    p = Path(asset_dir) / 'source-evidence' / 'organ-evidence-original.json'
    if not p.exists():
        return None
    h = json.loads(p.read_text()).get('camera_inputs', {}).get('dataparser_transforms')
    return h if isinstance(h, str) and re.fullmatch('[0-9a-f]{64}', h) else None


def associate_leaf(a, b, leaves):
    """Nearest plant-asset leaf to segment a-b by median point-to-segment distance; report only."""
    rows = sorted(({'leaf_id': lf['id'], 'role': lf['role'], 'points': int(len(lf['X'])),
                    'median_distance': float(np.median(segment_distances(lf['X'], a, b)))} for lf in leaves),
                  key=lambda r: (r['median_distance'], r['leaf_id']))
    rec = {'nearest': rows[0]['leaf_id'], 'nearest_median_distance': rows[0]['median_distance'], 'ranking': rows,
           'rule': 'median distance from each leaf point (dataparser frame) to the base-tip segment; report only, never relabels'}
    if len(rows) > 1:
        rec['second'] = rows[1]['leaf_id']
        rec['margin_to_second'] = rows[1]['median_distance'] - rows[0]['median_distance']
        rec['ratio_second_to_nearest'] = (rows[1]['median_distance'] / rows[0]['median_distance']
                                          if rows[0]['median_distance'] > 0 else math.inf)
    return rec


def load_scale(path, model):
    s = json.loads(Path(path).read_text())
    if s.get('schema') != 'scale-estimate' or s.get('schema_version') != 1 or s.get('source_frame') != 'dataparser':
        raise ValueError('Expected a scale-estimate v1 in the dataparser frame: ' + str(path))
    dp = sha256(model.dataparser_path)
    if dp not in {i.get('sha256') for i in s.get('inputs', [])}:
        raise ValueError('Scale estimate was made with a different dataparser transform')
    return s


# ----------------------------------------------------------------------------- CLI

def code_records():
    here = Path(__file__).resolve().parent
    return [{'path': str((here / f).resolve()), 'sha256': sha256(here / f)}
            for f in ('click_points.py', 'ruler_scale.py', 'plant_orientation.py', 'geometry.py') if (here / f).exists()]


def source_revision():
    import subprocess
    root = Path(__file__).resolve().parents[3]
    def git(*args):
        r = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    return {'root': str(root), 'commit': git('rev-parse', 'HEAD'), 'status': git('status', '--short')}


def load_params(path):
    params = dict(DEFAULTS)
    if path:
        params.update(json.loads(Path(path).read_text()))
    return check_parameters(params)


def write_new(path, value):
    path = Path(path)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def jsonable(v):
    """Replace nonfinite floats by None (JSON has no NaN)."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    return v


def cmd_triangulate(args):
    params = load_params(args.parameters)
    out = Path(args.output)
    if out.exists():
        raise SystemExit('Output exists: ' + str(out))
    doc = load_clicks(args.clicks)
    model = Model(args.colmap, args.dataparser, args.sparse, args.transforms)
    results = solve_all(model, doc, params)
    inputs = model.input_records()
    scale = load_scale(args.scale, model) if args.scale else None
    if scale:
        inputs.append({'path': str(Path(args.scale).resolve()), 'sha256': sha256(args.scale)})
    leaves = None
    if args.plant_asset:
        _, leaves, leaf_inputs = load_plant_leaves(args.plant_asset)
        inputs += leaf_inputs
        recorded, used = asset_dataparser_hash(args.plant_asset), sha256(model.dataparser_path)
        model.checks['plant_asset_dataparser'] = {'recorded_sha256': recorded, 'used_sha256': used,
                                                  'equal': None if recorded is None else recorded == used}
        if recorded is not None and recorded != used:
            raise ValueError('The plant asset was built with a different dataparser_transforms.json (%s) than %s'
                             % (recorded, model.dataparser_path))
    chords = {}
    for leaf, ends in sorted(leaf_pairs(results).items()):
        rec = {'base': ends.get('base'), 'tip': ends.get('tip')}
        if set(ends) != {'base', 'tip'}:
            chords[leaf] = dict(rec, status='incomplete', reason='needs both _base and _tip')
            continue
        rb, rt = results[ends['base']], results[ends['tip']]
        if 'xyz_dataparser' not in rb or 'xyz_dataparser' not in rt:
            chords[leaf] = dict(rec, status='refused', reason='an endpoint was refused')
            continue
        a, b = rb['xyz_dataparser'], rt['xyz_dataparser']
        L = float(np.linalg.norm(b - a))
        u = (b - a) / L if L > 0 else np.zeros(3)
        sigma = float(np.sqrt(max(u @ (rb['cov_dataparser'] + rt['cov_dataparser']) @ u, 0)))
        rec.update(status='ok' if rb['status'] == rt['status'] == 'ok' else 'flagged',
                   chord_scene=L, chord_sigma_scene=sigma, endpoint_methods=[rb['method'], rt['method']])
        if scale:
            m = float(scale['meters_per_scene_unit'])
            sm = float(scale['std_meters_per_scene_unit'])
            rec.update(chord_m=L * m, chord_sigma_m=math.hypot(sigma * m, L * sm))
        if leaves:
            rec['association'] = associate_leaf(a, b, leaves)
        chords[leaf] = rec
    result = {'schema': POINTS_SCHEMA, 'schema_version': POINTS_SCHEMA_VERSION, 'capture_id': args.capture_id,
              'source_frame': 'dataparser', 'units': 'scale_free_scene_units',
              'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
              'parameters': params, 'checks': model.checks,
              'points': [point_record(k, v) for k, v in results.items()],
              'leaf_chords': chords, 'scale_estimate': None if not scale else {
                  'meters_per_scene_unit': scale['meters_per_scene_unit'],
                  'std_meters_per_scene_unit': scale['std_meters_per_scene_unit']},
              'inputs': inputs, 'decisions': [{'path': str(Path(args.clicks).resolve()), 'sha256': sha256(args.clicks), 'by': 'Alex'}],
              'code': code_records(), 'source_revision': source_revision(), 'command': sys.argv}
    write_new(out, jsonable(result))
    for p in result['points']:
        xyz = p.get('xyz_dataparser')
        print('%-24s %-26s %-8s %s %s' % (p['label'], p['method'], p['status'],
                                          'refused' if xyz is None else '[%.5f %.5f %.5f]' % tuple(xyz),
                                          ','.join(p['flags'])))
    for leaf, c in chords.items():
        print('leaf %s: %s' % (leaf, ', '.join('%s=%s' % (k, c[k]) for k in ('status', 'chord_scene', 'chord_m') if k in c)))
    return 0


def add_model_args(ap):
    ap.add_argument('--clicks', required=True, help='image-clicks v1 JSON (Alex\'s decisions)')
    ap.add_argument('--colmap', required=True, help='COLMAP capture folder holding sparse/0 and transforms.json')
    ap.add_argument('--dataparser', required=True, help='the trained run\'s dataparser_transforms.json')
    ap.add_argument('--sparse', help='override the sparse model folder (default COLMAP/sparse/0)')
    ap.add_argument('--transforms', help='override transforms.json (default COLMAP/transforms.json)')
    ap.add_argument('--parameters', help='JSON object overriding click_points.DEFAULTS')
    ap.add_argument('--capture-id', required=True)
    ap.add_argument('--output', required=True, help='new JSON file (refused if it exists)')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='command', required=True)
    t = sub.add_parser('triangulate', help='all labeled points to 3D, leaf chords and optional leaf association')
    add_model_args(t)
    t.add_argument('--plant-asset', help='plant-asset folder for leaf association (report only)')
    t.add_argument('--scale', help='scale-estimate JSON for chords in meters')
    sub.add_parser('validate', help='validate a clicks file only').add_argument('clicks')
    args = ap.parse_args(argv)
    try:
        if args.command == 'validate':
            doc = load_clicks(args.clicks)
            obs = observations(doc)
            print('valid: %d points, %d labels (%d multi-view, %d single-view)' % (
                len(doc['points']), len(obs), sum(len(v) > 1 for v in obs.values()), sum(len(v) == 1 for v in obs.values())))
            return 0
        return cmd_triangulate(args)
    except ValueError as e:
        print('error: ' + str(e), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
