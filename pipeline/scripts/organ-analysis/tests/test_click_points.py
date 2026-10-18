"""Tests for click_points.py and ruler_scale.py: synthetic cameras with distortion and known 3D
points, refusal cases, and (when the private data exist) a real-camera regression on the
October 2 Welzo M1 model using COLMAP's own keypoints as fake clicks."""
import contextlib
import io
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import click_points as cp
import ruler_scale as rs

P = cp.check_parameters(dict(cp.DEFAULTS))
W, H = 4096, 3072
A = np.array([[1.0, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0]])  # nerfstudio applied_transform for COLMAP


def rot_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(t + 1)
        q = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = s / 4
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.asarray(q, float)
    return q / np.linalg.norm(q)


def look_at(center, target, up=(0, 0, 1.0)):
    """COLMAP world-to-camera (R, t) for a camera at center looking at target (OpenCV axes)."""
    f = target - center
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f])
    return R, -R @ center


class Scene:
    """Synthetic COLMAP model on disk: OPENCV cameras with distortion, poses, points, tracks,
    transforms.json and a dataparser transform (rotation, translation, scale)."""

    def __init__(self, tmp, centers, target=np.zeros(3), params=(2900.0, 2950.0, 2048.0, 1536.0, -0.05, 0.01, 1e-3, -5e-4),
                 model_id=4, dp_scale=0.37, dp_angle=0.7):
        self.dir = Path(tmp) / 'colmap'
        (self.dir / 'sparse' / '0').mkdir(parents=True)
        self.params, self.model_id = np.array(params, float), model_id
        self.poses = [look_at(np.asarray(c, float), np.asarray(target, float)) for c in centers]
        cam = {'model': 'x', 'model_id': model_id, 'width': W, 'height': H, 'params': self.params}
        self.cams = [cp.Camera(i + 1, 'IMG_%04d.png' % (i + 1), cam, R, t) for i, (R, t) in enumerate(self.poses)]
        c, s = np.cos(dp_angle), np.sin(dp_angle)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        T = np.hstack([Rz, [[0.1], [-0.2], [0.3]]])
        self.dp = {'transform': (T @ np.vstack([A, [0, 0, 0, 1]])).tolist(), 'scale': dp_scale}
        self.dp_path = Path(tmp) / 'dataparser_transforms.json'
        self.dp_path.write_text(json.dumps(self.dp))
        self.points = []  # (pid, xyz, [(iid, idx)])
        self.kp = {c.image_id: [] for c in self.cams}

    def add_point(self, X, images=None, noise=0.0, rng=None):
        pid = len(self.points) + 1
        track = []
        for cam in self.cams:
            if images is not None and cam.image_id not in images:
                continue
            uv, z = cam.project(np.asarray(X, float)[None])
            if z[0] <= 0:
                continue
            uv = uv[0] + (rng.normal(0, noise, 2) if noise else 0)
            track.append((cam.image_id, len(self.kp[cam.image_id])))
            self.kp[cam.image_id].append((uv[0], uv[1], pid))
        self.points.append((pid, np.asarray(X, float), track))
        return pid

    def write(self, transforms_extra=None):
        sp = self.dir / 'sparse' / '0'
        with open(sp / 'cameras.bin', 'wb') as f:
            f.write(struct.pack('<Q', 1))
            f.write(struct.pack('<iiQQ', 1, self.model_id, W, H))
            f.write(struct.pack('<%dd' % len(self.params), *self.params))
        with open(sp / 'images.bin', 'wb') as f:
            f.write(struct.pack('<Q', len(self.cams)))
            for cam in self.cams:
                f.write(struct.pack('<i4d3di', cam.image_id, *rot_to_quat(cam.R), *cam.t, 1))
                f.write(cam.name.encode() + b'\0')
                kps = self.kp[cam.image_id]
                f.write(struct.pack('<Q', len(kps)))
                for x, y, pid in kps:
                    f.write(struct.pack('<ddq', x, y, pid))
        with open(sp / 'points3D.bin', 'wb') as f:
            f.write(struct.pack('<Q', len(self.points)))
            for pid, X, track in self.points:
                f.write(struct.pack('<QdddBBBd', pid, *X, 0, 0, 0, 0.5))
                f.write(struct.pack('<Q', len(track)))
                for iid, j in track:
                    f.write(struct.pack('<ii', iid, j))
        frames = []
        for cam in self.cams:
            c2w = np.eye(4)
            c2w[:3, :3], c2w[:3, 3] = cam.R.T, cam.center
            c2w[:3, 1:3] *= -1
            c2w = c2w[[0, 2, 1, 3], :]
            c2w[2, :] *= -1
            frames.append({'file_path': 'images/frame_%05d.png' % cam.image_id, 'transform_matrix': c2w.tolist(),
                           'colmap_im_id': cam.image_id})
        fx, fy, cx, cy, k1, k2, p1, p2 = self.params if self.model_id == 4 else (self.params[0], self.params[0], *self.params[1:3], self.params[3], 0, 0, 0)
        t = {'w': W, 'h': H, 'fl_x': fx, 'fl_y': fy, 'cx': cx, 'cy': cy, 'k1': k1, 'k2': k2, 'p1': p1, 'p2': p2,
             'camera_model': 'OPENCV', 'applied_transform': A.tolist(), 'frames': frames}
        t.update(transforms_extra or {})
        (self.dir / 'transforms.json').write_text(json.dumps(t))
        return cp.Model(self.dir, self.dp_path)

    def click(self, label, X, images, noise=0.0, rng=None, offset=None):
        out = []
        for cam in self.cams:
            if cam.image_id in images:
                uv = cam.project(np.asarray(X, float)[None])[0][0]
                if noise:
                    uv = uv + rng.normal(0, noise, 2)
                if offset is not None and cam.image_id == images[0]:
                    uv = uv + offset
                out.append({'image': 'frame_%05d.png' % cam.image_id, 'label': label, 'x': float(uv[0]), 'y': float(uv[1])})
        return out


def clicks_doc(points, names=None):
    names = names or sorted({p['image'] for p in points})
    return {'schema': 'image-clicks', 'schema_version': 1, 'tool': 'click_points.html', 'tool_version': 'test',
            'created': '2026-10-18T12:00:00+09:00', 'pixel_convention': cp.PIXEL_CONVENTION,
            'images': [{'name': n, 'width': W, 'height': H} for n in names], 'points': points}


def ring(n=8, radius=1.2, height=0.6, phase=0.0):
    return [[radius * np.cos(phase + 2 * np.pi * i / n), radius * np.sin(phase + 2 * np.pi * i / n), height] for i in range(n)]


class ClicksValidation(unittest.TestCase):
    def good(self):
        return clicks_doc([{'image': 'frame_00024.png', 'label': 'tape_5cm', 'x': 1234.5, 'y': 678.25},
                           {'image': 'frame_00023.png', 'label': 'tape_5cm', 'x': 10, 'y': 20}])

    def test_good_file_and_observations(self):
        doc = cp.validate_clicks(self.good())
        self.assertEqual(cp.observations(doc), {'tape_5cm': [('frame_00024.png', 1234.5, 678.25), ('frame_00023.png', 10.0, 20.0)]})

    def test_refusals(self):
        cases = {
            'schema': lambda d: d.update(schema='image-click'),
            'version': lambda d: d.update(schema_version=2),
            'convention': lambda d: d.update(pixel_convention='pixel centers at integers'),
            'unknown image': lambda d: d['points'][0].update(image='frame_00099.png'),
            'duplicate': lambda d: d['points'].append(dict(d['points'][0], x=5)),
            'out of bounds x': lambda d: d['points'][0].update(x=4096.5),
            'negative y': lambda d: d['points'][0].update(y=-0.1),
            'nonfinite': lambda d: d['points'][0].update(x=float('nan')),
            'boolean': lambda d: d['points'][0].update(x=True),
            'empty label': lambda d: d['points'][0].update(label=' '),
            'duplicate image': lambda d: d['images'].append(dict(d['images'][0])),
            'size': lambda d: d['images'][0].update(width=0),
            'no points': lambda d: d.update(points=[]),
        }
        for name, change in cases.items():
            d = self.good()
            change(d)
            with self.subTest(name), self.assertRaises(ValueError):
                cp.validate_clicks(d)

    def test_edges_are_inside(self):
        d = self.good()
        d['points'][0].update(x=0, y=3072)
        cp.validate_clicks(d)


class Distortion(unittest.TestCase):
    def test_undistort_inverts_opencv_distortion(self):
        cam = cp.Camera(1, 'a', {'model': 'OPENCV', 'model_id': 4, 'width': W, 'height': H,
                                 'params': np.array([2900, 2950, 2048, 1536, -0.05, 0.01, 1e-3, -5e-4])}, np.eye(3), np.zeros(3))
        rng = np.random.default_rng(0)
        X = np.stack([rng.uniform(-0.6, 0.6, 500), rng.uniform(-0.45, 0.45, 500), np.ones(500)], 1)
        uv, _ = cam.project(X)
        xy = cam.undistort_pixels(uv)
        self.assertLess(np.abs(xy - X[:, :2]).max(), 1e-11)
        self.assertLess(np.abs(np.cross(cam.ray(uv), X / np.linalg.norm(X, axis=1, keepdims=True))).max(), 1e-11)

    def test_unsupported_model_refused(self):
        with self.assertRaises(ValueError):
            cp.Camera(1, 'a', {'model': 'unsupported-5', 'model_id': 5, 'width': W, 'height': H, 'params': np.zeros(8)},
                      np.eye(3), np.zeros(3))


class MultiView(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.scene = Scene(self.tmp.name, ring())
        self.model = self.scene.write()

    def tearDown(self):
        self.tmp.cleanup()

    def test_frame_chain_checks(self):
        self.assertTrue(self.model.checks['frame_map']['agrees_with_sorted_colmap_names'])
        self.assertLess(self.model.checks['frame_chain']['max_center_difference_dataparser'], 1e-9)

    def test_wrong_frame_mapping_refused(self):
        t = json.loads((self.scene.dir / 'transforms.json').read_text())
        t['frames'][0]['colmap_im_id'], t['frames'][1]['colmap_im_id'] = 2, 1
        (self.scene.dir / 'transforms.json').write_text(json.dumps(t))
        with self.assertRaisesRegex(ValueError, 'camera centers disagree'):
            cp.Model(self.scene.dir, self.scene.dp_path)

    def test_points_follow_colmap_to_dataparser_not_transforms_route(self):
        # The procedure 6a trap: the saved transform applied to transforms.json-frame points
        # (raw COLMAP times applied_transform) is wrong. The model maps raw COLMAP points.
        X = np.array([[0.1, -0.05, 0.08]])
        good = self.model.to_dataparser(X)[0]
        trap = cp.colmap_to_dataparser(X @ A[:, :3].T, self.scene.dp)[0]
        self.assertGreater(np.linalg.norm(good - trap), 0.01)
        stems, C, _ = cp.camera_poses(self.scene.dir / 'transforms.json', self.scene.dp_path)
        self.assertLess(np.abs(C[0] - self.model.to_dataparser(self.scene.cams[0].center[None])[0]).max(), 1e-12)

    def test_mismatched_intrinsics_refused(self):
        t = json.loads((self.scene.dir / 'transforms.json').read_text())
        t['k1'] = 0.0
        (self.scene.dir / 'transforms.json').write_text(json.dumps(t))
        with self.assertRaisesRegex(ValueError, 'Intrinsics differ'):
            cp.Model(self.scene.dir, self.scene.dp_path)

    def test_noise_free_recovery_is_exact(self):
        rng = np.random.default_rng(1)
        X = rng.uniform(-0.15, 0.15, (20, 3))
        pts = sum((self.scene.click('p%d' % i, x, [1, 3, 6]) for i, x in enumerate(X)), [])
        res = cp.solve_all(self.model, clicks_doc(pts), P)
        for i, x in enumerate(X):
            r = res['p%d' % i]
            self.assertEqual((r['method'], r['status']), ('multi-view', 'ok'))
            self.assertLess(np.linalg.norm(r['xyz_colmap'] - x), 1e-9)
            self.assertLess(np.linalg.norm(r['xyz_dataparser'] - cp.colmap_to_dataparser(x[None], self.scene.dp)[0]), 1e-9)
            self.assertLess(r['diagnostics']['max_residual_px'], 1e-6)

    def test_noisy_recovery_error_and_covariance(self):
        rng = np.random.default_rng(2)
        X = rng.uniform(-0.15, 0.15, (200, 3))
        pts = sum((self.scene.click('p%d' % i, x, [1, 2, 4, 7], noise=0.7, rng=rng) for i, x in enumerate(X)), [])
        res = cp.solve_all(self.model, clicks_doc(pts), dict(P, click_sigma_px=0.7))
        err = np.array([np.linalg.norm(res['p%d' % i]['xyz_colmap'] - x) for i, x in enumerate(X)])
        # depth ~1.3, f ~2900: 0.7 px is ~3e-4 laterally; wide baseline keeps depth error similar
        self.assertLess(np.median(err), 6e-4)
        z = np.array([(res['p%d' % i]['xyz_colmap'] - x) @ np.linalg.solve(res['p%d' % i]['cov_colmap'], res['p%d' % i]['xyz_colmap'] - x)
                      for i, x in enumerate(X)])
        self.assertAlmostEqual(np.mean(z), 3.0, delta=0.6)  # chi-square with 3 dof: covariance is calibrated
        self.assertTrue(all(res['p%d' % i]['diagnostics']['triangulation_angle_max_deg'] > 30 for i in range(len(X))))

    def test_mismatched_click_is_flagged(self):
        pts = self.scene.click('leaf_tip', np.array([0.05, 0.02, 0.1]), [1, 3, 5], offset=np.array([25.0, 0]))
        r = cp.solve_all(self.model, clicks_doc(pts), P)['leaf_tip']
        self.assertEqual(r['status'], 'flagged')
        self.assertIn('reprojection_residual_above_threshold', r['flags'])
        self.assertGreater(r['diagnostics']['max_residual_px'], P['max_reprojection_px'])

    def test_small_angle_flagged(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        scene = Scene(tmp.name, [[1.2, 0, 0.6], [1.2, 0.02, 0.6]])
        model = scene.write()
        r = cp.solve_all(model, clicks_doc(scene.click('x', np.zeros(3), [1, 2])), P)['x']
        self.assertIn('triangulation_angle_below_threshold', r['flags'])
        self.assertLess(r['diagnostics']['triangulation_angle_max_deg'], 2)

    def test_behind_camera_refused(self):
        # rays from opposite cameras that meet only behind both
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        scene = Scene(tmp.name, [[1.2, -0.1, 0.0], [1.2, 0.1, 0.0]], params=(2900, 2900, 2048, 1536, 0, 0, 0, 0))
        # diverging rays: the left camera clicks far left, the right camera far right
        r = cp.triangulate([scene.cams[0], scene.cams[1]], np.array([[2048 - 1500, 1536.0], [2048 + 1500, 1536.0]]), P)
        self.assertEqual(r['status'], 'refused')

    def test_model_image_checks(self):
        doc = clicks_doc(self.scene.click('x', np.zeros(3), [1, 2]))
        doc['images'][0]['width'] = 2048
        for p in doc['points']:
            p['x'] = min(p['x'], 2048)
        with self.assertRaisesRegex(ValueError, 'clicks say'):
            cp.solve_all(self.model, cp.validate_clicks(doc), P)
        doc = clicks_doc([{'image': 'frame_00099.png', 'label': 'x', 'x': 1, 'y': 1}])
        with self.assertRaisesRegex(ValueError, 'not in the COLMAP model'):
            cp.solve_all(self.model, doc, P)


def plane_scene(tmp, normal=(0.1, -0.2, 1.0), n_support=5000, noise=0.0, extra_layer=None, seed=3):
    """Scene with support points on a plane z ~ 0 (tilted), observed in all cameras."""
    scene = Scene(tmp, ring(phase=0.3))
    rng = np.random.default_rng(seed)
    n = np.asarray(normal, float)
    n /= np.linalg.norm(n)
    e1 = np.cross(n, [0, 1.0, 0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    for _ in range(n_support):
        a, b = rng.uniform(-0.3, 0.3, 2)
        scene.add_point(a * e1 + b * e2 + n * rng.normal(0, noise) if noise else a * e1 + b * e2)
    if extra_layer is not None:
        for _ in range(extra_layer[1]):
            a, b = rng.uniform(-0.3, 0.3, 2)
            scene.add_point(a * e1 + b * e2 - n * extra_layer[0])
    return scene, n, e1, e2


class SingleView(unittest.TestCase):
    def test_plane_intersection(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene, n, e1, e2 = plane_scene(tmp)
            model = scene.write()
            X = 0.05 * e1 - 0.03 * e2
            r = cp.solve_all(model, clicks_doc(scene.click('tape_5cm', X, [2])), P)['tape_5cm']
            self.assertEqual((r['method'], r['status'], r['diagnostics']['submethod']), ('single-view-local-surface', 'ok', 'plane'))
            self.assertLess(np.linalg.norm(r['xyz_colmap'] - X), 1e-9)
            self.assertGreaterEqual(r['diagnostics']['support_count'], P['min_support_plane'])
            self.assertLess(r['diagnostics']['reprojection_check_px'], 1e-6)

    def test_noisy_surface_error_within_reported_sigma(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene, n, e1, e2 = plane_scene(tmp, noise=0.002)
            model = scene.write()
            rng = np.random.default_rng(4)
            ratios = []
            for i in range(30):
                X = rng.uniform(-0.2, 0.2) * e1 + rng.uniform(-0.2, 0.2) * e2
                r = cp.solve_all(model, clicks_doc(scene.click('p', X, [3])), P)['p']
                self.assertEqual(r['diagnostics']['submethod'], 'plane')
                d = r['xyz_colmap'] - X
                ratios.append(abs(d @ n) / (r['diagnostics']['along_ray_sigma_colmap'] * abs(model.cameras[3].ray(
                    np.array([r['diagnostics']['click']]))[0] @ n)))
            self.assertLess(np.median(ratios), 2.0)

    def test_suspended_mark_lands_on_the_surface_below(self):
        """The trough risk: a mark 3 cm above the support surface is placed on the surface, not on
        the tape. Single-view cannot see this; only the collinearity check can."""
        with tempfile.TemporaryDirectory() as tmp:
            scene, n, e1, e2 = plane_scene(tmp)
            model = scene.write()
            X = 0.03 * n
            r = cp.solve_all(model, clicks_doc(scene.click('tape_9cm', X, [2])), P)['tape_9cm']
            self.assertEqual(r['status'], 'ok')
            self.assertLess(abs((r['xyz_colmap']) @ n), 1e-9)  # on the plane
            self.assertGreater(np.linalg.norm(r['xyz_colmap'] - X), 0.03)

    def test_mixed_surfaces_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene, n, e1, e2 = plane_scene(tmp, n_support=1500, extra_layer=(0.08, 1500))
            model = scene.write()
            r = cp.solve_all(model, clicks_doc(scene.click('p', np.zeros(3), [2])),
                             dict(P, support_radius_px=150.0))['p']
            self.assertIn('mixed_surfaces', r['flags'])
            self.assertEqual(r['status'], 'flagged')

    def test_median_depth_and_refusal_by_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene, n, e1, e2 = plane_scene(tmp, n_support=0)
            for k in range(5):
                scene.add_point(np.array([0.01 * np.cos(k), 0.01 * np.sin(k), 0.0]))
            model = scene.write()
            r = cp.solve_all(model, clicks_doc(scene.click('p', np.zeros(3), [2])), P)['p']
            self.assertEqual(r['diagnostics']['submethod'], 'median-depth')
            self.assertEqual(r['diagnostics']['support_count'], 5)
            r = cp.solve_all(model, clicks_doc(scene.click('p', np.zeros(3), [2])), dict(P, min_support=6, min_support_plane=8))['p']
            self.assertEqual(r['status'], 'refused')
            self.assertIn('min_support', r['reason'])
            far = scene.click('q', np.array([0.29, 0.29, 0]), [2])
            r = cp.solve_all(model, clicks_doc(far), P)['q']
            self.assertEqual((r['status'], r['diagnostics']['support_count']), ('refused', 0))

    def test_support_uses_only_points_observed_in_the_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene, n, e1, e2 = plane_scene(tmp, n_support=0)
            rng = np.random.default_rng(9)
            for _ in range(200):
                scene.add_point(np.append(rng.uniform(-0.05, 0.05, 2), 0), images={1, 3})  # not in image 2
            model = scene.write()
            r = cp.solve_all(model, clicks_doc(scene.click('p', np.zeros(3), [2])), P)['p']
            self.assertEqual(r['status'], 'refused')
            r = cp.solve_all(model, clicks_doc(scene.click('p', np.zeros(3), [3])), P)['p']
            self.assertNotEqual(r['status'], 'refused')


def tape_scene(tmp, suspended=None):
    """A straight tape on a tilted plane. COLMAP units = 2.5 x meters, dataparser scale 0.37,
    so meters per dataparser unit = 1 / (2.5 x 0.37)."""
    scene, n, e1, e2 = plane_scene(tmp, seed=7)
    k = 2.5
    origin = -0.25 * e1 + 0.02 * e2
    marks = {N: origin + e1 * (N / 100.0 * k) for N in (0, 2, 5, 8, 10, 13, 15, 18, 20)}
    if suspended:
        marks[suspended] = marks[suspended] + 0.015 * n * k  # 1.5 cm above the surface
    return scene, marks, 1 / (k * 0.37)


def run_cli(main, argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class RulerScale(unittest.TestCase):
    def write_clicks(self, tmp, pts):
        p = Path(tmp) / 'clicks.json'
        p.write_text(json.dumps(clicks_doc(pts)))
        return p

    def args(self, scene, clicks, out, *extra):
        return ['--clicks', str(clicks), '--colmap', str(scene.dir), '--dataparser', str(scene.dp_path),
                '--capture-id', 'synthetic', '--output', str(out), *extra]

    def test_weighted_median(self):
        self.assertEqual(rs.weighted_median([1, 2, 3], [1, 1, 1]), 2)
        self.assertEqual(rs.weighted_median([1, 2, 3], [1, 1, 10]), 3)
        self.assertEqual(rs.weighted_median([1, 2], [1, 1]), 1.5)

    def test_tape_labels(self):
        self.assertEqual(rs.tape_marks(['tape_5cm', 'tape_12.5cm', 'leaf_tip', 'tape_x']), {'tape_5cm': 5, 'tape_12.5cm': 12.5})
        with self.assertRaises(ValueError):
            rs.tape_marks(['tape_5cm', 'tape_5.0cm'])

    def test_scale_recovery_multi_and_single_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene, marks, truth = tape_scene(tmp)
            scene.write()
            rng = np.random.default_rng(11)
            pts = []
            for N, X in marks.items():
                views = [1, 4, 6] if N % 2 == 0 else [2]  # even marks multi-view, odd single-view
                pts += scene.click('tape_%gcm' % N, X, views, noise=0.5, rng=rng)
            clicks = self.write_clicks(tmp, pts)
            out = Path(tmp) / 'scale.json'
            code, so, se = run_cli(rs.main, self.args(scene, clicks, out))
            self.assertEqual(code, 0, se)
            s = json.loads(out.read_text())
            for key in ('schema', 'schema_version', 'capture_id', 'source_frame', 'meters_per_scene_unit',
                        'std_meters_per_scene_unit', 'n_pairs', 'method', 'pairs', 'points', 'inputs', 'decisions'):
                self.assertIn(key, s)
            self.assertEqual((s['schema'], s['schema_version'], s['source_frame']), ('scale-estimate', 1, 'dataparser'))
            self.assertEqual(s['n_pairs'], 36)
            self.assertLess(abs(s['meters_per_scene_unit'] / truth - 1), 0.005)
            self.assertGreater(s['std_meters_per_scene_unit'], 0)
            self.assertEqual({p['method'] for p in s['points']}, {'multi-view', 'single-view-local-surface'})
            self.assertEqual(set(s['pairs'][0]), {'a', 'b', 'known_m', 'scene_distance', 'meters_per_scene_unit',
                                                 'scene_distance_sigma', 'propagated_std'})
            self.assertLess(s['collinearity']['max_perpendicular_deviation_mm'], 1.0)
            self.assertEqual(s['decisions'][0]['by'], 'Alex')
            # refuses to overwrite
            code, _, se = run_cli(rs.main, self.args(scene, clicks, out))
            self.assertEqual(code, 2)

    def test_suspended_mark_shows_in_collinearity_and_can_be_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene, marks, truth = tape_scene(tmp, suspended=10)
            scene.write()
            pts = sum((scene.click('tape_%gcm' % N, X, [2]) for N, X in marks.items()), [])
            clicks = self.write_clicks(tmp, pts)
            out = Path(tmp) / 'scale.json'
            self.assertEqual(run_cli(rs.main, self.args(scene, clicks, out))[0], 0)
            s = json.loads(out.read_text())
            worst = max(s['collinearity']['per_mark'], key=lambda m: math.hypot(m['leave_one_out_spacing_residual_mm'],
                                                                                 m['leave_one_out_perpendicular_mm']))
            self.assertEqual(worst['label'], 'tape_10cm')
            self.assertLess(abs(s['meters_per_scene_unit'] / truth - 1), 0.01)  # weighted median resists one bad mark
            out2 = Path(tmp) / 'scale2.json'
            self.assertEqual(run_cli(rs.main, self.args(scene, clicks, out2, '--exclude', 'tape_10cm'))[0], 0)
            s2 = json.loads(out2.read_text())
            self.assertEqual((s2['n_pairs'], s2['excluded_labels']), (28, ['tape_10cm']))
            self.assertLess(abs(s2['meters_per_scene_unit'] / truth - 1), 0.002)

    def test_surface_offset_iterates_to_the_raised_tape(self):
        """A tape resting 1 cm above the support surface, all marks single-view: without the offset
        the marks land on the surface (biased scale); with --surface-offset-m 0.01 the iteration
        converts meters with the current estimate and recovers the true scale."""
        with tempfile.TemporaryDirectory() as tmp:
            scene, n, e1, e2 = plane_scene(tmp, seed=7)
            k = 2.5
            truth = 1 / (k * 0.37)
            marks = {N: -0.25 * e1 + 0.02 * e2 + e1 * (N / 100.0 * k) + 0.01 * k * n for N in (0, 5, 10, 15, 20)}
            scene.write()
            pts = sum((scene.click('tape_%gcm' % N, X, [2]) for N, X in marks.items()), [])
            clicks = self.write_clicks(tmp, pts)
            out0, out1 = Path(tmp) / 's0.json', Path(tmp) / 's1.json'
            self.assertEqual(run_cli(rs.main, self.args(scene, clicks, out0))[0], 0)
            self.assertEqual(run_cli(rs.main, self.args(scene, clicks, Path(tmp) / 'x.json', '--surface-offset-m', '0.01'))[0], 2)
            code, so, se = run_cli(rs.main, self.args(scene, clicks, out1, '--surface-offset-m', '0.01',
                                                      '--surface-offset-basis', 'test'))
            self.assertEqual(code, 0, se)
            s0, s1 = json.loads(out0.read_text()), json.loads(out1.read_text())
            self.assertGreater(abs(s0['meters_per_scene_unit'] / truth - 1), 0.003)
            self.assertLess(abs(s1['meters_per_scene_unit'] / truth - 1), 1e-6)
            self.assertTrue(s1['surface_offset']['converged'])
            self.assertGreater(len(s1['surface_offset']['iterations']), 2)
            self.assertAlmostEqual(s1['surface_offset']['final_offset_colmap'], 0.01 * k, places=6)
            m = s1['mark_summary'][0]
            self.assertAlmostEqual(m['surface_offset_mm'], 10.0, places=3)
            self.assertIn('support_signed_distance_from_plane_mm', m)

    def test_refusal_with_fewer_than_two_marks(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene, marks, truth = tape_scene(tmp)
            scene.write()
            # tape_8cm is clicked in an image corner where no support point lies: refused, leaving one mark
            pts = scene.click('tape_5cm', marks[5], [2]) + [{'image': 'frame_00002.png', 'label': 'tape_8cm', 'x': 5.0, 'y': 5.0}]
            clicks = self.write_clicks(tmp, pts)
            out = Path(tmp) / 'scale.json'
            code, _, se = run_cli(rs.main, self.args(scene, clicks, out))
            self.assertEqual(code, 3, se)
            self.assertFalse(out.exists())
            rec = json.loads((Path(tmp) / 'scale.refused.json').read_text())
            self.assertEqual(rec['status'], 'refused')
            self.assertEqual([p['label'] for p in rec['refused_points']], ['tape_8cm'])
            code, _, se = run_cli(rs.main, self.args(scene, clicks, Path(tmp) / 'x.json', '--exclude', 'tape_99cm'))
            self.assertEqual(code, 2)


class TriangulateCLI(unittest.TestCase):
    def test_points_chords_scale_and_leaf_association(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            scene, marks, truth = tape_scene(tmp)
            model = scene.write()
            pts = sum((scene.click('tape_%gcm' % N, X, [1, 4, 6]) for N, X in marks.items()), [])
            (tmp / 'tape.json').write_text(json.dumps(clicks_doc(pts)))
            args = ['--colmap', str(scene.dir), '--dataparser', str(scene.dp_path), '--capture-id', 'synthetic']
            self.assertEqual(run_cli(rs.main, ['--clicks', str(tmp / 'tape.json'), '--output', str(tmp / 'scale.json')] + args)[0], 0)
            # two synthetic leaves in the dataparser frame; clicks on the first one's ends
            rng = np.random.default_rng(5)
            a, b = np.array([0.0, 0.0, 0.05]), np.array([0.08, 0.03, 0.12])
            c, d = np.array([-0.1, 0.05, 0.05]), np.array([-0.15, 0.1, 0.1])
            asset = tmp / 'plant-asset'
            (asset / 'leaves').mkdir(parents=True)
            leaves = []
            for i, (p, q) in enumerate([(a, b), (c, d)]):
                t = rng.uniform(0, 1, 300)[:, None]
                Xc = p + t * (q - p) + rng.normal(0, 0.003, (300, 3))
                Xd = cp.colmap_to_dataparser(Xc, scene.dp)
                np.savez(asset / 'leaves' / ('leaf-%d.npz' % i), X=Xd)
                leaves.append({'id': 'syn:leaf:%d' % i, 'role': 'true-leaf', 'point_geometry': {
                    'path': 'leaves/leaf-%d.npz' % i, 'sha256': cp.sha256(asset / 'leaves' / ('leaf-%d.npz' % i))}})
            (asset / 'plant.json').write_text(json.dumps({'schema': 'plant-asset', 'leaves': leaves}))
            pts = scene.click('L1_base', a, [1, 3, 5]) + scene.click('L1_tip', b, [2, 5, 7]) + scene.click('L2_base', c, [2, 4])
            (tmp / 'leaf.json').write_text(json.dumps(clicks_doc(pts)))
            out = tmp / 'points.json'
            code, so, se = run_cli(cp.main, ['triangulate', '--clicks', str(tmp / 'leaf.json'), '--output', str(out),
                                             '--plant-asset', str(asset), '--scale', str(tmp / 'scale.json')] + args)
            self.assertEqual(code, 0, se)
            r = json.loads(out.read_text())
            self.assertEqual(r['schema'], 'image-click-points')
            ch = r['leaf_chords']['L1']
            chord_true = np.linalg.norm(cp.colmap_to_dataparser(b[None], scene.dp) - cp.colmap_to_dataparser(a[None], scene.dp))
            self.assertAlmostEqual(ch['chord_scene'], chord_true, places=7)
            self.assertAlmostEqual(ch['chord_m'], np.linalg.norm(b - a) / 2.5, delta=1e-4)
            self.assertEqual(ch['association']['nearest'], 'syn:leaf:0')
            self.assertGreater(ch['association']['margin_to_second'], 0.01)
            self.assertEqual(r['leaf_chords']['L2']['status'], 'incomplete')
            # tampered leaf file is refused
            np.savez(asset / 'leaves' / 'leaf-1.npz', X=np.zeros((3, 3)))
            code, _, se = run_cli(cp.main, ['triangulate', '--clicks', str(tmp / 'leaf.json'), '--output', str(tmp / 'p2.json'),
                                            '--plant-asset', str(asset)] + args)
            self.assertEqual(code, 2)
            self.assertIn('hash', se)
            plant = json.loads((asset / 'plant.json').read_text())
            plant['leaves'][1]['point_geometry']['sha256'] = cp.sha256(asset / 'leaves' / 'leaf-1.npz')
            (asset / 'plant.json').write_text(json.dumps(plant))
            (asset / 'source-evidence').mkdir()
            (asset / 'source-evidence' / 'organ-evidence-original.json').write_text(
                json.dumps({'camera_inputs': {'dataparser_transforms': '0' * 64}}))
            code, _, se = run_cli(cp.main, ['triangulate', '--clicks', str(tmp / 'leaf.json'), '--output', str(tmp / 'p3.json'),
                                            '--plant-asset', str(asset)] + args)
            self.assertEqual(code, 2)
            self.assertIn('different dataparser', se)

    def test_validate_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'c.json'
            p.write_text(json.dumps(clicks_doc([{'image': 'frame_00001.png', 'label': 'a_tip', 'x': 1, 'y': 2}])))
            code, so, _ = run_cli(cp.main, ['validate', str(p)])
            self.assertEqual(code, 0)
            self.assertIn('1 single-view', so)


REAL_COLMAP = Path('/home/alex/data/colmap/welzo-M1/welzo-m1_261008-01-3120')
REAL_DP = Path('/home/alex/data/outputs/welzo-M1/welzo-m1_261008-01-3120/splatfacto/2026-10-08_215724/dataparser_transforms.json')


@unittest.skipUnless(REAL_COLMAP.exists() and REAL_DP.exists(), 'private Welzo M1 data not on this machine')
class RealCameras(unittest.TestCase):
    """COLMAP's own keypoints as fake clicks: triangulation must return COLMAP's points."""

    @classmethod
    def setUpClass(cls):
        cls.model = cp.Model(REAL_COLMAP, REAL_DP)
        cls.iid = cls.model.by_name['frame_00024.png']

    def test_chain_and_names(self):
        self.assertEqual(self.model.cameras[self.iid].name, 'IMG_20261002_103539587.png')
        self.assertTrue(self.model.checks['frame_map']['agrees_with_sorted_colmap_names'])
        self.assertLess(self.model.checks['frame_chain']['max_center_difference_dataparser'], 1e-9)

    def test_multi_view_matches_colmap(self):
        errs = []
        for pid, (xyz, iids, idxs) in self.model.points.items():
            if self.iid not in iids or len(set(iids.tolist())) < 3:
                continue
            obs, seen = [], set()
            for i, j in zip(iids.tolist(), idxs.tolist()):
                if i not in seen:
                    seen.add(i)
                    x, y = self.model.images[i]['xys'][j]
                    obs.append((self.model.frame_name[i], float(x), float(y)))
            errs.append(np.linalg.norm(cp.solve_label(self.model, obs, P)['xyz_dataparser']
                                       - self.model.to_dataparser(xyz)[0]))
            if len(errs) == 200:
                break
        self.assertEqual(len(errs), 200)
        self.assertLess(np.median(errs), 1e-8)

    def test_single_view_held_out(self):
        ids, kp, X = self.model.support_table(self.iid)
        cam = self.model.cameras[self.iid]
        rel = []
        for pid, uv, xyz in list(zip(ids, kp, X))[::5]:
            r = cp.local_surface(self.model, cam, uv, P, exclude_point_ids=[pid])
            if r['status'] != 'refused':
                rel.append(np.linalg.norm(r['xyz_colmap'] - xyz) / (cam.R @ xyz + cam.t)[2])
        self.assertGreater(len(rel), 150)
        self.assertLess(np.median(rel), 0.005)


if __name__ == '__main__':
    unittest.main()
