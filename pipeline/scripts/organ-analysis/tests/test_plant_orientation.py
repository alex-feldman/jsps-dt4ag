import contextlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plant_orientation as po
import orient_plants as cli
from test_plant_clusters import write_ply

P = dict(po.DEFAULTS)


def rotation_to(n):
    """Rotation taking +z to unit n."""
    n = po.unit(n)
    z = np.array([0, 0, 1.0])
    v, c = np.cross(z, n), float(z @ n)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def ring_frames(normal, count=12, radius=1.0, height=0.3, wobble=0.0, seed=0):
    """OpenGL camera-to-world matrices on a ring in the plane of `normal`, looking at the origin,
    with image up along the normal's side of the view."""
    R = rotation_to(normal)
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(count):
        a = 2 * np.pi * i / count
        p = R @ np.array([radius * np.cos(a), radius * np.sin(a), height + wobble * rng.standard_normal()])
        f = po.unit(-p)
        right = po.unit(np.cross(f, R @ [0, 0, 1.0]))
        up = np.cross(right, f)
        M = np.eye(4)
        M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = right, up, -f, p
        frames.append({'file_path': 'images/frame_%05d.png' % (i + 1), 'transform_matrix': M.tolist(), 'colmap_im_id': i + 1})
    return frames


def cylinder(center_xy, z0, z1, radius, n, seed):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 2 * np.pi, n)
    r = radius * np.sqrt(rng.uniform(0, 1, n))
    return np.stack([center_xy[0] + r * np.cos(a), center_xy[1] + r * np.sin(a), rng.uniform(z0, z1, n)], 1)


def write_colmap(directory, points, images):
    """points: list of (id, xyz, [(image_id, idx)]); images: {id: (name, xys)}."""
    with open(directory / 'points3D.bin', 'wb') as f:
        f.write(struct.pack('<Q', len(points)))
        for pid, xyz, track in points:
            f.write(struct.pack('<QdddBBBd', pid, *xyz, 0, 0, 0, 0.5))
            f.write(struct.pack('<Q', len(track)))
            for iid, j in track:
                f.write(struct.pack('<ii', iid, j))
    with open(directory / 'images.bin', 'wb') as f:
        f.write(struct.pack('<Q', len(images)))
        for iid, (name, xys) in images.items():
            f.write(struct.pack('<i4d3di', iid, 1, 0, 0, 0, 0, 0, 0, 1))
            f.write(name.encode() + b'\0')
            f.write(struct.pack('<Q', len(xys)))
            for x, y in xys:
                f.write(struct.pack('<ddq', x, y, -1))


class CameraMeanUp(unittest.TestCase):
    def test_mean_image_up_of_a_tilted_ring_and_its_spread(self):
        n = po.unit([0.3, -0.2, 0.9])
        frames = ring_frames(n, wobble=0.01)
        with tempfile.TemporaryDirectory() as tmp:
            t, d = Path(tmp) / 't.json', Path(tmp) / 'd.json'
            t.write_text(json.dumps({'frames': frames}))
            d.write_text(json.dumps({'transform': np.eye(4)[:3].tolist(), 'scale': 2.0}))
            stems, C, U = po.camera_poses(t, d)
        rec = po.camera_mean_up(U)
        self.assertEqual(rec['status'], 'ok')
        self.assertEqual(stems[0], 'frame_00001')
        # The cameras look down at the center, so each image-up leans inward; the mean is the normal.
        self.assertLess(po.angle_deg(rec['normal'], n), 1.0)
        lean = np.degrees(np.arctan(0.3))
        self.assertAlmostEqual(rec['angular_rms_deg'], lean, delta=1.0)
        self.assertLess(po.angle_deg(po.camera_mean_up(-U)['normal'], -n), 1.0)

    def test_layout_never_refuses_only_degenerate_sets(self):
        rng = np.random.default_rng(3)
        U = np.tile([0, 0, 1.0], (40, 1)) + 0.2 * rng.standard_normal((40, 3))
        rec = po.camera_mean_up(U)  # any camera layout: centers are not used at all
        self.assertEqual(rec['status'], 'ok')
        self.assertGreater(rec['angular_rms_deg'], 5)
        self.assertEqual(po.camera_mean_up(np.zeros((0, 3)))['status'], 'refused')
        self.assertEqual(po.camera_mean_up(np.array([[0, 0, 1.0], [0, 0, -1.0]]))['status'], 'refused')


class SoilPlane(unittest.TestCase):
    def plane_points(self, n, count=400, outliers=100, seed=5):
        rng = np.random.default_rng(seed)
        R = rotation_to(n)
        Q = np.stack([rng.uniform(-.2, .2, count), rng.uniform(-.2, .2, count), rng.normal(0, .001, count)], 1) @ R.T
        O = rng.uniform(-.2, .2, (outliers, 3))
        return np.vstack([Q, O])

    def test_recovers_tilted_plane_and_signs_toward_plant(self):
        n = po.unit([0.14, -0.36, 0.92])
        X = self.plane_points(n)
        hits, seen = np.full(len(X), 3), np.full(len(X), 4)
        rec, soil, local = po.soil_plane_up(X, hits, seen, np.zeros(3), -0.5 * n, P)
        self.assertEqual(rec['status'], 'ok')
        self.assertLess(po.angle_deg(rec['normal'], -n), 1.0)
        self.assertGreaterEqual(rec['inliers'], 380)
        rec2, _, _ = po.soil_plane_up(X, hits, seen, np.zeros(3), 0.5 * n, P)
        self.assertLess(po.angle_deg(rec2['normal'], n), 1.0)
        self.assertEqual(rec2['inliers'], rec['inliers'])

    def test_vote_rule_and_refusals(self):
        n = np.array([0, 0, 1.0])
        X = self.plane_points(n, count=200, outliers=0)
        hits = np.full(len(X), 1)
        rec, _, _ = po.soil_plane_up(X, hits, np.full(len(X), 1), np.zeros(3), n, P)
        self.assertEqual(rec['status'], 'refused')
        self.assertEqual(rec['soil_supported_points'], 0)
        rec, _, _ = po.soil_plane_up(X, np.full(len(X), 2), np.full(len(X), 4), np.zeros(3), n, P)
        self.assertEqual(rec['soil_supported_points'], 0)  # share 0.5 < 0.6
        rec, _, _ = po.soil_plane_up(X, np.full(len(X), 3), np.full(len(X), 4), np.array([5.0, 0, 0]), n, P)
        self.assertEqual(rec['status'], 'refused')  # nothing within soil_local_radius

    def test_colmap_readers_and_track_votes(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            points = [(7, (0.1, 0.2, 0.3), [(1, 0), (2, 0), (2, 1)]), (9, (1.0, 2.0, 3.0), [(1, 1), (3, 0)])]
            images = {1: ('frame_00001.png', [(1.5, 1.5), (8.2, 1.0)]), 2: ('frame_00002.png', [(2.0, 2.0), (9.0, 9.0)]),
                      3: ('frame_00003.png', [(50.0, 1.0)])}
            write_colmap(d, points, images)
            pts, ims = po.read_points3d_binary(d / 'points3D.bin'), po.read_images_binary(d / 'images.bin')
        self.assertEqual(sorted(pts), [7, 9])
        np.testing.assert_allclose(pts[9][0], [1, 2, 3])
        np.testing.assert_array_equal(pts[7][1], [1, 2, 2])
        self.assertEqual(ims[2][0], 'frame_00002.png')
        np.testing.assert_allclose(ims[1][1][1], [8.2, 1.0])
        m = np.zeros((10, 10), bool)
        m[:5, :5] = True
        ids, hits, seen = po.soil_votes(pts, ims, {1: 'A', 2: 'B', 3: 'C'}, {'A': m, 'B': m, 'C': m})
        np.testing.assert_array_equal(ids, [7, 9])
        # Point 7: A hit, B hit once (one vote per view). Point 9: A miss; C out of bounds is not a view.
        np.testing.assert_array_equal(hits, [2, 0])
        np.testing.assert_array_equal(seen, [2, 1])
        np.testing.assert_allclose(po.colmap_to_dataparser(np.array([[1.0, 0, 0]]),
                                   {'transform': [[0, -1, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]], 'scale': 2.0}), [[2, 2, 0]])


def tilted_scene(tilt_deg=25.0, seed=11):
    """A plant cylinder standing on a soil plane, both tilted tilt_deg from dataparser z."""
    n = po.unit([np.sin(np.radians(tilt_deg)), 0, np.cos(np.radians(tilt_deg))])
    R = rotation_to(n)
    rng = np.random.default_rng(seed)
    plant = cylinder((0, 0), 0.01, 0.4, 0.06, 3000, seed) @ R.T
    soil = np.stack([rng.uniform(-.6, .6, 3000), rng.uniform(-.6, .6, 3000), rng.normal(0, .0005, 3000)], 1) @ R.T
    return n, plant, soil


class SoilIteration(unittest.TestCase):
    def test_converges_to_the_soil_normal_from_a_tilted_bootstrap(self):
        n, plant, S = tilted_scene()
        hits, seen = np.full(len(S), 3), np.full(len(S), 3)
        rec = po.soil_plane_iterative(S, hits, seen, plant, [0, 0, 1.0], P)
        self.assertEqual(rec['status'], 'ok')
        self.assertTrue(rec['converged'])
        its = rec['iterations']
        self.assertGreaterEqual(len(its), 2)
        self.assertLessEqual(len(its), P['max_soil_iterations'])
        self.assertEqual(its[0]['base_along'], 'camera_mean_image_up')
        self.assertTrue(its[-1]['inlier_set_unchanged'])
        self.assertFalse(its[0]['inlier_set_unchanged'])
        self.assertLess(po.angle_deg(rec['normal'], n), 1.0)  # signed toward the plant
        self.assertAlmostEqual(its[0]['angle_to_bootstrap_deg'], 25.0, delta=1.0)
        # The result does not depend on the bootstrap direction once converged.
        rec2 = po.soil_plane_iterative(S, hits, seen, plant, po.unit([-0.2, 0.1, 1.0]), P)
        self.assertTrue(rec2['converged'])
        self.assertLess(po.angle_deg(rec2['normal'], rec['normal']), 0.5)

    def test_iteration_cap_and_refusal_are_recorded(self):
        n, plant, S = tilted_scene()
        hits, seen = np.full(len(S), 3), np.full(len(S), 3)
        rec = po.soil_plane_iterative(S, hits, seen, plant, [0, 0, 1.0], dict(P, max_soil_iterations=1))
        self.assertEqual(rec['status'], 'ok')
        self.assertFalse(rec['converged'])
        self.assertIn('not converged', rec['note'])
        rec = po.soil_plane_iterative(S, np.zeros(len(S), int), seen, plant, [0, 0, 1.0], P)
        self.assertEqual(rec['status'], 'refused')
        self.assertFalse(rec['converged'])
        self.assertEqual(len(rec['iterations']), 1)


class Choice(unittest.TestCase):
    def test_soil_primary_disagreement_reported_and_fallback(self):
        c_up = {'status': 'ok', 'normal': [0, 0, 1.0]}
        near = {'status': 'ok', 'normal': po.floats(po.unit([0, np.sin(np.radians(8)), np.cos(np.radians(8))]))}
        far = {'status': 'ok', 'normal': po.floats(po.unit([0, np.sin(np.radians(20)), np.cos(np.radians(20))]))}
        c = po.choose_up(c_up, near, P)
        self.assertEqual((c['method'], c['status']), ('soil_plane', 'primary'))
        self.assertFalse(c['up_disagreement'])
        self.assertAlmostEqual(c['angle_b_c_deg'], 8.0, places=6)
        c = po.choose_up(c_up, far, P)
        self.assertTrue(c['up_disagreement'])
        self.assertEqual(c['method'], 'soil_plane')  # reported, never switched
        self.assertEqual(c['up'], far['normal'])
        c = po.choose_up(c_up, None, P)
        self.assertEqual((c['method'], c['status']), ('camera_mean_image_up', 'fallback'))
        self.assertIn('no soil inputs', c['reason'])
        c = po.choose_up(c_up, {'status': 'refused', 'normal': None, 'reason': '12 inliers < min_inliers 30'}, P)
        self.assertEqual(c['status'], 'fallback')
        self.assertIn('min_inliers', c['reason'])
        c = po.choose_up({'status': 'refused', 'normal': None, 'reason': 'no cameras'}, far, P)
        self.assertEqual(c['method'], 'soil_plane')
        self.assertIsNone(c['angle_b_c_deg'])
        c = po.choose_up({'status': 'refused', 'normal': None, 'reason': 'no cameras'}, None, P)
        self.assertIsNone(c['up'])
        self.assertEqual(c['status'], 'refused')


class AxisAndAssignment(unittest.TestCase):
    def test_vertical_cylinder_is_straight(self):
        X = cylinder((0.2, -0.1), 0.0, 0.5, 0.05, 6000, 1)
        ax = po.plant_axis(X, [0, 0, 1.0], P)
        self.assertEqual(ax['status'], 'ok')
        self.assertEqual(len(ax['slices']), 25)
        self.assertLess(ax['straightness']['max_horizontal_deviation'], 0.01)
        self.assertLess(ax['straightness']['line_tilt_from_up_deg'], 3)
        for s in ax['slices']:
            np.testing.assert_allclose(s['axis_point'][:2], [0.2, -0.1], atol=0.01)
            self.assertAlmostEqual(s['radius'], 0.05 * np.sqrt(0.95), delta=0.006)
        self.assertAlmostEqual(ax['height_range'][0], X[:, 2].min())
        self.assertAlmostEqual(ax['height_range'][1], X[:, 2].min() + 25 * 0.02)

    def test_sparse_plant_has_no_axis(self):
        ax = po.plant_axis(cylinder((0, 0), 0, 0.5, 0.05, 50, 1), [0, 0, 1.0], P)
        self.assertEqual(ax['status'], 'insufficient')

    def test_detached_assignment(self):
        up = np.array([0, 0, 1.0])
        axes = {'A': po.plant_axis(cylinder((0, 0), 0, 0.6, 0.1, 6000, 1), up, P),
                'B': po.plant_axis(cylinder((0.5, 0), 0, 0.6, 0.1, 6000, 2), up, P)}
        inside = cylinder((0.03, 0.02), 0.30, 0.34, 0.01, 50, 3)
        rec = po.assign_detached(inside, axes, up, P)
        self.assertEqual(rec['assignment'], 'A')
        self.assertLessEqual(rec['plants']['A']['ratio'], 1.0)
        self.assertGreaterEqual(rec['plants']['B']['ratio'], 2.0)
        between = cylinder((0.25, 0), 0.30, 0.34, 0.01, 50, 4)
        rec = po.assign_detached(between, axes, up, P)
        self.assertEqual(rec['assignment'], 'unassigned')
        above = cylinder((0.0, 0), 0.9, 0.92, 0.01, 50, 5)
        rec = po.assign_detached(above, axes, up, P)
        self.assertEqual(rec['assignment'], 'unassigned')
        self.assertTrue(rec['plants']['A']['outside_slice_centers'])
        self.assertFalse(rec['plants']['A']['within_height_range_with_margin'])
        axes['C'] = {'status': 'insufficient'}
        self.assertEqual(po.assign_detached(inside, axes, up, P)['assignment'], 'unassigned')


class Command(unittest.TestCase):
    """The public command on a synthetic tilted scene with two plants, a detached blob and soil."""

    def scene(self, tmp, normal, soil=True, count=12, camera_up=None):
        tmp = Path(tmp)
        R = rotation_to(normal)
        frames = ring_frames(normal if camera_up is None else camera_up, count=count, radius=1.5)
        w, h = 40, 30
        (tmp / 't.json').write_text(json.dumps({'frames': frames, 'w': w, 'h': h}))
        (tmp / 'd.json').write_text(json.dumps({'transform': np.eye(4)[:3].tolist(), 'scale': 1.0}))
        A = cylinder((0, 0), 0.0, 0.4, 0.06, 3000, 1) @ R.T
        B = cylinder((0.45, 0), 0.0, 0.4, 0.06, 3000, 2) @ R.T
        blob = cylinder((0.01, 0.01), 0.2, 0.23, 0.01, 40, 3) @ R.T
        X = np.vstack([A, B, blob])
        write_ply(tmp / 'a.ply', A, np.full(len(A), 2.0))
        write_ply(tmp / 'b.ply', B, np.full(len(B), 2.0))
        write_ply(tmp / 'blob.ply', blob, np.full(len(blob), 2.0))
        cfg = {'capture_id': 'synthetic', 'inputs': {'transforms': 't.json', 'dataparser_transforms': 'd.json'},
               'plants': [{'name': 'A', 'ply': 'a.ply'}, {'name': 'B', 'ply': 'b.ply'}],
               'detached': [{'name': 'blob', 'ply': 'blob.ply'}]}
        if soil:
            sparse = tmp / 'sparse'
            sparse.mkdir()
            rng = np.random.default_rng(9)
            S = np.stack([rng.uniform(-.25, .25, 200), rng.uniform(-.25, .25, 200), np.full(200, -0.02)], 1) @ R.T
            pts = [(i + 1, tuple(S[i]), [(1, i), (2, i), (3, i)]) for i in range(len(S))]
            write_colmap(sparse, pts, {k: ('frame_%05d.png' % k, [(5.0, 5.0)] * len(S)) for k in (1, 2, 3)})
            for name in ['soil', 'plant']:
                (tmp / name).mkdir()
            for k in range(1, count + 1):
                Image.fromarray(np.zeros((h, w), np.uint8)).save(tmp / 'plant' / ('IMG_%d.png' % k))
                if k <= 3:
                    Image.fromarray(np.full((h, w), 255, np.uint8)).save(tmp / 'soil' / ('IMG_%d.png' % k))
            (tmp / 'fmap.json').write_text(json.dumps({'frame_%05d' % k: 'IMG_%d' % k for k in range(1, count + 1)}))
            cfg['soil'] = {'colmap_sparse': 'sparse', 'soil_masks': 'soil', 'plant_masks': 'plant',
                           'frame_map': 'fmap.json', 'base_plant': 'A'}
        (tmp / 'cfg.json').write_text(json.dumps(cfg))
        return tmp / 'cfg.json', cfg

    def run_cli(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return cli.main([str(a) for a in args])

    def test_end_to_end_soil_primary(self):
        n = po.unit([0.2, 0.1, 0.97])
        with tempfile.TemporaryDirectory() as tmp:
            cfg, _ = self.scene(tmp, n)
            out = Path(tmp) / 'out'
            self.assertEqual(self.run_cli('--config', cfg, '--output', out, '--validate-only'), 0)
            self.assertFalse(out.exists())
            self.assertEqual(self.run_cli('--config', cfg, '--output', out), 0)
            res = json.loads((out / 'orientation.json').read_text())
            rec = json.loads((out / 'run-record.json').read_text())
            self.assertEqual(rec['status'], 'complete')
            self.assertLess(po.angle_deg(res['up_camera_mean_image_up']['normal'], n), 0.5)
            b = res['up_soil_plane']
            self.assertEqual(b['status'], 'ok')
            self.assertTrue(b['converged'])
            self.assertLess(po.angle_deg(b['normal'], n), 0.5)
            self.assertEqual((res['up']['method'], res['up']['status']), ('soil_plane', 'primary'))
            self.assertFalse(res['up']['up_disagreement'])
            self.assertEqual(res['detached']['blob']['assignment'], 'A')
            self.assertEqual(res['cross_check_camera_mean_image_up']['detached']['blob']['assignment'], 'A')
            self.assertLess(res['plants']['A']['axis']['straightness']['max_horizontal_deviation'], 0.01)
            self.assertIn('orientation.json', rec['output_sha256'])
            with self.assertRaises(ValueError):
                self.run_cli('--config', cfg, '--output', out)

    def test_disagreement_is_reported_not_switched(self):
        n = po.unit([np.sin(np.radians(20)), 0, np.cos(np.radians(20))])
        with tempfile.TemporaryDirectory() as tmp:
            cfg, _ = self.scene(tmp, n, camera_up=[0, 0, 1.0])
            out = Path(tmp) / 'out'
            self.assertEqual(self.run_cli('--config', cfg, '--output', out), 0)
            res = json.loads((out / 'orientation.json').read_text())
            self.assertEqual(res['up']['method'], 'soil_plane')
            self.assertTrue(res['up']['up_disagreement'])
            self.assertAlmostEqual(res['up']['angle_b_c_deg'], 20.0, delta=1.0)
            self.assertLess(po.angle_deg(res['up']['up'], n), 0.5)
            self.assertEqual(res['detached']['blob']['assignment'], 'A')
            self.assertIsNotNone(res['cross_check_camera_mean_image_up'])

    def test_fallback_refusal_and_config_refusals(self):
        n = np.array([0, 0, 1.0])
        with tempfile.TemporaryDirectory() as tmp:
            cfg, base = self.scene(tmp, n, soil=False, count=6)
            out = Path(tmp) / 'out'
            self.assertEqual(self.run_cli('--config', cfg, '--output', out), 0)
            res = json.loads((out / 'orientation.json').read_text())
            self.assertEqual((res['up']['method'], res['up']['status']), ('camera_mean_image_up', 'fallback'))
            self.assertIsNone(res['cross_check_camera_mean_image_up'])
            self.assertEqual(res['detached']['blob']['assignment'], 'A')
            (Path(tmp) / 't.json').write_text(json.dumps({'frames': [], 'w': 40, 'h': 30}))
            out = Path(tmp) / 'out2'
            self.assertEqual(self.run_cli('--config', cfg, '--output', out), 3)
            res = json.loads((out / 'orientation.json').read_text())
            self.assertEqual(res['status'], 'refused')
            self.assertIsNone(res['up']['up'])
            self.assertEqual(res['detached'], {})
            for bad in [dict(base, extra=1),
                        dict(base, prefer_soil=True),
                        dict(base, plants=[{'name': 'A', 'clusters': [0]}]),
                        dict(base, plants=[{'name': 'A', 'ply': 'a.ply'}, {'name': 'A', 'ply': 'b.ply'}]),
                        dict(base, detached='auto'),
                        dict(base, parameters={'slice': -1}),
                        dict(base, parameters={'nope': 1}),
                        dict(base, soil={'colmap_sparse': '.', 'soil_masks': '.', 'plant_masks': '.', 'frame_map': 't.json'})]:
                (Path(tmp) / 'bad.json').write_text(json.dumps(bad))
                with self.assertRaises((ValueError, KeyError)):
                    self.run_cli('--config', Path(tmp) / 'bad.json', '--output', Path(tmp) / 'never', '--validate-only')
            self.assertFalse((Path(tmp) / 'never').exists())


if __name__ == '__main__':
    unittest.main()
