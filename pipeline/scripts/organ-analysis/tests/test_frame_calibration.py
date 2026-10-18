import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import frame_calibration as fc
import calibrate_frame as cli
from analyze_organs import sha

H = 'a' * 64


def rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    R = np.eye(3)
    R[i, i] = R[j, j] = c
    R[i, j], R[j, i] = -s, s
    return R


def calibrated(origin=(0.1, -0.2, 0.3)):
    cal = fc.new('cap-1', H, {'export_ply_sha256': 'b' * 64})
    R = rot(0, 20) @ rot(2, 35)
    fc.set_rotation(cal, R, origin, 'test rotation', 'algorithmic', [{'path': '/x/o.json', 'sha256': H}], {'k': 1})
    return cal


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = cli.main(argv)
        except ValueError as exc:
            return 2, out.getvalue(), str(exc)
    return code, out.getvalue(), err.getvalue()


class Library(unittest.TestCase):
    def test_new_is_identity_and_valid(self):
        cal = fc.new('cap-1', H)
        self.assertEqual(fc.validate(cal), [])
        np.testing.assert_array_equal(fc.compose(cal), np.eye(4))
        self.assertEqual(cal['target_frame']['units'], 'scale_free_scene_units')

    def test_round_trip_and_deterministic_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / 'a.json', Path(d) / 'b.json'
            fc.save(calibrated(), a)
            fc.save(calibrated(), b)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            text = a.read_text()
            self.assertTrue(text.endswith('}\n'))
            self.assertEqual(text, json.dumps(json.loads(text), sort_keys=True, indent=2) + '\n')
            loaded = fc.load(a)
            self.assertEqual(loaded, json.loads(fc.dumps(calibrated())))
            c = Path(d) / 'c.json'
            fc.save(loaded, c)
            self.assertEqual(a.read_bytes(), c.read_bytes())
            with self.assertRaises(FileExistsError):
                fc.save(calibrated(), a)

    def test_compose_and_apply_agree_with_formula(self):
        cal = calibrated()
        R = np.array(cal['rotation']['matrix'])
        o = np.array(cal['origin']['point_source_frame'])
        P = np.random.default_rng(0).normal(size=(50, 3))
        want = (P - o) @ R.T
        np.testing.assert_allclose(fc.apply(cal, P), want, atol=1e-14)
        T = fc.compose(cal)
        np.testing.assert_allclose(P @ T[:3, :3].T + T[:3, 3], want, atol=1e-14)
        np.testing.assert_allclose(fc.apply(cal, o[None]), [[0, 0, 0]], atol=1e-15)
        with self.assertRaises(ValueError):
            fc.apply(cal, np.zeros((3, 2)))

    def test_validation_failures(self):
        good = json.loads(fc.dumps(calibrated()))
        self.assertEqual(fc.validate(good), [])

        def errs(mutate):
            c = copy.deepcopy(good)
            mutate(c)
            return ' | '.join(fc.validate(c))

        def non_ortho(c):
            c['rotation']['matrix'][0][0] += 1e-6
        self.assertIn('not orthonormal', errs(non_ortho))

        def reflect(c):
            c['rotation']['matrix'] = [[1, 0, 0], [0, 1, 0], [0, 0, -1]]
        self.assertIn('determinant', errs(reflect))
        self.assertIn('target_frame.units', errs(lambda c: c['target_frame'].update(units='m')))
        self.assertIn('rotation.inputs[0].sha256', errs(lambda c: c['rotation']['inputs'][0].update(sha256='ABC')))
        self.assertIn('binding', errs(lambda c: c['source_frame']['binding'].update(dataparser_transforms_sha256='0' * 63)))
        self.assertIn('rotation.inputs must be a nonempty list', errs(lambda c: c['rotation'].update(inputs=[])))
        self.assertIn('rotation.method', errs(lambda c: c['rotation'].update(method='')))
        self.assertIn('origin.method', errs(lambda c: c['origin'].update(method=' ')))
        self.assertIn('rotation.status', errs(lambda c: c['rotation'].update(status='agent')))

        def tamper(c):
            c['transform_4x4'][0][3] += 1e-9
        self.assertIn('transform_4x4 differs', errs(tamper))
        self.assertIn('unsupported schema_version', errs(lambda c: c.update(schema_version=2)))
        self.assertIn('origin.point_source_frame', errs(lambda c: c['origin'].update(point_source_frame=[0, 0, float('nan')])))

        c = copy.deepcopy(good)
        c['scale'] = {'meters_per_scene_unit': -1.0, 'uncertainty': {'std_meters_per_scene_unit': 0.0, 'n_pairs': 1, 'method': 'm'},
                      'method': 'm', 'decisions': [], 'inputs': [{'path': 'p', 'sha256': H}], 'diagnostics': {}}
        c['target_frame']['units'] = 'm'
        self.assertIn('meters_per_scene_unit', ' | '.join(fc.validate(c)))
        c['scale']['meters_per_scene_unit'] = 2.0
        c['transform_4x4'] = fc.compose(c).tolist()
        self.assertEqual(fc.validate(c), [])
        c['scale']['decisions'] = [{'path': 'd', 'sha256': H, 'by': 'agent'}]
        self.assertIn('by must be "Alex"', ' | '.join(fc.validate(c)))

    def test_load_refuses_unknown_version_and_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'a.json'
            fc.save(calibrated(), p)
            c = json.loads(p.read_text())
            c['schema_version'] = 99
            q = Path(d) / 'q.json'
            q.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError, 'schema_version'):
                fc.load(q)
            c['schema_version'] = 1
            c['transform_4x4'][2][2] = 0.5
            r = Path(d) / 'r.json'
            r.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError, 'transform_4x4'):
                fc.load(r)

    def test_setters_refuse_and_keep_previous_state(self):
        cal = calibrated()
        before = fc.dumps(cal)
        bad = np.eye(3) * 2
        with self.assertRaises(ValueError):
            fc.set_rotation(cal, bad, None, 'm', 'algorithmic', [{'path': 'p', 'sha256': H}], {})
        self.assertEqual(fc.dumps(cal), before)
        with self.assertRaises(ValueError):
            fc.set_scale(cal, 0.0, 0.0, 1, 'u', 'm', [], [{'path': 'p', 'sha256': H}], {})
        self.assertEqual(fc.dumps(cal), before)

    def test_horizontal_x(self):
        up = np.array([0.1, -0.35, 0.93])
        R, d = fc.horizontal_x(up)
        np.testing.assert_allclose(R @ (up / np.linalg.norm(up)), [0, 0, 1], atol=1e-15)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-15)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=14)
        x_img = R @ np.array([1.0, 0, 0])
        self.assertAlmostEqual(x_img[1], 0.0, places=15)
        self.assertGreater(x_img[0], 0)
        R0, _ = fc.horizontal_x([0, 0, 1])
        np.testing.assert_allclose(R0, np.eye(3), atol=1e-15)
        with self.assertRaisesRegex(ValueError, 'Degenerate'):
            fc.horizontal_x([1, 0, 0.001])
        with self.assertRaisesRegex(ValueError, 'Degenerate'):
            fc.horizontal_x([-1, 0, 0])


def synthetic_orientation(dp_path, dp_sha, soil=True, normal=(0.09222772587860252, -0.34876833138183805, 0.9326600117965761)):
    n = np.asarray(normal) / np.linalg.norm(normal)
    point = np.array([-0.05, 0.2, -0.57])
    base = point + np.array([0.1, 0.05, 0.0])
    base = base - ((base - point) @ n) * n + 0.037 * n
    cam = {'method': 'camera_mean_image_up', 'status': 'ok', 'normal': [0.0, 0.0, 1.0], 'angular_rms_deg': 55.0}
    up_soil = None
    if soil:
        up_soil = {'method': 'soil_plane', 'status': 'ok', 'normal': [float(v) for v in n], 'point': point.tolist(),
                   'base': base.tolist(), 'inliers': 243, 'inlier_rms': 0.002, 'converged': True,
                   'base_offset_above_plane': 0.037, 'base_plant': 'M1'}
    angle = float(np.degrees(np.arccos(n[2])))
    choice = ({'method': 'soil_plane', 'status': 'primary', 'up': up_soil['normal'], 'angle_b_c_deg': angle,
               'up_disagreement': angle > 10, 'disagreement_threshold_deg': 10.0} if soil else
              {'method': 'camera_mean_image_up', 'status': 'fallback', 'up': cam['normal'], 'angle_b_c_deg': None,
               'up_disagreement': False, 'disagreement_threshold_deg': 10.0})
    return {'schema': 'plant-orientation', 'schema_version': 1, 'version': 'test', 'capture_id': 'cap-1',
            'units': 'scale_free_scene_units', 'status': 'complete', 'up_camera_mean_image_up': cam,
            'up_soil_plane': up_soil, 'up': choice,
            'plants': {'M1': {'axis': {'status': 'ok', 'polyline': [base.tolist(), (base + 0.1 * n).tolist()]}}},
            'input_sha256': {str(dp_path): dp_sha}}


class Orient(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.dp = self.d / 'dataparser_transforms.json'
        self.dp.write_text(json.dumps({'transform': np.eye(4)[:3].tolist(), 'scale': 0.1}))
        self.dp_sha = sha(self.dp)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, o, name='orientation.json'):
        p = self.d / name
        p.write_text(json.dumps(o))
        return p

    def test_orient_soil(self):
        o = synthetic_orientation(self.dp, self.dp_sha)
        p = self.write(o)
        ply = self.d / 'export.ply'
        ply.write_bytes(b'ply\n')
        out = self.d / 'cal.json'
        code, text, err = run(['orient', '--orientation', str(p), '--dataparser-transforms', str(self.dp),
                               '--capture-id', 'cap-1', '--bind-file', 'export_ply=' + str(ply), '--up', 'z', '--out', str(out)])
        self.assertEqual(code, 0, err)
        cal = fc.load(out)
        self.assertEqual(cal['target_frame']['up'], '+z')
        self.assertIsNone(cal['scale'])
        self.assertEqual(cal['target_frame']['units'], 'scale_free_scene_units')
        self.assertEqual(cal['source_frame']['binding'], {'dataparser_transforms_sha256': self.dp_sha,
                                                          'export_ply_sha256': sha(ply)})
        n = np.array(o['up_soil_plane']['normal'])
        R = np.array(cal['rotation']['matrix'])
        np.testing.assert_allclose(R @ n, [0, 0, 1], atol=1e-14)
        self.assertAlmostEqual((R @ [1, 0, 0])[1], 0.0, places=14)
        base, point = np.array(o['up_soil_plane']['base']), np.array(o['up_soil_plane']['point'])
        org = np.array(cal['origin']['point_source_frame'])
        self.assertAlmostEqual(float((org - point) @ n), 0.0, places=14)
        np.testing.assert_allclose(fc.apply(cal, base[None])[0], [0, 0, 0.037], atol=1e-12)
        np.testing.assert_allclose(fc.apply(cal, point[None])[0, 2], 0.0, atol=1e-14)
        d = cal['rotation']['diagnostics']
        self.assertAlmostEqual(d['camera_vs_soil_angle_deg'], 21.1466668, places=5)
        self.assertTrue(d['orientation_up_disagreement'])
        self.assertEqual(d['up_source'], 'soil_plane')
        self.assertEqual(cal['rotation']['status'], 'algorithmic')
        self.assertEqual([i['sha256'] for i in cal['rotation']['inputs']], [sha(p), self.dp_sha])
        self.assertEqual([i['sha256'] for i in cal['origin']['inputs']], [sha(p)])
        self.assertIn('up_soil_plane.base', cal['origin']['method'])
        # repeat is byte-identical
        out2 = self.d / 'cal2.json'
        run(['orient', '--orientation', str(p), '--dataparser-transforms', str(self.dp), '--capture-id', 'cap-1',
             '--bind-file', 'export_ply=' + str(ply), '--up', 'z', '--out', str(out2)])
        self.assertEqual(out.read_bytes(), out2.read_bytes())
        self.assertEqual(run(['validate', str(out)])[0], 0)
        # existing output refused
        self.assertEqual(run(['orient', '--orientation', str(p), '--dataparser-transforms', str(self.dp),
                              '--capture-id', 'cap-1', '--out', str(out)])[0], 2)

    def _asset(self, axis0, dp_sha=None):
        a = self.d / 'asset'
        (a / 'source-evidence').mkdir(parents=True, exist_ok=True)
        ev = a / 'source-evidence' / 'organ-evidence-original.json'
        ev.write_text(json.dumps({'capture_id': 'cap-1', 'source_ply_sha256': 'c' * 64,
                                  'camera_inputs': {'dataparser_transforms': dp_sha or self.dp_sha}}))
        plant = a / 'plant.json'
        plant.write_text(json.dumps({'plant_id': 'M1', 'capture_id': 'cap-1', 'units': 'scale_free_scene_units',
                                     'main_axis': {'id': 'M1:axis:main', 'polyline': [list(axis0), [0, 0, 1.0]]},
                                     'source_evidence': [{'path': 'source-evidence/organ-evidence-original.json',
                                                          'sha256': sha(ev)}]}))
        return plant

    def test_explicit_origin_from_asset_and_point(self):
        o = synthetic_orientation(self.dp, self.dp_sha)
        p = self.write(o)
        n = np.array(o['up_soil_plane']['normal'])
        plane_pt = np.array(o['up_soil_plane']['point'])
        root = plane_pt + np.array([0.06, -0.07, 0.0])
        root = root - ((root - plane_pt) @ n) * n + 0.033 * n   # 0.033 above the plane, off the default base
        plant = self._asset(root)
        base_args = ['orient', '--orientation', str(p), '--dataparser-transforms', str(self.dp), '--capture-id', 'cap-1']
        decision = ['--origin-decided-by', 'Alex', '--origin-decision', 'Alex 2026-10-18: bottom of the base stem']
        out = self.d / 'cal-asset.json'
        code, text, err = run(base_args + ['--origin-from-asset', str(plant)] + decision + ['--out', str(out)])
        self.assertEqual(code, 0, err)
        cal = fc.load(out)
        org = np.array(cal['origin']['point_source_frame'])
        want = root - 0.033 * n
        np.testing.assert_allclose(org, want, atol=1e-14)
        self.assertAlmostEqual(float((org - plane_pt) @ n), 0.0, places=14)
        d = cal['origin']['diagnostics']
        self.assertAlmostEqual(d['projection_distance_along_up'], 0.033, places=14)
        np.testing.assert_allclose(d['unprojected_point_source_frame'], root, atol=1e-15)
        self.assertEqual(d['decision'], {'by': 'Alex', 'statement': 'Alex 2026-10-18: bottom of the base stem'})
        self.assertEqual(d['asset']['frame_check']['evidence_dataparser_transforms_sha256'], self.dp_sha)
        self.assertEqual([i['sha256'] for i in cal['origin']['inputs']], [sha(p), sha(plant)])
        self.assertIn('explicit human decision by Alex', cal['origin']['method'])
        self.assertIn('bottom of the base stem', cal['target_frame']['origin_definition'])
        # Same rotation as the default; only the origin differs, and the shift is recorded.
        default = self.d / 'cal-default.json'
        self.assertEqual(run(base_args + ['--out', str(default)])[0], 0)
        dcal = fc.load(default)
        self.assertEqual(dcal['rotation']['matrix'], cal['rotation']['matrix'])
        dorg = np.array(dcal['origin']['point_source_frame'])
        np.testing.assert_allclose(d['shift_from_default_origin_source_frame'], org - dorg, atol=1e-15)
        self.assertNotIn('decision', dcal['origin']['diagnostics'])
        # Differences between points are unchanged under the new origin (pure translation in the plant frame).
        pts = np.random.default_rng(0).normal(size=(20, 3))
        A, B = fc.apply(cal, pts), fc.apply(dcal, pts)
        np.testing.assert_allclose(A - A[0], B - B[0], atol=1e-14)
        self.assertEqual(run(['validate', str(out)])[0], 0)
        # --origin-point gives the same origin.
        out2 = self.d / 'cal-point.json'
        code, _, err = run(base_args + ['--origin-point', ','.join(repr(float(v)) for v in root)] + decision + ['--out', str(out2)])
        self.assertEqual(code, 0, err)
        np.testing.assert_allclose(fc.load(out2)['origin']['point_source_frame'], want, atol=1e-14)

    def test_explicit_origin_refusals(self):
        o = synthetic_orientation(self.dp, self.dp_sha)
        p = self.write(o)
        base_args = ['orient', '--orientation', str(p), '--dataparser-transforms', str(self.dp), '--capture-id', 'cap-1']
        decision = ['--origin-decided-by', 'Alex', '--origin-decision', 'Alex 2026-10-18: bottom of the base stem']
        k = iter(range(100))

        def code(extra):
            out = self.d / ('r%d.json' % next(k))
            c, _, err = run(base_args + extra + ['--out', str(out)])
            self.assertFalse(out.exists())
            return c, err
        self.assertEqual(code(['--origin-point', '0,0,0'])[0], 2)                 # no decision
        self.assertEqual(code(decision)[0], 2)                                     # decision without a point
        self.assertEqual(code(['--origin-point', '0,0'] + decision)[0], 2)         # malformed
        plant = self._asset((0, 0, 0))
        self.assertEqual(code(['--origin-point', '0,0,0', '--origin-from-asset', str(plant)] + decision)[0], 2)
        other = self._asset((0, 0, 0), dp_sha='d' * 64)                            # different frame
        c, err = code(['--origin-from-asset', str(other)] + decision)
        self.assertEqual(c, 2)
        self.assertIn('different frame', err)
        ev = self.d / 'asset' / 'source-evidence' / 'organ-evidence-original.json'
        ev.write_text('{"tampered": true}')                                        # evidence hash mismatch
        self.assertEqual(code(['--origin-from-asset', str(other)] + decision)[0], 2)
        nosoil = synthetic_orientation(self.dp, self.dp_sha, soil=False)
        p2 = self.write(nosoil, 'nosoil.json')
        out = self.d / 'ns.json'
        c, _, err = run(['orient', '--orientation', str(p2), '--dataparser-transforms', str(self.dp), '--capture-id',
                         'cap-1', '--origin-point', '0,0,0'] + decision + ['--out', str(out)])
        self.assertEqual(c, 2)
        self.assertIn('soil plane', err)

    def test_orient_y_up_default(self):
        o = synthetic_orientation(self.dp, self.dp_sha)
        p = self.write(o)
        out = self.d / 'cal-y.json'
        code, _, err = run(['orient', '--orientation', str(p), '--dataparser-transforms', str(self.dp),
                            '--capture-id', 'cap-1', '--out', str(out)])
        self.assertEqual(code, 0, err)
        cal = fc.load(out)
        self.assertEqual(cal['target_frame']['up'], '+y')
        R = np.array(cal['rotation']['matrix'])
        n = np.array(o['up_soil_plane']['normal'])
        np.testing.assert_allclose(R @ n, [0, 1, 0], atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=14)
        zc = cli.calibration_from_orientation(o, p, self.dp, 'cap-1', up_axis='+z')
        Rz = np.array(zc['rotation']['matrix'])
        np.testing.assert_allclose(R, np.array(fc.Z_UP_TO_Y_UP) @ Rz, atol=1e-15)
        np.testing.assert_allclose(R[0], Rz[0], atol=1e-15)  # plant +x kept
        base = np.array(o['up_soil_plane']['base'])
        np.testing.assert_allclose(fc.apply(cal, base[None])[0], [0, 0.037, 0], atol=1e-12)
        self.assertEqual(cal['rotation']['up_source_frame'], cal['rotation']['diagnostics']['up_source_frame'])
        # validator: up setting must match R
        bad = json.loads(out.read_text())
        bad['target_frame']['up'] = '+z'
        self.assertIn('does not map the recorded up vector', ' | '.join(fc.validate(bad)))
        bad['target_frame']['up'] = '-y'
        self.assertIn('target_frame.up must be one of', ' | '.join(fc.validate(bad)))

    def test_old_z_up_files_stay_valid(self):
        o = synthetic_orientation(self.dp, self.dp_sha)
        zc = cli.calibration_from_orientation(o, self.write(o), self.dp, 'cap-1', up_axis='+z')
        old = json.loads(fc.dumps(zc))
        del old['rotation']['up_source_frame']  # files written before the up setting existed
        self.assertEqual(fc.validate(old), [])
        del old['target_frame']['up']  # a missing up setting reads as +z
        self.assertEqual(fc.validate(old), [])
        old['rotation']['diagnostics']['up_source_frame'] = [0.0, 1.0, 0.0]
        self.assertIn('does not map', ' | '.join(fc.validate(old)))

    def test_orient_fallback_recorded(self):
        o = synthetic_orientation(self.dp, self.dp_sha, soil=False)
        cal = cli.calibration_from_orientation(o, self.write(o), self.dp, 'cap-1', up_axis='+z')
        d = cal['rotation']['diagnostics']
        self.assertTrue(d['up_source'].startswith('camera_mean_image_up (fallback'))
        self.assertNotIn('camera_vs_soil_angle_deg', d)
        np.testing.assert_allclose(cal['rotation']['matrix'], np.eye(3), atol=1e-15)
        self.assertEqual(cal['origin']['point_source_frame'], o['plants']['M1']['axis']['polyline'][0])
        self.assertIn('polyline[0]', cal['origin']['method'])
        self.assertEqual(fc.validate(cal), [])

    def test_orient_refusals(self):
        o = synthetic_orientation(self.dp, self.dp_sha)
        p = self.write(o)
        with self.assertRaisesRegex(ValueError, 'capture'):
            cli.calibration_from_orientation(o, p, self.dp, 'other')
        other = self.d / 'other' / 'dataparser_transforms.json'
        other.parent.mkdir()
        other.write_text('{"different": true}')
        with self.assertRaisesRegex(ValueError, 'matches none'):
            cli.calibration_from_orientation(o, p, other, 'cap-1')
        x_up = synthetic_orientation(self.dp, self.dp_sha, normal=(1.0, 0.0, 0.0001))
        with self.assertRaisesRegex(ValueError, 'Degenerate'):
            cli.calibration_from_orientation(x_up, self.write(x_up, 'x.json'), self.dp, 'cap-1')
        o2 = dict(o, status='refused')
        with self.assertRaisesRegex(ValueError, 'complete'):
            cli.calibration_from_orientation(o2, self.write(o2, 'r.json'), self.dp, 'cap-1')
        o3 = synthetic_orientation(self.dp, self.dp_sha, soil=False)
        o3['up_camera_mean_image_up']['status'] = 'refused'
        with self.assertRaisesRegex(ValueError, 'no up'):
            cli.calibration_from_orientation(o3, self.write(o3, 'n.json'), self.dp, 'cap-1')


class Scale(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.cal_path = self.d / 'cal.json'
        fc.save(calibrated(), self.cal_path)
        dec = self.d / 'pairs-decision.json'
        dec.write_text('{"pairs": "chosen by Alex"}')
        self.estimate = {'schema': 'scale-estimate', 'schema_version': 1, 'capture_id': 'cap-1',
                         'source_frame': 'dataparser', 'meters_per_scene_unit': 0.25,
                         'std_meters_per_scene_unit': 0.004, 'n_pairs': 3, 'method': 'tape pairs, median ratio',
                         'pairs': [{}, {}, {}], 'points': [{}, {}, {}, {}],
                         'inputs': [{'path': '/data/clicks.json', 'sha256': 'c' * 64}],
                         'decisions': [{'path': str(dec), 'sha256': sha(dec), 'by': 'Alex'}]}

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, e, name='estimate.json'):
        p = self.d / name
        p.write_text(json.dumps(e))
        return p

    def test_scale_ingestion(self):
        e = self.write(self.estimate)
        out = self.d / 'scaled.json'
        code, text, err = run(['scale', '--calibration', str(self.cal_path), '--estimate', str(e), '--out', str(out)])
        self.assertEqual(code, 0, err)
        cal = fc.load(out)
        self.assertEqual(cal['target_frame']['units'], 'm')
        sc = cal['scale']
        self.assertEqual(sc['meters_per_scene_unit'], 0.25)
        self.assertEqual(sc['uncertainty'], {'std_meters_per_scene_unit': 0.004, 'n_pairs': 3,
                                             'method': sc['uncertainty']['method']})
        self.assertEqual(sc['method'], 'tape pairs, median ratio')
        self.assertEqual(sc['inputs'][-1], {'path': str(e.resolve()), 'sha256': sha(e)})
        self.assertEqual(sc['inputs'][0]['sha256'], 'c' * 64)
        self.assertEqual(sc['decisions'], self.estimate['decisions'])
        base = fc.load(self.cal_path)
        P = np.random.default_rng(1).normal(size=(10, 3))
        np.testing.assert_allclose(fc.apply(cal, P), 0.25 * fc.apply(base, P), atol=1e-15)
        T = fc.compose(cal)
        np.testing.assert_allclose(np.linalg.det(T[:3, :3]), 0.25 ** 3, atol=1e-15)

    def test_scale_refusals(self):
        cal = fc.load(self.cal_path)
        for change, msg in [({'capture_id': 'cap-2'}, 'capture_id'), ({'source_frame': 'colmap'}, 'source_frame'),
                            ({'schema_version': 2}, 'schema_version'), ({'meters_per_scene_unit': 0}, 'meters_per_scene_unit'),
                            ({'inputs': [{'path': '/r/dataparser_transforms.json', 'sha256': 'd' * 64}]}, 'bound to dataparser'),
                            ({'decisions': [{'path': 'x', 'sha256': H, 'by': 'agent'}]}, 'Alex')]:
            e = dict(self.estimate, **change)
            with self.assertRaisesRegex(ValueError, msg):
                fc.set_scale_from_estimate(copy.deepcopy(cal), e, self.write(e, 'bad.json'))
            (self.d / 'bad.json').unlink()
        bad = self.write(dict(self.estimate, capture_id='cap-2'))
        self.assertEqual(run(['scale', '--calibration', str(self.cal_path), '--estimate', str(bad),
                              '--out', str(self.d / 'never.json')])[0], 2)
        self.assertFalse((self.d / 'never.json').exists())


class Apply(unittest.TestCase):
    def test_apply_npy_and_npz(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cp = d / 'cal.json'
            fc.save(calibrated(), cp)
            cal = fc.load(cp)
            P = np.random.default_rng(2).normal(size=(20, 3)).astype(np.float32)
            np.save(d / 'p.npy', P)
            np.savez(d / 'leaf.npz', X=P, label=np.arange(20))
            self.assertEqual(run(['apply', '--calibration', str(cp), '--input', str(d / 'p.npy'), '--out', str(d / 'q.npy')])[0], 0)
            np.testing.assert_allclose(np.load(d / 'q.npy'), fc.apply(cal, P), atol=1e-15)
            self.assertEqual(run(['apply', '--calibration', str(cp), '--input', str(d / 'leaf.npz'), '--out', str(d / 'out.npz')])[0], 0)
            with np.load(d / 'out.npz') as z:
                np.testing.assert_allclose(z['X'], fc.apply(cal, P), atol=1e-15)
                np.testing.assert_array_equal(z['label'], np.arange(20))
            with np.load(d / 'leaf.npz') as z:
                np.testing.assert_array_equal(z['X'], P)
            self.assertEqual(run(['apply', '--calibration', str(cp), '--input', str(d / 'p.npy'), '--out', str(d / 'p.npy')])[0], 2)
            self.assertEqual(run(['apply', '--calibration', str(cp), '--input', str(d / 'p.npy'), '--out', str(d / 'q.npy')])[0], 2)
            self.assertEqual(run(['apply', '--calibration', str(cp), '--input', str(d / 'leaf.npz'), '--array', 'Y',
                                  '--out', str(d / 'y.npz')])[0], 2)

    def test_validate_command_reports_errors(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'bad.json'
            c = json.loads(fc.dumps(calibrated()))
            c['transform_4x4'][0][0] = 3.0
            p.write_text(json.dumps(c))
            code, text, _ = run(['validate', str(p)])
            self.assertEqual(code, 1)
            self.assertIn('transform_4x4', text)


if __name__ == '__main__':
    unittest.main()
