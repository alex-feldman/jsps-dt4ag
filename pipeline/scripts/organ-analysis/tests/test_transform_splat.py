import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import frame_calibration as fc
import transform_splat as ts
from analyze_organs import sha

H = 'a' * 64
NAMES = (['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2'] + ['f_rest_%d' % i for i in range(45)]
         + ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'])


def random_rotation(rng):
    q = rng.normal(size=4)
    return ts.quat_to_rotmat(q / np.linalg.norm(q))


def gaussians(n=200, dtype='<f8', seed=0):
    rng = np.random.default_rng(seed)
    a = np.zeros(n, dtype=[(k, dtype) for k in NAMES])
    for k in NAMES:
        a[k] = rng.normal(size=n)
    for k in ('scale_0', 'scale_1', 'scale_2'):
        a[k] = rng.uniform(-6, -2, size=n)
    return a


def colors(a, dirs):
    """SH color (without the 0.5 offset or clamp) of each Gaussian at its own direction, degree 3."""
    K = 15
    B = ts.sh_basis(dirs, 3)
    out = np.zeros((len(a), 3))
    for c in range(3):
        coef = np.concatenate([a['f_dc_%d' % c][:, None], np.stack([a['f_rest_%d' % (c * K + k)] for k in range(K)], 1)], 1)
        out[:, c] = (B * coef).sum(1)
    return out


def covariance(a):
    Rq = ts.quat_to_rotmat(np.stack([a['rot_%d' % i] for i in range(4)], 1))
    S = np.exp(np.stack([a['scale_%d' % i] for i in range(3)], 1))
    M = Rq * S[:, None, :]
    return M @ np.transpose(M, (0, 2, 1))


def calibration(R, o, s=None):
    cal = fc.new('cap', H, up='+y')
    up = R.T @ np.array([0, 1.0, 0])
    fc.set_rotation(cal, R, o, 'test', 'algorithmic', [{'path': 'p', 'sha256': H}], {}, up_source=up)
    if s is not None:
        fc.set_scale(cal, s, 0.0, 1, 'u', 'm', [], [{'path': 'p', 'sha256': H}], {})
    return cal


class Synthetic(unittest.TestCase):
    def test_sh_basis_matches_gsplat_constants_and_is_orthonormal(self):
        d = ts.sample_directions(200000, 1)
        B = ts.sh_basis(d, 3)
        G = B.T @ B * (4 * math.pi / len(d))
        self.assertLess(np.abs(G - np.eye(16)).max(), 0.02)
        np.testing.assert_allclose(ts.sh_basis(np.array([[0, 0, 1.0]]), 1)[0], [0.2820947917738781, 0, 0.48860251190292, 0])

    def test_sh_rotation_fit(self):
        rng = np.random.default_rng(3)
        for _ in range(5):
            mats, rec = ts.sh_rotation(random_rotation(rng), 3)
            for b in rec['bands']:
                self.assertLess(b['max_fit_residual'], 1e-12)
                self.assertLess(b['max_orthogonality_deviation'], 1e-12)
        mats, _ = ts.sh_rotation(np.eye(3), 3)
        for D in mats:
            np.testing.assert_allclose(D, np.eye(len(D)), atol=1e-13)

    def test_color_preserved_under_rotation(self):
        rng = np.random.default_rng(4)
        a = gaussians()
        for _ in range(3):
            R = random_rotation(rng)
            o, s = rng.normal(size=3), 0.37
            b, rec = ts.transform(a, R, o, s)
            for _ in range(20):
                d = ts.sample_directions(len(a), int(rng.integers(1 << 30)))
                err = np.abs(colors(b, d @ R.T) - colors(a, d)).max()
                self.assertLess(err, 1e-12)

    def test_covariance_centers_quaternions_scales(self):
        rng = np.random.default_rng(5)
        a = gaussians()
        R, o, s = random_rotation(rng), rng.normal(size=3), 0.5
        b, _ = ts.transform(a, R, o, s)
        want = s * s * (R @ covariance(a) @ R.T)
        got = covariance(b)
        self.assertLess(np.abs(got - want).max() / np.abs(want).max(), 1e-13)
        P = np.stack([a['x'], a['y'], a['z']], 1)
        np.testing.assert_allclose(np.stack([b['x'], b['y'], b['z']], 1), s * (P - o) @ R.T, atol=1e-13)
        qa = np.stack([a['rot_%d' % i] for i in range(4)], 1)
        qb = np.stack([b['rot_%d' % i] for i in range(4)], 1)
        np.testing.assert_allclose(np.linalg.norm(qb, axis=1), np.linalg.norm(qa, axis=1), rtol=1e-13)
        np.testing.assert_allclose(b['scale_0'], a['scale_0'] + math.log(s), atol=1e-14)
        for k in ('f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity'):
            np.testing.assert_array_equal(b[k], a[k])
        N = np.stack([a['nx'], a['ny'], a['nz']], 1)
        np.testing.assert_allclose(np.stack([b['nx'], b['ny'], b['nz']], 1), N @ R.T, atol=1e-13)

    def test_rotmat_to_quat(self):
        rng = np.random.default_rng(6)
        Rs = [random_rotation(rng) for _ in range(50)] + [np.diag([1, -1, -1.0]), np.diag([-1, 1, -1.0]),
                                                          np.diag([-1, -1, 1.0]), np.array(fc.Z_UP_TO_Y_UP)]
        for R in Rs:
            np.testing.assert_allclose(ts.quat_to_rotmat(ts.rotmat_to_quat(R)), R, atol=1e-14)

    def test_inverse_round_trip(self):
        rng = np.random.default_rng(7)
        a = gaussians()
        cal = calibration(random_rotation(rng), rng.normal(size=3), 0.4)
        b, _ = ts.transform(a, *ts.parameters(cal))
        c, _ = ts.transform(b, *ts.parameters(cal, inverse=True))
        for k in NAMES:
            if k.startswith('rot_'):
                continue
            np.testing.assert_allclose(c[k], a[k], atol=1e-12, err_msg=k)
        qa = np.stack([a['rot_%d' % i] for i in range(4)], 1)
        qc = np.stack([c['rot_%d' % i] for i in range(4)], 1)
        sign = np.sign((qa * qc).sum(1))[:, None]
        np.testing.assert_allclose(qc * sign, qa, atol=1e-12)


class Command(unittest.TestCase):
    def test_cli_new_file_sidecar_header(self):
        rng = np.random.default_rng(8)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            a = gaussians(50, '<f4')
            header = ['ply', 'format binary_little_endian 1.0', 'comment Generated by Nerstudio 1.1.5',
                      'comment Vertical Axis: z', 'element vertex 50'] + ['property float ' + k for k in NAMES] + ['end_header']
            src = d / 'in.ply'
            ts.write_ply(src, header, a)
            lines, back = ts.read_ply(src)
            self.assertEqual(lines, header)
            np.testing.assert_array_equal(back, a)
            cp = d / 'cal.json'
            fc.save(calibration(random_rotation(rng), rng.normal(size=3), 0.5), cp)
            out = d / 'out.ply'
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(ts.main(['--calibration', str(cp), '--input', str(src), '--out', str(out)]), 0)
            lines2, b = ts.read_ply(out)
            self.assertEqual([l for l in lines2 if l.startswith(('element', 'property'))],
                             [l for l in header if l.startswith(('element', 'property'))])
            self.assertIn('comment Vertical Axis: y', lines2)
            self.assertEqual(b.dtype, a.dtype)
            side = json.loads((d / 'out.transform.json').read_text())
            self.assertEqual(side['input']['sha256'], sha(src))
            self.assertEqual(side['output']['sha256'], sha(out))
            self.assertEqual(side['calibration']['sha256'], sha(cp))
            self.assertEqual(side['frame'], {'name': 'plant', 'up': '+y', 'units': 'm'})
            self.assertEqual(side['settings']['s'], 0.5)
            self.assertEqual(sorted(side['unchanged_properties']), ['f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity'])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ts.main(['--calibration', str(cp), '--input', str(src), '--out', str(d / 'ns.ply'),
                                          '--no-scale']), 0)
            ns = json.loads((d / 'ns.transform.json').read_text())
            self.assertEqual((ns['settings']['s'], ns['frame']['units']), (1.0, 'scale_free_scene_units'))
            with self.assertRaisesRegex(ValueError, 'exists'):
                ts.main(['--calibration', str(cp), '--input', str(src), '--out', str(out)])
            with self.assertRaisesRegex(ValueError, 'in place|exists'):
                ts.main(['--calibration', str(cp), '--input', str(src), '--out', str(src)])
            self.assertEqual(sha(src), side['input']['sha256'])

    def test_refuses_bad_layout(self):
        a = np.zeros(3, dtype=[(k, '<f4') for k in ['x', 'y', 'z', 'f_rest_0']])
        with self.assertRaises(ValueError):
            ts.layout(a)


if __name__ == '__main__':
    unittest.main()
