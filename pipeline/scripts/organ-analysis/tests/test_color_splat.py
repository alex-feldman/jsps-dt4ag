import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
import color_splat as cs
import frame_calibration as fc
import transform_splat as ts
from analyze_organs import sha
import test_plant_asset as asset_fixture


PROPERTIES = (['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2'] +
              ['f_rest_%d' % i for i in range(9)] +
              ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'])
HASH = 'a' * 64


def fixture_data():
    data = np.zeros(6, dtype=[(name, '<f4') for name in PROPERTIES])
    data['x'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    data['y'] = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    data['z'] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    data['f_dc_0'] = [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6]
    data['f_dc_1'] = [0.1, 0.3, 0.5, -0.5, -0.3, -0.1]
    data['f_dc_2'] = [0.6, 0.4, 0.2, 0.0, -0.2, -0.4]
    for index in range(9):
        data['f_rest_%d' % index] = (index + 1) * 0.01
    data['opacity'] = [2.0, 1.0, 0.5, 0.0, -2.0, 3.0]
    for name in ('scale_0', 'scale_1', 'scale_2'):
        data[name] = -2.0
    data['rot_0'] = 1.0
    return data


def data_for_points(points):
    points = np.asarray(points, np.float64)
    data = np.zeros(len(points), dtype=[(name, '<f8') for name in PROPERTIES])
    for axis, name in enumerate(('x', 'y', 'z')):
        data[name] = points[:, axis]
    data['opacity'] = 1.0
    for name in ('scale_0', 'scale_1', 'scale_2'):
        data[name] = -2.0
    data['rot_0'] = 1.0
    return data


def calibration():
    theta = 0.31
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    origin = np.array([0.25, -0.1, 0.4])
    result = fc.new('synthetic-capture', HASH, up='+y')
    fc.set_rotation(result, rotation, origin, 'synthetic test', 'algorithmic',
                    [{'path': 'fixture', 'sha256': HASH}], {}, up_source=rotation.T @ np.array([0.0, 1.0, 0.0]))
    fc.set_scale(result, 0.5, 0.0, 1, 'synthetic', 'm', [], [{'path': 'fixture', 'sha256': HASH}], {})
    return result


def palette_dc(color):
    rgb = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)], np.float64) / 255.0
    return (rgb - 0.5) / cs.C0


class ColorSplatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'synthetic-export.ply'
        self.data = fixture_data()
        self.header = ['ply', 'format binary_little_endian 1.0', 'comment Vertical Axis: z',
                       'element vertex 6'] + ['property float ' + name for name in PROPERTIES] + ['end_header']
        ts.write_ply(self.source, self.header, self.data)
        self.calibration_path = self.root / 'calibration.json'
        fc.save(calibration(), self.calibration_path)
        self.labels_dir = self.root / 'labels'
        self.labels_dir.mkdir()
        keep = cs.opacity_keep(self.data, 0.3)
        self.assertEqual(np.flatnonzero(keep).tolist(), [0, 1, 2, 3, 5])
        xyz = np.stack([self.data['x'], self.data['y'], self.data['z']], axis=1).astype(np.float64)
        labels = np.array([0, 0, 9, -1, -2], np.int64)
        self.labels_path = self.labels_dir / 'leaf_ids2.npz'
        np.savez(self.labels_path, label=labels, X=xyz[keep])
        self.evidence_path = self.labels_dir / 'organ-evidence.json'
        self.evidence = {
            'source_ply_sha256': sha(self.source),
            'point_labels': {'file': self.labels_path.name, 'sha256': sha(self.labels_path)},
            'leaves': [
                {'id': 0, 'association_support': 'supported'},
                {'id': 9, 'association_support': 'small_unreviewed_cluster'},
            ],
        }
        self.write_evidence()

    def write_evidence(self):
        self.evidence_path.write_text(json.dumps(self.evidence, indent=2) + '\n', encoding='utf-8')

    def invoke(self, output, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            return cs.main(['--input', str(self.source), '--calibration', str(self.calibration_path),
                            '--labels', str(self.labels_path), '--out-dir', str(output), *extra])

    def test_both_variants_sidecars_and_display_copy(self):
        out = self.root / 'colored'
        self.assertEqual(self.invoke(out, '--viewer', 'supersplat'), 0)
        self.assertTrue((out / 'legend.md').is_file())
        legend = json.loads((out / 'legend.json').read_text(encoding='utf-8'))
        self.assertEqual(legend['schema'], 'leaf-color-legend')
        self.assertEqual(legend['regions'], [{'region_id': 0, 'color': cs.PALETTE[0], 'filtered_gaussians': 2}])

        R, origin, scale = ts.parameters(fc.load(self.calibration_path))
        calibrated, _ = ts.transform(self.data, R, origin, scale)
        a_header, a = ts.read_ply(out / 'synthetic-export_leafcolor-A.ply')
        b_header, b = ts.read_ply(out / 'synthetic-export_leafcolor-B.ply')
        self.assertEqual([line for line in a_header if line.startswith(('format ', 'element ', 'property '))],
                         [line for line in self.header if line.startswith(('format ', 'element ', 'property '))])
        self.assertIn('comment Vertical Axis: y', a_header)
        expected = palette_dc(cs.PALETTE[0])
        np.testing.assert_allclose([a['f_dc_%d' % i][0] for i in range(3)], expected, rtol=0, atol=1e-7)
        np.testing.assert_allclose([a['f_dc_%d' % i][2] for i in range(3)],
                                   [calibrated['f_dc_%d' % i][2] for i in range(3)], rtol=0, atol=0)
        np.testing.assert_array_equal(a['f_rest_0'][2:], calibrated['f_rest_0'][2:])
        for index in range(9):
            self.assertTrue(np.all(b['f_rest_%d' % index] == 0))
        self.assertFalse(np.array_equal(a['f_dc_0'][2:], b['f_dc_0'][2:]))
        for name in PROPERTIES:
            if name.startswith(('f_dc_', 'f_rest_')):
                continue
            self.assertEqual(a[name].tobytes(), calibrated[name].tobytes(), name)
            self.assertEqual(b[name].tobytes(), calibrated[name].tobytes(), name)

        side = json.loads((out / 'synthetic-export_leafcolor-A.color.json').read_text(encoding='utf-8'))
        self.assertEqual(side['schema'], 'colored-splat')
        self.assertEqual(side['input']['sha256'], sha(self.source))
        self.assertEqual(side['calibration']['sha256'], sha(self.calibration_path))
        self.assertEqual(side['labels']['sha256'], sha(self.labels_path))
        self.assertEqual(side['labels']['source_ply_binding'], 'matched')
        self.assertEqual(side['palette']['source'], 'pipeline/scripts/organ-analysis/viewer.py:PALETTE')
        self.assertEqual(side['region_counts'], {'0': 2})
        self.assertEqual(side['unlabeled_breakdown'],
                         {'non_leaf': 1, 'unlabeled': 1, 'residual': 1, 'below_opacity_cutoff': 1})
        self.assertFalse(side['display_only'])
        self.assertTrue(side['self_checks']['positions_scales_rotations_opacity_bitwise_equal'])

        display_path = out / 'synthetic-export_leafcolor-B-supersplat.ply'
        display_side = out / 'synthetic-export_leafcolor-B-supersplat.color.json'
        d_header, display = ts.read_ply(display_path)
        display_record = json.loads(display_side.read_text(encoding='utf-8'))
        spec = ts.VIEWERS['supersplat']
        analysis_path = out / 'synthetic-export_leafcolor-B.ply'
        _, analysis = ts.read_ply(analysis_path)
        xyz = np.stack([analysis['x'], analysis['y'], analysis['z']], axis=1).astype(np.float64)
        want = (xyz @ np.asarray(spec['rotation'], np.float64).T).astype(display.dtype['x'])
        got = np.stack([display['x'], display['y'], display['z']], axis=1)
        np.testing.assert_array_equal(got, want)
        self.assertIn('comment DISPLAY ONLY', '\n'.join(d_header))
        self.assertTrue(display_record['display_only'])
        self.assertEqual(display_record['analysis_file']['sha256'], sha(analysis_path))
        self.assertTrue(display_record['self_checks']['positions_equal_viewer_rotation'])

    def test_include_residual_and_refuse_mismatched_provenance_or_rows(self):
        out = self.root / 'include-residual'
        self.invoke(out, '--variant', 'A', '--include-residual')
        side = json.loads((out / 'synthetic-export_leafcolor-A.color.json').read_text(encoding='utf-8'))
        self.assertEqual(side['region_counts'], {'0': 2, '9': 1})
        self.assertEqual(side['unlabeled_breakdown']['residual'], 1)

        self.evidence['source_ply_sha256'] = '0' * 64
        self.write_evidence()
        mismatched = self.root / 'mismatched-hash'
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            self.invoke(mismatched)
        self.assertFalse(mismatched.exists())

        self.evidence['source_ply_sha256'] = sha(self.source)
        self.evidence['point_labels']['sha256'] = sha(self.labels_path)
        self.write_evidence()
        arrays = np.load(self.labels_path, allow_pickle=False)
        shortened = {'label': arrays['label'][:-1], 'X': arrays['X'][:-1]}
        arrays.close()
        np.savez(self.labels_path, **shortened)
        self.evidence['point_labels']['sha256'] = sha(self.labels_path)
        self.write_evidence()
        short_output = self.root / 'short-labels'
        with self.assertRaisesRegex(ValueError, 'Label length'):
            self.invoke(short_output)
        self.assertFalse(short_output.exists())

    def test_existing_output_directory_is_refused(self):
        out = self.root / 'exists'
        out.mkdir()
        sentinel = out / 'keep.txt'
        sentinel.write_text('keep', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.invoke(out)
        self.assertEqual(sentinel.read_text(encoding='utf-8'), 'keep')

    def test_plant_asset_roster_and_sorted_leaf_hashes(self):
        fixture = asset_fixture.PlantAssetTests('test_package_emits_valid_relocatable_contract')
        fixture.setUp()
        try:
            with np.load(fixture.review / 'leaf_ids2.npz', allow_pickle=False) as arrays:
                points = arrays['X'].copy()
            data = data_for_points(points)
            source = fixture.root / 'source-export.ply'
            header = ['ply', 'format binary_little_endian 1.0', 'comment Vertical Axis: z',
                      'element vertex %d' % len(data)] + ['property double ' + name for name in PROPERTIES] + ['end_header']
            ts.write_ply(source, header, data)
            evidence_path = fixture.review / 'organ-evidence.json'
            evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
            evidence['source_ply_sha256'] = sha(source)
            evidence_path.write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
            asset_dir = fixture.package() / 'plant-asset'
            calibration_path = fixture.root / 'calibration.json'
            fc.save(calibration(), calibration_path)
            output = fixture.root / 'asset-colors'
            with contextlib.redirect_stdout(io.StringIO()):
                cs.main(['--input', str(source), '--calibration', str(calibration_path),
                         '--labels', str(asset_dir), '--out-dir', str(output), '--variant', 'A'])
            side = json.loads((output / 'source-export_leafcolor-A.color.json').read_text(encoding='utf-8'))
            leaf_hashes = sorted(sha(path) for path in (asset_dir / 'leaves').glob('*.npz'))
            expected = hashlib.sha256(''.join(leaf_hashes).encode('ascii')).hexdigest()
            self.assertEqual(side['labels']['sha256'], expected)
            self.assertEqual(side['labels']['selection_basis'], 'plant.json leaves roster')
            self.assertEqual(side['labels']['source_ply_binding'], 'matched')
            self.assertEqual(side['region_counts'], {'0': 10, '1': 10, '2': 10, '3': 10})
        finally:
            fixture.doCleanups()


if __name__ == '__main__':
    unittest.main()
