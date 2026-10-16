"""Exercise computed-axis integration through the real organ-analysis CLI."""
import json
import unittest

import numpy as np

import test_analysis as fixtures
from test_analysis import cli


class ComputedAxisTests(unittest.TestCase):
    setUp = fixtures.OrganAnalysisTests.setUp
    save = fixtures.OrganAnalysisTests.save
    invoke = fixtures.OrganAnalysisTests.invoke
    def fixture(self, conflict=False, tilted=False, near_tie=False):
        leaves = self.root / 'labels'
        self.invoke(leaves, '--stage', 'leaves')
        stem = np.column_stack([np.zeros(41), np.zeros(41), np.linspace(0, .2, 41)])
        blade = np.column_stack([np.linspace(.002, .004, 20), np.zeros(20), np.full(20, .2)])
        X = np.concatenate([stem, blade])
        labels = [-2] * len(stem) + [0] * len(blade)
        rgb = [[.1, .8, .1]] * len(stem) + [[.4, .4, .4]] * len(blade)
        if conflict or near_tie:
            if near_tie:
                branch = np.column_stack([np.linspace(.0015, .03, 20), np.zeros(20), np.linspace(.104, .18, 20)])
                blade2 = np.column_stack([np.full(20, .032), np.linspace(0, .002, 20), np.full(20, .196)])
            else:
                branch = np.column_stack([np.linspace(.005, .22, 44), np.zeros(44), np.full(44, .08)])
                blade2 = np.column_stack([np.full(20, .222), np.linspace(0, .002, 20), np.full(20, .08)])
            X = np.concatenate([X, branch, blade2])
            labels += [-2] * len(branch) + [1] * len(blade2)
            rgb += [[.1, .8, .1]] * len(branch) + [[.4, .4, .4]] * len(blade2)
        up = [0., 0., 1.]
        if tilted:
            X = X[:, [2, 1, 0]] * [1, 1, -1]
            up = [1., 0., 0.]
        np.savez(leaves / 'leaf_ids2.npz', X=X, label=np.array(labels), rgb=np.array(rgb))
        associations = cli.read(leaves / 'leaf_ids2.json')
        associations['clusters'] = [{'leaf_id': i} for i in range(2 if conflict or near_tie else 1)]
        (leaves / 'leaf_ids2.json').write_text(json.dumps(associations))
        record = cli.read(leaves / 'run-record.json')
        for name in ['leaf_ids2.npz', 'leaf_ids2.json']:
            record['output_sha256'][name] = cli.sha(leaves / name)
        (leaves / 'run-record.json').write_text(json.dumps(record))
        directory = self.root / 'orientation'
        directory.mkdir()
        inputs = {k: str(self.root / self.spec['inputs'][k]) for k in ['transforms', 'dataparser_transforms']}
        hashes = {v: cli.sha(v) for v in inputs.values()}
        hashes[str(self.root / 'frame_map.json')] = cli.sha(self.root / 'frame_map.json')
        artifact = {'schema': 'plant-orientation', 'schema_version': 1, 'status': 'complete',
                    'capture_id': self.spec['capture_id'], 'input_sha256': hashes,
                    'up': {'up': up, 'method': 'soil_plane', 'status': 'primary'},
                    'up_soil_plane': {'status': 'ok', 'normal': up, 'point': [0, 0, 0]}}
        self.artifact = directory / 'orientation.json'
        self.artifact.write_text(json.dumps(artifact))
        (directory / 'run-record.json').write_text(json.dumps({'status': 'complete',
            'spec': {'capture_id': self.spec['capture_id'], 'inputs': inputs,
                     'soil': {'frame_map': str(self.root / 'frame_map.json')}},
            'input_sha256': hashes, 'output_sha256': {'orientation.json': cli.sha(self.artifact)}}))
        self.spec.update(orientation=str(self.artifact), apex_mode='morphology')
        self.save()
        return leaves

    def run_skeleton(self, leaves, name='computed'):
        out = self.root / name
        self.invoke(out, '--stage', 'skeleton', '--labels-dir', str(leaves))
        return out, cli.read(out / 'skeleton.json')

    def test_consensus_and_tilted_root_use_orientation(self):
        leaves = self.fixture(tilted=True)
        out, graph = self.run_skeleton(leaves)
        report = graph['computed_axis']
        self.assertTrue(report['consensus'])
        self.assertEqual(graph['apex_leaf'], 0)
        self.assertEqual(report['base']['point_label_array_row'], 0)
        self.assertEqual(report['base']['signed_plane_distance'], 0)
        self.assertEqual(report['supported_leaves'][0]['median_height_rank'], 1)
        self.assertGreater(graph['main_stem'][-1][0], graph['main_stem'][0][0])
        self.assertIn(str(self.artifact), cli.read(out / 'run-record.json')['input_sha256'])
        self.assertNotIn('apex_leaf', cli.read(out / 'run-record.json')['spec'])

    def test_conflicting_candidates_refuse_and_override_records_decision(self):
        leaves = self.fixture(conflict=True)
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            self.run_skeleton(leaves)
        out = self.root / 'computed'
        report = cli.read(out / 'apex-candidates.json')
        self.assertFalse(report['consensus'])
        self.assertIsNone(report['apex_leaf'])
        self.assertFalse((out / 'skeleton.json').exists())
        record = cli.read(out / 'run-record.json')
        self.assertEqual(record['status'], 'failed')
        self.assertTrue(record['source_hashes_verified_on_failure'])
        self.spec['apex_leaf'] = 0
        self.save()
        _, graph = self.run_skeleton(leaves, 'override')
        self.assertEqual(graph['apex_leaf'], 0)
        self.assertFalse(graph['computed_axis']['consensus'])
        self.assertEqual(graph['computed_axis']['decision_basis'], 'explicit_apex_leaf_override')

    def change_artifact(self, **changes):
        value = cli.read(self.artifact)
        value.update(changes)
        self.artifact.write_text(json.dumps(value))
        path = self.artifact.parent / 'run-record.json'
        record = cli.read(path)
        record['output_sha256']['orientation.json'] = cli.sha(self.artifact)
        path.write_text(json.dumps(record))

    def test_orientation_foreign_malformed_and_tampering_fail_before_output(self):
        leaves = self.fixture()
        original = self.artifact.read_text()
        for change, message in [({'capture_id': 'foreign'}, 'capture_id'),
                                ({'up': {'up': [0, 0, 2]}}, 'unit vector'),
                                ({'status': 'refused'}, 'completed')]:
            self.change_artifact(**change)
            with self.assertRaisesRegex(ValueError, message):
                self.run_skeleton(leaves)
            self.assertFalse((self.root / 'computed').exists())
            self.artifact.write_text(original)
        # Tampering without regenerating the completed record's output digest.
        self.artifact.write_text(original + ' ')
        with self.assertRaisesRegex(ValueError, 'changed since completion'):
            self.run_skeleton(leaves)
        self.assertFalse((self.root / 'computed').exists())

    def test_orientation_source_change_rejected_before_output(self):
        leaves = self.fixture()
        # A non-analysis orientation source is still checked.
        source = self.root / 'soil-source.txt'
        source.write_text('original')
        artifact = cli.read(self.artifact)
        artifact['input_sha256'][str(source)] = cli.sha(source)
        self.change_artifact(input_sha256=artifact['input_sha256'])
        path = self.artifact.parent / 'run-record.json'
        record = cli.read(path)
        record['input_sha256'] = artifact['input_sha256']
        path.write_text(json.dumps(record))
        source.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Orientation source changed'):
            self.run_skeleton(leaves)
        self.assertFalse((self.root / 'computed').exists())

    def test_camera_fallback_is_explicit_and_plane_distance_unknown(self):
        leaves = self.fixture()
        self.change_artifact(up={'up': [0, 0, 1], 'method': 'camera_mean_image_up', 'status': 'fallback'},
                             up_soil_plane=None)
        with self.assertRaisesRegex(ValueError, 'soil-primary'):
            self.run_skeleton(leaves)
        self.assertFalse((self.root / 'computed').exists())
        self.spec['allow_camera_fallback'] = True
        self.save()
        _, graph = self.run_skeleton(leaves)
        self.assertIsNone(graph['computed_axis']['base']['signed_plane_distance'])
        self.assertEqual(graph['computed_axis']['orientation']['status'], 'fallback')

    def test_protected_orientation_and_frame_mismatch_rejected(self):
        leaves = self.fixture()
        self.spec['orientation'] = str(self.root / 'images' / 'orientation.json')
        self.save()
        with self.assertRaisesRegex(ValueError, 'protected source'):
            self.run_skeleton(leaves)
        self.spec['orientation'] = str(self.artifact)
        self.save()
        foreign = self.root / 'other-transforms.json'
        foreign.write_text('{}')
        path = self.artifact.parent / 'run-record.json'
        record = cli.read(path)
        record['spec']['inputs']['transforms'] = str(foreign)
        record['input_sha256'][str(foreign)] = cli.sha(foreign)
        self.change_artifact(input_sha256=record['input_sha256'])
        record['output_sha256']['orientation.json'] = cli.sha(self.artifact)
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'coordinate-frame hash'):
            self.run_skeleton(leaves)
        self.assertFalse((self.root / 'computed').exists())

    def test_supported_leaf_height_tie_refuses_without_numeric_id_choice(self):
        leaves = self.fixture(conflict=True)
        with np.load(leaves / 'leaf_ids2.npz') as data:
            arrays = {key: data[key] for key in data.files}
        arrays['X'][arrays['label'] == 1, 2] = .2
        np.savez(leaves / 'leaf_ids2.npz', **arrays)
        path = leaves / 'run-record.json'
        record = cli.read(path)
        record['output_sha256']['leaf_ids2.npz'] = cli.sha(leaves / 'leaf_ids2.npz')
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            self.run_skeleton(leaves)
        report = cli.read(self.root / 'computed' / 'apex-candidates.json')
        self.assertEqual([r['median_height_rank'] for r in report['supported_leaves']], [1, 1])
        self.assertEqual(report['decision_basis'], 'ambiguous_refused')

    def test_sub_bin_height_difference_refuses_despite_other_unique_extrema(self):
        leaves = self.fixture(near_tie=True)
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            self.run_skeleton(leaves)
        out = self.root / 'computed'
        report = cli.read(out / 'apex-candidates.json')
        tolerance = report['tolerance']['absolute']
        self.assertEqual(tolerance, cli.DEFAULTS['BIN'])
        heights = [row['median_height'] for row in report['supported_leaves']]
        difference = abs(heights[0] - heights[1])
        self.assertGreater(difference, tolerance / 10)
        self.assertLess(difference, tolerance)
        self.assertEqual(report['extrema']['highest_median_leaves'], [0, 1])
        self.assertEqual(report['extrema']['farthest_associated_leaves'], [0])
        self.assertEqual(len(report['extrema']['highest_terminals']), 1)
        endpoint = report['supported_leaves'][0]['endpoint']
        self.assertEqual(report['extrema']['highest_terminals'], [endpoint])
        self.assertEqual(report['decision_basis'], 'ambiguous_refused')
        self.assertFalse((out / 'skeleton.json').exists())
        self.assertEqual(cli.read(out / 'run-record.json')['status'], 'failed')
