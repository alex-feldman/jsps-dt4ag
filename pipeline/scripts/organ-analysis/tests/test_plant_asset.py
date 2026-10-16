import contextlib
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from branches import graph_digest, identify_candidates
import package_plant_asset as packager
import plant_asset as pa


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hashes(folder):
    return {p.relative_to(folder).as_posix(): sha(p) for p in sorted(Path(folder).rglob('*')) if p.is_file()}


def viewer_data(path):
    page = Path(path).read_text()
    return json.JSONDecoder().raw_decode(page.split('const D=', 1)[1])[0]


class PlantAssetTests(unittest.TestCase):
    """Synthetic reviewed plant: algorithm petiole (0), human endpoint (1), inferred (2), apex (3)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        positions = {0: [0.0, 0.0, 0.0], 1: [0.0, 0.0, 0.1], 2: [0.0, 0.0, 0.2], 3: [0.0, 0.0, 0.3], 4: [0.0, 0.0, 0.4],
                     5: [0.1, 0.0, 0.1], 6: [0.2, 0.0, 0.1], 7: [0.0, 0.1, 0.2], 8: [0.0, 0.2, 0.2], 9: [-0.1, 0.0, 0.3]}
        main = [(0, 1), (1, 2), (2, 3), (3, 4)]
        edges = main + [(1, 5), (5, 6), (2, 7), (7, 8), (3, 9)]
        ramp = np.arange(1, 11) * 0.002
        blocks = {0: np.column_stack([0.2 + ramp, 0 * ramp, 0.1 + 0 * ramp]),
                  1: np.column_stack([0 * ramp, 0.2 + ramp, 0.2 + 0 * ramp]),
                  2: np.column_stack([0 * ramp, -0.038 - ramp, 0.25 + 0 * ramp]),
                  3: np.column_stack([0 * ramp, 0 * ramp, 0.403 + ramp / 2])}
        stem = np.column_stack([np.zeros(9), np.zeros(9), np.linspace(0.02, 0.38, 9)])
        X = np.concatenate([stem] + [blocks[c] for c in range(4)])
        label = np.array([-2] * 9 + [c for c in range(4) for _ in range(10)])
        self.review = self.root / 'review'
        self.review.mkdir()
        np.savez(self.review / 'leaf_ids2.npz', X=X, label=label, rgb=np.full((len(X), 3), 0.5),
                 vis_count=np.arange(len(X)))
        leaves = [{'leaf_id': 0, 'insertion_along_stem': 0.1, 'insertion_xyz': positions[1], 'petiole_network_length': 0.2,
                   'base_xyz': blocks[0][0].tolist(), 'gap': 0.002, 'base_rule': 'tip'},
                  {'leaf_id': 1, 'insertion_along_stem': 0.2, 'insertion_xyz': positions[2], 'petiole_network_length': 0,
                   'base_xyz': blocks[1][0].tolist(), 'gap': 0.002, 'base_rule': 'nearest'},
                  {'leaf_id': 2, 'insertion_along_stem': 0.2, 'insertion_xyz': positions[2], 'petiole_network_length': 0,
                   'base_xyz': blocks[2][0].tolist(), 'gap': 0.04, 'base_rule': 'nearest'},
                  {'leaf_id': 3, 'insertion_along_stem': 0.4, 'insertion_xyz': positions[4], 'petiole_network_length': 0,
                   'base_xyz': blocks[3][0].tolist(), 'gap': float(np.linalg.norm(blocks[3][0] - positions[4])), 'base_rule': 'tip'}]
        self.graph = {'params': {}, 'petiole_points': {}, 'apex_leaf': 3,
                      'skeleton_nodes': [{'id': n, 'xyz': p} for n, p in positions.items()],
                      'skeleton_edges': [{'a': a, 'b': b, 'main': (a, b) in main} for a, b in edges],
                      'main_stem': [positions[n] for n in range(5)], 'leaves': leaves}
        report = identify_candidates(self.graph, X, label, 0.006)
        self.endpoints = {e['source_node_id']: e['endpoint_id'] for e in report['endpoint_records']}
        bundle = {'schema': 'organ-evidence-v1', 'capture_id': 'tiny-plant', 'units': pa.UNITS,
                  'point_labels': {'file': 'leaf_ids2.npz', 'sha256': sha(self.review / 'leaf_ids2.npz')},
                  'leaves': [{'id': c, 'point_count': 10, 'association_support': 'supported', 'annotation': None,
                              'annotation_status': 'unreviewed', 'geometry_status': 'algorithm_candidate'} for c in range(4)],
                  'source_graph': self.graph}
        (self.review / 'organ-evidence.json').write_text(json.dumps(bundle))
        (self.review / 'branch-candidates.json').write_text(json.dumps(report))
        self.csv = self.root / 'endpoint-review.csv'
        with self.csv.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['capture_id', 'source_graph_sha256', 'endpoint_id', 'source_node_id', 'x', 'y', 'z',
                             'review_group', 'prior_disposition', 'review_status', 'human_category', 'human_notes'])
            for node, eid in sorted(self.endpoints.items()):
                category = 'false-branch' if node == 9 else 'leaf' if node == 8 else 'not reviewed'
                writer.writerow(['tiny-plant', graph_digest(self.graph), eid, node, *positions[node], '', '', 'reviewed', category, ''])
        self.decisions = {'schema': 'plant-asset-decisions', 'schema_version': 1, 'plant_id': 'tiny',
                          'source': 'synthetic fixture', 'source_graph_sha256': graph_digest(self.graph),
                          'leaves': {'0': {'role': 'true-leaf', 'petiole': {'source': 'algorithm'}},
                                     '1': {'role': 'true-leaf', 'petiole': {'source': 'endpoint', 'endpoint_id': self.endpoints[8]}},
                                     '2': {'role': 'cotyledon', 'petiole': {'source': 'inferred', 'method': packager.INFERENCE_METHOD,
                                                                           'human_review': 'synthetic acceptance'}},
                                     '3': {'role': 'combined-apex-region', 'petiole': None}}}
        self.decisions_path = self.root / 'decisions.json'
        self.save_decisions()

    def save_decisions(self):
        self.decisions_path.write_text(json.dumps(self.decisions))

    def package(self, name='out'):
        out = self.root / name
        with contextlib.redirect_stdout(io.StringIO()):
            packager.main(['--review', str(self.review), '--decisions', str(self.decisions_path),
                           '--endpoint-review', str(self.csv), '--output', str(out)])
        return out

    def test_package_emits_valid_relocatable_contract(self):
        sources = tree_hashes(self.review)
        out = self.package()
        folder = out / 'plant-asset'
        asset = pa.load_asset(folder)
        self.assertEqual((asset['schema'], asset['schema_version']), ('plant-asset', 1))
        self.assertEqual((asset['units'], asset['metric_scale']), (pa.UNITS, None))
        summary = pa.validate_asset(folder)
        self.assertEqual(summary['extracted_petioles'], ['tiny:leaf:0', 'tiny:leaf:1'])
        self.assertEqual(summary['inferred_petioles'], ['tiny:leaf:2'])
        self.assertEqual(summary['edge_classification'], {'main_axis': 4, 'extracted_petiole': 4, 'unassigned': 0})
        petioles = {p['leaf_id']: p for p in asset['petioles']}
        self.assertEqual(petioles['tiny:leaf:0']['source_node_ids'], [1, 5, 6])
        self.assertEqual(petioles['tiny:leaf:1']['source_node_ids'], [2, 7, 8])
        inferred = petioles['tiny:leaf:2']
        self.assertEqual((inferred['classification'], inferred['geometry_status']), ('inferred-petiole', 'inferred'))
        self.assertEqual(inferred['inference']['reference_leaf_ids'], [1, 0])
        self.assertAlmostEqual(inferred['polyline'][0][2], 0.25, places=12)
        self.assertIsNone(next(l for l in asset['leaves'] if l['source_cluster_id'] == 3)['petiole_id'])
        self.assertNotIn(9, [n['id'] for n in pa.read(folder / 'skeleton-extracted-reviewed.json')['skeleton_nodes']])
        self.assertEqual(json.loads((out / 'checks.json').read_text())['rejected_endpoints'], {self.endpoints[9]: 'endpoint review CSV: false-branch'})
        with np.load(folder / 'leaves/leaf-1.npz') as data:
            self.assertEqual(data['source_point_row_ids'].tolist(), list(range(19, 29)))
        for leaf in asset['leaves']:
            self.assertFalse(Path(leaf['point_geometry']['path']).is_absolute())
        moved = self.root / 'elsewhere' / 'copy'
        shutil.copytree(folder, moved)
        shutil.rmtree(out)
        self.assertEqual(pa.validate_asset(moved)['leaves'], 4)
        self.assertEqual(tree_hashes(self.review), sources)

    def test_repeat_packaging_is_byte_identical(self):
        first, second = self.package('first'), self.package('second')
        self.assertEqual(tree_hashes(first / 'plant-asset'), tree_hashes(second / 'plant-asset'))
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.package('first')

    def test_validator_refuses_contract_violations(self):
        folder = self.package() / 'plant-asset'
        original = pa.load_asset(folder)
        def broken(change):
            asset = copy.deepcopy(original)
            change(asset)
            with self.assertRaises(pa.PlantAssetError):
                pa.validate_asset(folder, asset)
        inferred = lambda a: next(p for p in a['petioles'] if p['inference'])
        broken(lambda a: inferred(a).update(geometry_status='measured'))
        broken(lambda a: inferred(a).update(classification='extracted-petiole-candidate'))
        broken(lambda a: a['attachment_nodes'][0].update(geometry_status='inferred'))
        broken(lambda a: a['petioles'][0].update(attachment_id='tiny:attachment:missing'))
        broken(lambda a: a['attachment_nodes'][0]['xyz'].__setitem__(0, 1.0))
        broken(lambda a: a['leaves'][0]['point_geometry'].update(sha256='bad'))
        broken(lambda a: a['leaves'][0]['point_geometry'].update(path='/abs/leaf-0.npz'))
        broken(lambda a: a['leaves'][0]['point_geometry'].update(path='../leaf-0.npz'))
        broken(lambda a: a.update(metric_scale=1.0))
        broken(lambda a: a.update(schema_version=2))
        broken(lambda a: a.pop('limits'))
        broken(lambda a: a['leaves'][1].update(petiole_id=None))
        broken(lambda a: a['ordered_attachment_ids'].reverse())
        broken(lambda a: a['reviewed_graph']['edge_classification'].update(unassigned=1))
        self.assertTrue(pa.claims_measured('measured petiole'))
        self.assertFalse(pa.claims_measured('unresolved zero sampling interval; not measured biological internode'))

    def test_single_leaf_export_and_reload_preserve_status(self):
        folder = self.package() / 'plant-asset'
        before = tree_hashes(folder)
        asset = pa.load_asset(folder)
        for cluster, status in [(1, 'extracted'), (2, 'inferred-petiole'), (3, 'none')]:
            target = self.root / ('leaf-' + str(cluster))
            with contextlib.redirect_stdout(io.StringIO()):
                import export_leaf
                export_leaf.main(['--asset', str(folder), '--leaf', str(cluster), '--output', str(target)])
            moved = self.root / 'moved' / target.name
            shutil.copytree(target, moved)
            document, arrays = pa.load_leaf_export(moved)
            leaf = pa.find_leaf(asset, cluster)
            self.assertEqual(document['petiole_status'], status)
            self.assertEqual(document['petiole'], next((p for p in asset['petioles'] if p['id'] == leaf['petiole_id']), None))
            with np.load(folder / leaf['point_geometry']['path']) as source:
                for name in source.files:
                    np.testing.assert_array_equal(arrays[name], source[name])
            if status == 'inferred-petiole':
                self.assertEqual(document['attachment']['geometry_status'], 'inferred')
                relabelled = copy.deepcopy(document)
                relabelled['petiole']['geometry_status'] = 'extracted'
                (moved / 'leaf.json').write_text(json.dumps(relabelled))
                with self.assertRaisesRegex(pa.PlantAssetError, 'Inferred status'):
                    pa.load_leaf_export(moved)
        self.assertEqual(tree_hashes(folder), before)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            pa.export_leaf(folder, 1, self.root / 'leaf-1')
        with self.assertRaisesRegex(pa.PlantAssetError, 'No leaf'):
            pa.export_leaf(folder, 99, self.root / 'leaf-99')

    def test_viewer_embeds_reloadable_leaf_exports_offline(self):
        out = self.package()
        page = (out / 'viewer.html').read_text()
        for forbidden in ['http://', 'https://', '<script src', 'fetch(', 'XMLHttpRequest']:
            self.assertNotIn(forbidden, page)
        self.assertIn('id="leafChoice"', page)
        self.assertIn('id="exportLeaf"', page)
        payload = viewer_data(out / 'viewer.html')['plantAsset']
        self.assertEqual(payload['plant_json_sha256'], sha(out / 'plant-asset/plant.json'))
        self.assertEqual(sorted(payload['exports']), ['0', '1', '2', '3'])
        folder = out / 'plant-asset'
        for cluster, embedded in payload['exports'].items():
            download = self.root / ('download-' + cluster + '.json')
            download.write_text(json.dumps(embedded) + '\n')
            document, arrays = pa.load_leaf_export(download)
            cli = pa.leaf_export_document(folder, int(cluster))
            for key in cli:
                if key not in ('leaf', 'export_route'):
                    self.assertEqual(document[key], cli[key], key)
            self.assertEqual(document['export_route'], 'viewer-download')
            with np.load(folder / document['leaf']['point_geometry']['source_path']) as source:
                np.testing.assert_array_equal(arrays['X'], source['X'])
                np.testing.assert_array_equal(arrays['source_point_row_ids'], source['source_point_row_ids'])
        self.assertEqual(payload['exports']['2']['petiole']['classification'], 'inferred-petiole')

    def test_decisions_must_match_graph_and_evidence(self):
        def refused(message, change):
            decisions = copy.deepcopy(self.decisions)
            change(decisions)
            self.decisions_path.write_text(json.dumps(decisions))
            with self.assertRaisesRegex(ValueError, message):
                packager.build(self.review, self.decisions_path, self.csv)
        refused('different source graph', lambda d: d.update(source_graph_sha256='0' * 64))
        refused('no extracted algorithm path', lambda d: d['leaves']['1'].update(petiole={'source': 'algorithm'}))
        refused('Unknown endpoint', lambda d: d['leaves']['1']['petiole'].update(endpoint_id='zz-9'))
        refused('rejected or is not a side tip', lambda d: d['leaves']['1']['petiole'].update(endpoint_id=self.endpoints[9]))
        refused('needs a decision', lambda d: d['leaves'].pop('2'))
        refused('Unknown inferred keys', lambda d: d['leaves']['2']['petiole'].update(geometry_status='measured'))
        refused('never claim measurement', lambda d: d['leaves']['1']['petiole'].update(geometry_status='extracted; measured'))
        refused('petiole_unresolved_reason', lambda d: d['leaves']['2'].update(petiole=None))
        refused('apex', lambda d: d['leaves']['0'].update(role='combined-apex-region', petiole=None))
        self.save_decisions()
        rows = self.csv.read_text().replace(graph_digest(self.graph), '1' * 64)
        self.csv.write_text(rows)
        with self.assertRaisesRegex(ValueError, 'different graph'):
            packager.build(self.review, self.decisions_path, self.csv)


if __name__ == '__main__':
    unittest.main()
