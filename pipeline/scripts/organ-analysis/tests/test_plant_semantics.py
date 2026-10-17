import copy
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plant_asset as pa
import plant_semantics as ps
import plant_semantics_cli as cli
import test_plant_asset as fixture_module


class PartialSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.PlantAssetTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.source = self.fixture.package() / 'plant-asset'
        self.output = self.root / 'semantic'
        ps.build(self.source, self.output)
        self.doc = json.loads((self.output / 'semantic.json').read_text())

    def save(self, doc):
        (self.output / 'semantic.json').write_text(json.dumps(doc))

    def test_bytes_relocation_registry_rebuild_and_source_unchanged(self):
        before = ps.tree_hashes(self.source)
        rebuild = self.root / 'rebuild'
        ps.build(self.source, rebuild, self.output / 'identity-registry.json')
        self.assertEqual(ps.tree_hashes(self.output), ps.tree_hashes(rebuild))
        self.assertEqual(before, ps.tree_hashes(rebuild / 'plant-asset'))
        shutil.copytree(rebuild, self.root / 'moved')
        self.assertEqual(ps.validate(self.root / 'moved'), ps.validate(self.output))
        self.assertEqual(before, ps.tree_hashes(self.source))
        with self.assertRaisesRegex(ValueError, 'already exists'):
            ps.build(self.source, rebuild)

    def test_region_candidates_inference_and_queries(self):
        self.assertEqual((len(self.doc['leaves']), len(self.doc['regions'])), (3, 1))
        self.assertEqual(self.doc['accepted_nodes'], [])
        self.assertEqual(self.doc['accepted_internodes'], [])
        self.assertTrue(all(n['node_rank'] is None for n in self.doc['insertion_candidates']))
        groups = ps.query(self.output, 'candidate-groups')
        self.assertFalse(groups['botanical_order_known'])
        statuses = [p['classification'] for g in groups['groups'] for c in g['candidates'] for p in c['source_petioles']]
        self.assertEqual(statuses.count(pa.INFERRED), 1)
        candidate = self.doc['insertion_candidates'][0]
        self.assertEqual(len(ps.query(self.output, 'candidate-leaves', candidate['id'])), 1)
        with self.assertRaises(ValueError):
            ps.query(self.output, 'candidate-leaves', self.doc['regions'][0]['id'])

    def test_false_promotion_roles_sources_and_registry_refs_rejected(self):
        mutations = [lambda d: d.update(schema_version=True),
                     lambda d: d['axes'][0].update(basal_node_known=0),
                     lambda d: d['accepted_nodes'].append({'id': 'invented', 'status': 'accepted', 'evidence': 'claim'}),
                     lambda d: d['regions'][0].update(role='true-leaf'),
                     lambda d: d['leaves'][0].update(source_leaf_id='absent'),
                     lambda d: d['insertion_candidates'][0].update(node_rank=0),
                     lambda d: d.update(registry_path='../other.json'),
                     lambda d: d['insertion_candidates'][0].update(id=d['regions'][0]['id'])]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                doc = copy.deepcopy(self.doc)
                mutate(doc)
                self.save(doc)
                with self.assertRaises(ValueError):
                    ps.validate(self.output)
        self.save(self.doc)

    def test_registry_binding_mapping_kind_and_duplicate_rejections(self):
        original = json.loads((self.output / 'identity-registry.json').read_text())
        mutations = [lambda r: r.update(schema_version=True),
                     lambda r: r['source'].update(plant_json_sha256='0' * 64),
                     lambda r: r['identities'].pop(next(iter(r['identities']))),
                     lambda r: next(iter(r['identities'].values())).update(kind='node'),
                     lambda r: next(iter(r['identities'].values())).update(id='plant:source-label-9')]
        for index, mutate in enumerate(mutations):
            r = copy.deepcopy(original)
            mutate(r)
            path = self.root / ('registry-' + str(index) + '.json')
            path.write_text(json.dumps(r))
            with self.assertRaises(ValueError):
                ps.build(self.source, self.root / ('bad-' + str(index)), path)
        changed = pa.load_asset(self.source)
        changed['limits'].append('changed source identity requires explicit correspondence')
        (self.source / 'plant.json').write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'source mismatch'):
            ps.build(self.source, self.root / 'changed', self.output / 'identity-registry.json')

    def test_copied_geometry_corruption_rejected(self):
        source = pa.load_asset(self.output / 'plant-asset')
        path = self.output / 'plant-asset' / source['leaves'][0]['point_geometry']['path']
        path.write_bytes(path.read_bytes() + b'tamper')
        with self.assertRaises(ValueError):
            ps.validate(self.output)

    def test_generator_provenance_survives_newer_runtime(self):
        with patch.object(ps, 'VERSION', '0.3.0-alpha.999'):
            ps.validate(self.output)
        doc = copy.deepcopy(self.doc)
        doc['generator_version'] = ''
        self.save(doc)
        with self.assertRaises(ValueError):
            ps.validate(self.output)

    def test_registry_binds_extra_and_missing_files(self):
        extra = self.source / 'extra-unreferenced.txt'
        extra.write_text('source bytes')
        with self.assertRaisesRegex(ValueError, 'source mismatch'):
            ps.build(self.source, self.root / 'extra-build', self.output / 'identity-registry.json')
        extra.unlink()
        ps.build(self.source, self.root / 'restored', self.output / 'identity-registry.json')

    def test_cli_build_validate_query(self):
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            self.assertEqual(cli.main(['build', '--source', str(self.source), '--output', str(self.root / 'cli'),
                                       '--registry', str(self.output / 'identity-registry.json')]), 0)
            self.assertEqual(cli.main(['validate', str(self.root / 'cli')]), 0)
            self.assertEqual(cli.main(['query', str(self.root / 'cli'), 'topology']), 0)
        self.assertIn('accepted_internodes', captured.getvalue())


class AcceptedTopologyTests(unittest.TestCase):
    def fixture(self):
        return {'axes': [{'id': 'axis', 'basal_node_known': True, 'basal_node_id': 'n0'}],
                'leaves': [{'id': 'l0', 'accepted_node_id': 'n1'}, {'id': 'l1', 'accepted_node_id': 'n1'}],
                'accepted_nodes': [{'id': 'n0', 'axis_id': 'axis', 'node_rank': 0, 'leaf_ids': [],
                                    'status': 'accepted', 'evidence': 'synthetic fixture only'},
                                   {'id': 'n1', 'axis_id': 'axis', 'node_rank': 1, 'leaf_ids': ['l0', 'l1'],
                                    'status': 'accepted', 'evidence': 'synthetic fixture only'}],
                'accepted_internodes': [{'id': 'i0', 'axis_id': 'axis', 'basal_node_id': 'n0', 'apical_node_id': 'n1',
                                         'status': 'accepted', 'evidence': 'synthetic fixture only'}]}

    def test_accepted_endpoint_count_order_zero_many_and_below_node(self):
        doc = self.fixture()
        self.assertEqual(ps.validate_topology(doc), {'accepted_nodes': 2, 'accepted_internodes': 1})
        self.assertEqual(ps.leaves_at(doc, 'n0', True), [])
        self.assertEqual(len(ps.leaves_at(doc, 'n1', True)), 2)
        self.assertEqual(ps.below_node_phytomer(doc, 'n1')['internode']['id'], 'i0')
        self.assertFalse(ps.below_node_phytomer(doc, 'n0')['basal_boundary_complete'])
        ids = [r['id'] for r in doc['accepted_nodes']]
        doc['axes'][0].update(basal_node_known=False, basal_node_id=None)
        for node in doc['accepted_nodes']:
            node['node_rank'] = None
        ps.validate_topology(doc)
        self.assertEqual([r['id'] for r in doc['accepted_nodes']], ids)

    def test_bad_endpoints_cycles_ranks_and_evidence(self):
        mutations = [lambda d: d['accepted_internodes'][0].update(basal_node_id='candidate'),
                     lambda d: d['accepted_nodes'][1].update(axis_id='other-axis'),
                     lambda d: d['accepted_nodes'][0].update(node_rank=1),
                     lambda d: d['accepted_nodes'][0].update(evidence=''),
                     lambda d: d['accepted_internodes'][0].update(evidence=''),
                     lambda d: d['accepted_nodes'][0].update(leaf_ids=['l0']),
                     lambda d: d['accepted_internodes'].append({'id': 'i1', 'axis_id': 'axis', 'basal_node_id': 'n1',
                         'apical_node_id': 'n0', 'status': 'accepted', 'evidence': 'synthetic'})]
        for mutate in mutations:
            doc = self.fixture()
            mutate(doc)
            with self.assertRaises(ValueError):
                ps.validate_topology(doc)