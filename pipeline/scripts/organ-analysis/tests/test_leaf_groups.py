import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import render_viewer
from viewer import check_leaf_groups, write_viewer

# SHA-256 of the fixture page written by viewer.py before leaf groups existed
# (origin/main efa73c4, v0.3.0-alpha.13). The ungrouped page must not change.
UNGROUPED_PAGE_SHA256 = 'b87c72e30b8cb73a0d5e9a25aaf7f654c5d2a1954ffd83aed39bda7abced0b2d'


def page_data(path):
    return json.JSONDecoder().raw_decode(path.read_text().split('const D=', 1)[1])[0]


class LeafGroupViewerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        points = np.array([[0.0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0.5, 0, 0], [0.6, 0, 0], [0.5, 0.1, 0],
                           [1.0, 0, 0], [1.1, 0, 0], [1.0, 0.1, 0], [1.5, 0, 0], [1.6, 0, 0], [1.5, 0.1, 0]])
        labels = np.array([4, 4, 4, 17, 17, 17, 13, 13, 13, 15, 15, 15])
        self.labels = self.root/'pts.npz'
        np.savez(self.labels, X=points, label=labels)
        self.bundle = {'capture_id': 'tiny-plant', 'source_graph': None,
                       'leaves': [{'id': 4, 'point_count': 3}, {'id': 13, 'point_count': 3},
                                  {'id': 15, 'point_count': 3}, {'id': 17, 'point_count': 3}]}
        self.groups = {
            'schema': 'organ-leaf-groups', 'schema_version': 1, 'capture_id': 'tiny-plant',
            'source_labels': {'path': str(self.labels), 'sha256': hashlib.sha256(self.labels.read_bytes()).hexdigest()},
            'reviewer': 'Alex', 'date': '2026-10-19', 'status': 'accepted',
            'groups': [{'id': 'G1', 'kind': 'same_leaf', 'members': [4, 17]},
                       {'id': 'G2', 'kind': 'side_branch_pair', 'members': [13, 15]}]}

    def render(self, name, groups=None):
        out = self.root/name
        out.mkdir()
        write_viewer(out, self.bundle, labels_path=self.labels, leaf_groups=groups)
        return out/'viewer.html'

    def test_no_groups_output_unchanged(self):
        page = self.render('plain')
        self.assertEqual(hashlib.sha256(page.read_bytes()).hexdigest(), UNGROUPED_PAGE_SHA256)
        self.assertNotIn('leafGroups', page_data(page))

    def test_grouping_renders_shared_color_and_combined_label(self):
        page = self.render('grouped', self.groups)
        text, data = page.read_text(), page_data(page)
        display = {g['id']: g for g in data['leafGroups']['display']}
        self.assertEqual(display['G1']['text'], 'L: C4+C17')
        self.assertEqual(display['G1']['colorId'], 4)
        self.assertIn(display['G1']['anchor'], data['points']['4']+data['points']['17'])
        self.assertEqual(display['G2']['text'], 'Side branch pair: C13 | C15')
        # Members keep their own IDs, points and per-cluster labels: nothing is merged.
        self.assertEqual(sorted(data['points']), ['13', '15', '17', '4'])
        self.assertEqual(sorted(data['pointLabels']), ['13', '15', '17', '4'])
        self.assertIn("text='C'+item.id", text)
        # Shared color applies only to same_leaf members; every color site goes through leafColor.
        self.assertIn("if(g.kind==='same_leaf')groupColor[m]=g.colorId", text)
        for site in ('leafColor(p[3])', 'leafColor(item.id)', 'leafColor(leaf.id)', 'leafColor(l.leaf_id)'):
            self.assertIn(site, text)
        self.assertNotRegex(text, r'colors\[(p\[3\]|item\.id|leaf\.id|l\.leaf_id)%colors\.length\]')
        self.assertIn('id="leafGroups"', text)
        self.assertEqual(data['leafGroups']['reviewer'], 'Alex')

    def test_unknown_cluster_rejected(self):
        bad = copy.deepcopy(self.groups)
        bad['groups'][0]['members'] = [4, 99]
        with self.assertRaisesRegex(ValueError, 'unknown cluster ID'):
            self.render('bad', bad)
        self.assertFalse((self.root/'bad/viewer.html').exists())

    def test_label_hash_capture_and_membership_checks(self):
        for change, message in [(lambda g: g['source_labels'].update(sha256='0'*64), 'does not match'),
                                (lambda g: g.update(capture_id='other'), 'does not match evidence capture'),
                                (lambda g: g['groups'].append({'id': 'G3', 'kind': 'same_leaf', 'members': [17, 13]}),
                                 'two same_leaf groups'),
                                (lambda g: g['groups'][1].update(kind='merge'), 'kind must be'),
                                (lambda g: g.update(unresolved_clusters=[{'id': 4}]), 'unresolved and grouped')]:
            bad = copy.deepcopy(self.groups)
            change(bad)
            with self.assertRaisesRegex(ValueError, message):
                check_leaf_groups(bad, self.bundle, self.labels)

    def test_entry_point_writes_new_folder_and_leaves_source_unchanged(self):
        source = self.root/'source'
        source.mkdir()
        (source/'leaf_ids2.npz').write_bytes(self.labels.read_bytes())
        (source/'organ-evidence.json').write_text(json.dumps(self.bundle))
        groups_path = self.root/'groups.json'
        groups_path.write_text(json.dumps(self.groups))
        before = {p.name: p.read_bytes() for p in source.iterdir()}
        out = self.root/'viewer-out'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(render_viewer.main(['--source', str(source), '--output', str(out),
                                                 '--leaf-groups', str(groups_path)]), 0)
        self.assertEqual({p.name: p.read_bytes() for p in source.iterdir()}, before)
        record = json.loads((out/'viewer-record.json').read_text())
        self.assertTrue(record['source_unchanged'])
        self.assertEqual(record['leaf_groups']['sha256'], hashlib.sha256(groups_path.read_bytes()).hexdigest())
        with self.assertRaisesRegex(ValueError, 'fresh output'):
            render_viewer.main(['--source', str(source), '--output', str(out)])
        bad = copy.deepcopy(self.groups)
        bad['groups'][0]['members'] = [4, 99]
        groups_path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, 'unknown cluster ID'):
            render_viewer.main(['--source', str(source), '--output', str(self.root/'refused'),
                                '--leaf-groups', str(groups_path)])
        self.assertFalse((self.root/'refused').exists())


if __name__ == '__main__':
    unittest.main()
