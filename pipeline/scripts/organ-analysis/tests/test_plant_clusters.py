import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plant_clusters as pc
import cluster_plants as cli
from geometry import load_ply

SPLAT = ['x', 'y', 'z', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'f_dc_0', 'f_dc_1', 'f_dc_2']


def write_ply(path, X, opacity, extra=True):
    """Synthetic splat PLY; extra adds mixed-type attributes so byte copies are tested."""
    props = [(p, '<f4') for p in SPLAT] + ([('rot_0', '<f4'), ('label', '<i4'), ('red', 'u1')] if extra else [])
    data = np.zeros(len(X), dtype=props)
    data['x'], data['y'], data['z'] = X[:, 0], X[:, 1], X[:, 2]
    data['opacity'] = opacity
    for p in ['scale_0', 'scale_1', 'scale_2']:
        data[p] = -4.6
    if extra:
        data['rot_0'] = np.arange(len(X)) * .5
        data['label'] = np.arange(len(X)) * 7 - 3
        data['red'] = np.arange(len(X)) % 256
    names = {'<f4': 'float', '<i4': 'int', 'u1': 'uchar'}
    header = ('ply\nformat binary_little_endian 1.0\ncomment synthetic\nelement vertex %d\n' % len(X) +
              ''.join('property %s %s\n' % (names[t], p) for p, t in props) + 'end_header\n')
    with open(path, 'wb') as f:
        f.write(header.encode())
        f.write(data.tobytes())
    return data


def brute_components(X, r):
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    return connected_components(D <= r, directed=False)[1]


def brute_minimax(X):
    M = np.linalg.norm(X[:, None] - X[None], axis=2)
    for k in range(len(X)):
        M = np.minimum(M, np.maximum(M[:, k:k + 1], M[k:k + 1, :]))
    return M


def canonical(lab):
    return pc.relabel_by_size(lab)[0]


def blobs(seed=1):
    rng = np.random.default_rng(seed)
    parts = [rng.normal(c, s, (n, 3)) for c, s, n in [((0, 0, 0), .05, 60), ((.6, 0, 0), .04, 50), ((0, .5, .2), .03, 30)]]
    return np.concatenate(parts + [rng.uniform(-.3, .9, (20, 3))])


class SpanningTreeTests(unittest.TestCase):
    def test_components_equal_bruteforce_radius_graph(self):
        X = blobs()
        mst = pc.build_mst(X)
        self.assertEqual(mst['info']['method'], 'delaunay')
        self.assertEqual(len(mst['W']), len(X) - 1)
        for r in [.01, .03, .05, .08, .12, .2, .5]:
            np.testing.assert_array_equal(canonical(pc.components_at(mst, r)), canonical(brute_components(X, r)))

    def test_join_heights_equal_bruteforce_minimax(self):
        X = blobs(2)
        M = brute_minimax(X)
        groups = {'a': np.arange(0, 60), 'b': np.arange(60, 110), 'c': np.arange(110, 140),
                  'scattered': np.array([140, 150, 159])}
        heights, events = pc.join_heights(pc.build_mst(X), groups)
        self.assertEqual(len(heights), 6)
        for (a, b), h in heights.items():
            self.assertAlmostEqual(h, M[np.ix_(groups[a], groups[b])].min(), places=12)
        self.assertEqual([e['radius'] for e in events], sorted(e['radius'] for e in events))
        overlap, _ = pc.join_heights(pc.build_mst(X), {'a': np.arange(0, 10), 'b': np.arange(5, 20)})
        self.assertEqual(overlap[('a', 'b')], 0.0)

    def test_known_gap_and_join_radius_with_degenerate_fallback(self):
        line_a = np.column_stack([np.arange(101) * .01, np.zeros(101), np.zeros(101)])
        line_b = line_a + [1.5, 0, 0]
        bridge = np.array([[1.1, 0, 0], [1.2, 0, 0], [1.3, 0, 0], [1.4, 0, 0]])
        X = np.concatenate([line_a, line_b, bridge])
        mst = pc.build_mst(X)
        self.assertEqual(mst['info']['method'], 'complete_graph_degenerate_input')
        lab = canonical(pc.components_at(mst, .02))
        self.assertEqual(len(set(lab[:202])), 2)
        heights, _ = pc.join_heights(pc.build_mst(X[:202]), {'a': np.arange(101), 'b': np.arange(101, 202)})
        self.assertAlmostEqual(heights[('a', 'b')], .5, places=12)
        heights, _ = pc.join_heights(mst, {'a': np.arange(101), 'b': np.arange(101, 202)})
        self.assertAlmostEqual(heights[('a', 'b')], .1, places=12)
        self.assertAlmostEqual(pc.direct_gap(line_a, line_b)[0], .5, places=12)
        gaps = pc.pairwise_gaps(X, np.array([0] * 101 + [1] * 101 + [-1] * 4), [0, 1])
        self.assertAlmostEqual(gaps[(0, 1)], .5, places=12)

    def test_duplicates_merged_and_large_degenerate_refused(self):
        X = blobs()
        Xd = np.concatenate([X, X[:10]])
        mst = pc.build_mst(Xd)
        self.assertEqual(mst['info']['duplicate_rows_merged'], 10)
        lab = pc.components_at(mst, .03)
        np.testing.assert_array_equal(lab[:10], lab[-10:])
        np.testing.assert_array_equal(canonical(lab[:len(X)]), canonical(brute_components(X, .03)))
        flat = np.column_stack([np.random.default_rng(0).uniform(0, 1, (pc.COMPLETE_GRAPH_LIMIT + 1, 2)),
                                np.zeros(pc.COMPLETE_GRAPH_LIMIT + 1)])
        with self.assertRaisesRegex(ValueError, 'Degenerate'):
            pc.build_mst(flat)

    def test_named_clusters_and_sweep(self):
        X = blobs()
        mst = pc.build_mst(X)
        comp, sizes = pc.relabel_by_size(pc.components_at(mst, .08))
        named = pc.named_clusters(comp, sizes, 25)
        self.assertTrue(np.all(np.diff(sizes) <= 0))
        self.assertEqual(set(named[named >= 0]), set(np.flatnonzero(sizes >= 25)))
        rows = pc.sweep(mst, .01, [1, 8, 1000], min_size=25)
        self.assertEqual(rows[-1]['components'], 1)
        self.assertEqual(rows[1]['top_sizes'], [int(s) for s in sizes[:8]])


class GuardTests(unittest.TestCase):
    def test_guard_refuses_over_budget_and_allows_small_radius(self):
        X = np.random.default_rng(3).uniform(0, 1, (3000, 3))
        tree = cKDTree(X)
        with self.assertRaises(pc.PairBudgetExceeded):
            pc.guarded_query_pairs(tree, X, .3, budget=100_000)
        pairs = pc.guarded_query_pairs(tree, X, .03, budget=100_000)
        self.assertEqual(len(pairs), len(tree.query_pairs(.03)))
        exact = tree.count_neighbors(tree, .1) - len(X)
        est = pc.estimate_pairs(tree, X, .1, sample=3000)
        self.assertAlmostEqual(est, exact / 2, delta=1e-6 * exact)


class DerivativeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_rows_copied_byte_for_byte_and_existing_refused(self):
        X = blobs()
        src = self.root / 'src.ply'
        data = write_ply(src, X, np.full(len(X), 3.0))
        before = src.read_bytes()
        rows = np.array([0, 5, 6, 40, 159])
        dest = self.root / 'out.ply'
        digest = pc.write_derivative(src, rows, dest, data.dtype)
        raw = dest.read_bytes()
        self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
        header, body = raw.split(b'end_header\n')
        self.assertEqual(header + b'end_header\n', before.split(b'end_header\n')[0].replace(b'element vertex 160', b'element vertex 5') + b'end_header\n')
        self.assertEqual(body, data[rows].tobytes())
        self.assertEqual(load_ply(dest).dtype, data.dtype)
        self.assertEqual(src.read_bytes(), before)
        with self.assertRaises(FileExistsError):
            pc.write_derivative(src, rows, dest, data.dtype)
        for bad in [np.array([5, 0]), np.array([0, 160]), np.array([1, 1]), np.array([], int)]:
            with self.assertRaisesRegex(ValueError, 'Rows'):
                pc.write_derivative(src, bad, self.root / 'bad.ply', data.dtype)

    def test_trailing_bytes_refused(self):
        src = self.root / 'src.ply'
        data = write_ply(src, blobs(), np.full(160, 3.0))
        with src.open('ab') as f:
            f.write(b'\0' * 7)
        with self.assertRaisesRegex(ValueError, 'extra or missing'):
            pc.ply_layout(src, data.dtype)


def scene():
    """Two separated plants A and B (jittered 6^3 grids, spacing 0.01), a 5-point unclustered
    stub near B, and four faint Gaussians: near A, near B, between, near the stub."""
    rng = np.random.default_rng(7)
    g = np.stack(np.meshgrid(*[np.arange(6) * .01] * 3, indexing='ij'), -1).reshape(-1, 3)
    A = g + rng.uniform(-1e-4, 1e-4, g.shape)
    B = g + [.5, 0, 0] + rng.uniform(-1e-4, 1e-4, g.shape)
    stub = np.column_stack([.6 + np.arange(5) * .01, np.full(5, .025), np.full(5, .025)])
    faint = {'near_a': [-.02, 0, 0], 'near_b': [.5, -.02, 0], 'between': [.25, 0, 0], 'near_stub': [.62, .025, .045]}
    X = np.concatenate([[faint['near_a']], A, [faint['near_b']], B, stub, [faint['between']], [faint['near_stub']]])
    op = np.full(len(X), 3.0)
    idx = {'near_a': 0, 'A': np.arange(1, 217), 'near_b': 217, 'B': np.arange(218, 434), 'stub': np.arange(434, 439),
           'between': 439, 'near_stub': 440}
    op[[0, 217, 439, 440]] = -3.0
    return X, op, idx


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.X, op, self.idx = scene()
        self.ply = self.root / 'model.ply'
        self.data = write_ply(self.ply, self.X, op)
        self.digest = hashlib.sha256(self.ply.read_bytes()).hexdigest()
        self.spec = {'capture_id': 'two-plants', 'inputs': {'ply': 'model.ply'},
                     'parameters': {'radius_mult': 3, 'min_cluster_size': 50, 'sweep_mults': [1, 3, 10]}}
        self.config = self.root / 'config.json'
        self.save()

    def save(self):
        self.config.write_text(json.dumps(self.spec))

    def invoke(self, output, *args):
        with contextlib.redirect_stdout(io.StringIO()):
            return cli.main(['--config', str(self.config), '--output', str(output), *args])

    def test_cluster_report_selects_nothing(self):
        out = self.root / 'clusters'
        self.assertEqual(self.invoke(out), 0)
        report = cli.read(out / 'clusters.json')
        self.assertEqual({k: v['size'] for k, v in report['clusters'].items()}, {'c0': 216, 'c1': 216})
        self.assertEqual(report['unclustered_centers'], 5)
        self.assertAlmostEqual(report['radius'], 3 * report['median_nn_spacing_d1'])
        self.assertAlmostEqual(report['direct_gap_opacity_kept']['c0-c1'], .45, delta=1e-3)
        self.assertAlmostEqual(report['join_radius_opacity_kept']['c0-c1'], .45, delta=1e-3)
        self.assertAlmostEqual(report['all_rows']['join_radius']['c0-c1'], .25, delta=1e-3)
        self.assertNotIn('target', json.dumps(report).lower().replace('not a target choice', ''))
        self.assertFalse((out / 'selection.json').exists())
        record = cli.read(out / 'run-record.json')
        self.assertEqual(record['status'], 'complete')
        self.assertEqual(record['input_sha256'][str(self.ply)], self.digest)
        self.assertEqual(record['spec']['parameters']['opacity_cutoff'], pc.DEFAULTS['opacity_cutoff'])
        with np.load(out / 'labels.npz') as arrays:
            np.testing.assert_array_equal(arrays['kept_rows'], np.sort(np.concatenate([self.idx['A'], self.idx['B'], self.idx['stub']])))
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.invoke(out)

    def derivatives(self, *extra):
        self.spec.update(source_ply_sha256=self.digest, expected_cluster_sizes={'c0': 216, 'c1': 216},
                         derivatives=[{'name': 'plant-a', 'clusters': [0], 'basis': 'test fixture'},
                                      {'name': 'plant-b', 'clusters': [1], 'expand_unclustered_within': .1, 'basis': 'test fixture'},
                                      *extra])
        self.save()

    def test_derivatives_follow_faint_rule_and_copy_rows(self):
        np.save(self.root / 'a_rows.npy', self.idx['A'])
        self.derivatives({'name': 'from-rows', 'rows': 'a_rows.npy', 'basis': 'row list'})
        out = self.root / 'selected'
        self.invoke(out)
        sel = cli.read(out / 'selection.json')
        by = {d['name']: d for d in sel['derivatives']}
        a = np.load(out / 'plant-a_rows.npy')
        b = np.load(out / 'plant-b_rows.npy')
        np.testing.assert_array_equal(a, np.concatenate([[0], self.idx['A']]))
        np.testing.assert_array_equal(b, np.sort(np.concatenate([[217, 440], self.idx['B'], self.idx['stub']])))
        np.testing.assert_array_equal(np.load(out / 'from-rows_rows.npy'), a)
        self.assertEqual(by['plant-b']['expansion']['unclustered_centers_added'], 5)
        self.assertEqual(by['plant-b']['faint_rows_attached'], 2)
        self.assertNotIn(439, a)
        self.assertNotIn(439, b)
        self.assertEqual(sel['row_overlap_between_derivatives'], {'plant-a|plant-b': 0, 'plant-a|from-rows': 217, 'plant-b|from-rows': 0})
        for name, rows in [('plant-a', a), ('plant-b', b)]:
            raw = (out / (name + '.ply')).read_bytes()
            self.assertEqual(raw.split(b'end_header\n')[1], self.data[rows].tobytes())
            self.assertEqual(by[name]['sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(hashlib.sha256(self.ply.read_bytes()).hexdigest(), self.digest)
        self.assertEqual(cli.read(out / 'run-record.json')['status'], 'complete')

    def test_selection_refusals(self):
        self.derivatives()
        self.spec['source_ply_sha256'] = '0' * 64
        self.save()
        with self.assertRaisesRegex(ValueError, 'another model'):
            self.invoke(self.root / 'x1')
        self.assertFalse((self.root / 'x1').exists())
        self.derivatives()
        self.spec['expected_cluster_sizes'] = {'c0': 215}
        self.save()
        with self.assertRaisesRegex(ValueError, 'expected_cluster_sizes'):
            self.invoke(self.root / 'x2')
        record = cli.read(self.root / 'x2' / 'run-record.json')
        self.assertEqual(record['status'], 'failed')
        self.assertFalse((self.root / 'x2' / 'plant-a.ply').exists())
        self.derivatives({'name': 'faint', 'rows': [0, 1], 'basis': 'includes a faint row'})
        with self.assertRaisesRegex(ValueError, 'not opacity-kept'):
            self.invoke(self.root / 'x3')
        self.derivatives({'name': 'ghost', 'clusters': [9], 'basis': 'no such cluster'})
        with self.assertRaisesRegex(ValueError, 'Unknown cluster'):
            self.invoke(self.root / 'x4')
        self.derivatives({'name': 'nobasis', 'clusters': [0]})
        with self.assertRaisesRegex(ValueError, 'basis'):
            self.invoke(self.root / 'x5')
        self.spec.pop('source_ply_sha256')
        self.spec['derivatives'] = [{'name': 'p', 'clusters': [0], 'basis': 'b'}]
        self.save()
        with self.assertRaisesRegex(ValueError, 'source_ply_sha256'):
            self.invoke(self.root / 'x6')

    def test_validate_only_and_parameters(self):
        self.assertEqual(self.invoke(self.root / 'v', '--validate-only'), 0)
        self.assertFalse((self.root / 'v').exists())
        for bad in [{'radius_mult': 0}, {'min_cluster_size': 1.5}, {'opacity_cutoff': 1}, {'unknown': 1}, {'sweep_mults': []}]:
            self.spec['parameters'] = bad
            self.save()
            with self.assertRaises(ValueError):
                self.invoke(self.root / 'v')


if __name__ == '__main__':
    unittest.main()
