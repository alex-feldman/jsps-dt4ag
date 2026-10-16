"""Memory-lean single-linkage clustering of Gaussian-splat centers into candidate plants.

Every clustering and connection question is answered from one Euclidean minimum
spanning tree (MST) built on 3D Delaunay edges, so no step materializes "all point
pairs within r". Memory is O(N). See DESIGN.md for the method and its provenance,
README.md for the command that uses it. Distances are in the PLY's own scale-free
scene units. This module selects no plant: identity choices are explicit inputs.
"""
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial import Delaunay, QhullError, cKDTree

from analyze_organs import DEFAULTS as ORGAN_DEFAULTS
from geometry import load_ply

# Experimental settings from the October 2026 Welzo M1 separation; not universal tuning.
DEFAULTS = {
    'opacity_cutoff': ORGAN_DEFAULTS['opacity_cutoff'],
    'radius_mult': 6.0,
    'min_cluster_size': 300,
    'sweep_mults': [1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40],
    'sweep_min_size': 500,
    'sweep_top': 8,
    'knn_fallback': 8,
    'all_rows': True,
    'max_gap_clusters': 100,
}
PAIR_BUDGET = 4_000_000
COMPLETE_GRAPH_LIMIT = 1000


# ----------------------------------------------------------------------------- measurement

def _status_kib(field):
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith(field + ':'):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


def _reset_peak():
    try:
        with open('/proc/self/clear_refs', 'w') as f:
            f.write('5')
        return True
    except OSError:
        return False


class StepMeter:
    """Wall time and peak RSS (Linux VmHWM after a per-step reset) for each named step.
    Peak values are None where /proc is unavailable."""

    def __init__(self):
        self.steps = []

    @contextmanager
    def step(self, name):
        reset = _reset_peak()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.steps.append({'step': name, 'wall_s': round(time.perf_counter() - t0, 3),
                               'peak_rss_kib': _status_kib('VmHWM') if reset else None})

    def peak_kib(self):
        values = [s['peak_rss_kib'] for s in self.steps if s['peak_rss_kib'] is not None]
        return max(values) if values else None


# ----------------------------------------------------------------------------- input and output

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def centers(data):
    """float64 (N, 3) centers of a structured splat array from geometry.load_ply."""
    return np.stack([data['x'], data['y'], data['z']], 1).astype(np.float64)


def opacity_kept_rows(data, cutoff):
    """Source row indices whose sigmoid(opacity logit) exceeds cutoff (the organ-analysis rule)."""
    return np.flatnonzero(sigmoid(np.asarray(data['opacity'], np.float64)) > cutoff)


def load_centers(path, cutoff):
    """Read a splat PLY with geometry.load_ply. Returns (data, all-row centers, kept rows)."""
    data = load_ply(path)
    return data, centers(data), opacity_kept_rows(data, cutoff)


def ply_layout(path, dtype):
    """Raw header lines, vertex count and data offset of a PLY already accepted by
    geometry.load_ply with this dtype. Refuses bytes before or after the vertex block."""
    lines = []
    with open(path, 'rb') as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError('Truncated PLY header')
            lines.append(line)
            if line.strip() == b'end_header':
                break
        offset = f.tell()
    counts = [int(l.split()[2]) for l in lines if l.split()[:2] == [b'element', b'vertex']]
    if len(counts) != 1:
        raise ValueError('Expected one vertex element')
    if Path(path).stat().st_size != offset + dtype.itemsize * counts[0]:
        raise ValueError('PLY has extra or missing bytes after the vertex block')
    return lines, counts[0], offset


def check_rows(rows, count):
    rows = np.asarray(rows)
    if rows.ndim != 1 or rows.size == 0 or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError('Rows must be a nonempty 1-D integer array')
    rows = rows.astype(np.int64)
    if rows[0] < 0 or rows[-1] >= count or np.any(np.diff(rows) <= 0):
        raise ValueError('Rows must be strictly increasing source row indices within the PLY')
    return rows


def write_derivative(source, rows, dest, dtype, chunk=8192):
    """Copy source PLY rows byte for byte, by index, into a new PLY at dest.

    Every property is kept; the header is identical except the vertex count. The source
    is mapped read-only and never written. dest must not exist. The written file is
    re-read with geometry.load_ply and compared with the source rows. Returns its sha256."""
    import hashlib
    lines, count, offset = ply_layout(source, dtype)
    rows = check_rows(rows, count)
    header = b''.join((b'element vertex %d' % len(rows)) + (b'\r\n' if l.endswith(b'\r\n') else b'\n')
                      if l.split()[:2] == [b'element', b'vertex'] else l for l in lines)
    src = np.memmap(source, dtype=dtype, mode='r', offset=offset, shape=(count,))
    try:
        with open(dest, 'xb') as f:
            f.write(header)
            for s in range(0, len(rows), chunk):
                f.write(src[rows[s:s + chunk]].tobytes())
        back = load_ply(dest)
        if len(back) != len(rows) or back.dtype != dtype:
            raise ValueError('Derivative re-read differs in length or layout')
        for s in range(0, len(rows), chunk):
            if back[s:s + chunk].tobytes() != src[rows[s:s + chunk]].tobytes():
                raise ValueError('Derivative row bytes differ from the source')
        del back
    finally:
        del src
    h = hashlib.sha256()
    with open(dest, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------------- spanning tree

def median_spacing(X):
    """Median distance from each center to its nearest other center (d1)."""
    if len(X) < 2:
        raise ValueError('At least two centers are needed for a spacing scale')
    return float(np.median(cKDTree(X).query(X, k=2)[0][:, 1]))


def dedupe(X):
    """Merge exact duplicate rows (exact for single linkage: duplicates join at 0).
    Returns (U, inverse) with X == U[inverse]."""
    U, inv = np.unique(X, axis=0, return_inverse=True)
    return U, inv.reshape(-1)


def _complete_edges(n):
    a, b = np.triu_indices(n, 1)
    return a.astype(np.int64), b.astype(np.int64)


def candidate_edges(U, knn_fallback=8):
    """Unique undirected edges (a < b) that contain the Euclidean MST of distinct points U:
    3D Delaunay edges plus k-NN edges for any point qhull leaves out of every simplex.
    Small or degenerate (for example coplanar) inputs up to COMPLETE_GRAPH_LIMIT points use
    the complete graph instead, which is also exact; larger degenerate inputs are refused."""
    n = len(U)
    info = {'method': 'delaunay', 'points_outside_triangulation': 0}
    if n < 2:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), dict(info, method='none', unique_edges=0)
    if n <= 4:
        a, b = _complete_edges(n)
        return a, b, dict(info, method='complete_graph', unique_edges=int(len(a)))
    try:
        tri = Delaunay(U)
    except QhullError as exc:
        if n > COMPLETE_GRAPH_LIMIT:
            raise ValueError('Degenerate point set (qhull: %s) is too large for the exact '
                             'complete-graph fallback' % str(exc).splitlines()[0]) from exc
        a, b = _complete_edges(n)
        return a, b, dict(info, method='complete_graph_degenerate_input', unique_edges=int(len(a)))
    s = tri.simplices
    del tri
    keys = []
    for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        a, b = s[:, i].astype(np.int64), s[:, j].astype(np.int64)
        keys.append(np.minimum(a, b) * n + np.maximum(a, b))
    used = np.zeros(n, bool)
    used[s.ravel()] = True
    info['simplices'] = int(len(s))
    del s
    keys = np.unique(np.concatenate(keys))
    missing = np.flatnonzero(~used)
    if len(missing):
        k = min(knn_fallback + 1, n)
        _, nb = cKDTree(U).query(U[missing], k=k)
        a = np.repeat(missing, k - 1).astype(np.int64)
        b = nb[:, 1:].ravel().astype(np.int64)
        keys = np.unique(np.concatenate([keys, np.minimum(a, b) * n + np.maximum(a, b)]))
    info.update(points_outside_triangulation=int(len(missing)), unique_edges=int(len(keys)),
                edges_per_point=round(len(keys) / n, 3))
    return keys // n, keys % n, info


def build_mst(X, knn_fallback=8, meter=None, prefix=''):
    """Euclidean MST of the rows of X (rows may repeat). Returns a dict with U (distinct
    points), inv (row -> U index), E (n_u - 1, 2) int32 edges, W float64 lengths, info."""
    m = meter or StepMeter()
    with m.step(prefix + 'dedupe'):
        U, inv = dedupe(np.asarray(X, np.float64))
    with m.step(prefix + 'candidate_edges'):
        a, b, info = candidate_edges(U, knn_fallback)
    with m.step(prefix + 'mst'):
        n = len(U)
        if n < 2:
            E, W = np.zeros((0, 2), np.int32), np.zeros(0)
        else:
            w = np.linalg.norm(U[a] - U[b], axis=1)
            if not w.min() > 0:
                raise ValueError('Zero-length edge between distinct points')
            T = minimum_spanning_tree(coo_matrix((w, (a, b)), shape=(n, n)).tocsr()).tocoo()
            del a, b, w
            E = np.stack([T.row, T.col], 1).astype(np.int32)
            W = T.data.astype(np.float64)
            del T
            if len(W) != n - 1:
                raise ValueError('Candidate graph is not connected')
    info.update(rows=int(len(X)), distinct_points=int(n), duplicate_rows_merged=int(len(X) - n),
                max_edge=float(W.max()) if len(W) else 0.0)
    return {'U': U, 'inv': inv, 'E': E, 'W': W, 'info': info}


# ----------------------------------------------------------------------------- clustering

def components_at(mst, r):
    """Single-linkage components at radius r (exactly the components of the graph joining
    centers at distance <= r), as an arbitrary label per original row."""
    n = len(mst['U'])
    E = mst['E'][mst['W'] <= r]
    _, lab = connected_components(coo_matrix((np.ones(len(E), np.int8), (E[:, 0], E[:, 1])), shape=(n, n)),
                                  directed=False)
    return lab[mst['inv']]


def relabel_by_size(lab):
    """Relabel 0..K-1 by decreasing size, ties by first row. Returns (labels, sizes).
    The numbering is bookkeeping, not a ranking of which cluster is a target."""
    _, first, inv, cnt = np.unique(lab, return_index=True, return_inverse=True, return_counts=True)
    order = np.lexsort((first, -cnt))
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return remap[inv.reshape(-1)], cnt[order]


def named_clusters(lab_sized, sizes, min_size):
    """Labels with components smaller than min_size set to -1 (unclustered)."""
    out = lab_sized.copy()
    out[sizes[lab_sized] < min_size] = -1
    return out


def sweep(mst, d1, mults, min_size=500, top=8):
    """Component statistics at each radius mult * d1, from one MST."""
    rows = []
    for k in mults:
        _, sizes = relabel_by_size(components_at(mst, k * d1))
        rows.append({'mult_d1': k, 'radius': float(k * d1), 'components': int(len(sizes)),
                     'components_ge_%d' % min_size: int((sizes >= min_size).sum()),
                     'top_sizes': [int(s) for s in sizes[:top]], 'singletons': int((sizes == 1).sum())})
    return rows


def join_heights(mst, groups):
    """Exact single-linkage join radius for every pair of row groups.

    groups maps a name to row indices of the MST's input. The join radius of A and B is
    the smallest r at which some row of A and some row of B fall in one component at r,
    i.e. the shortest possible longest hop of any chain between them. Every row of a group
    is tracked, so groups need not be internally connected; overlapping groups join at 0.
    Returns ({(a, b): radius} with a < b, merge events in increasing radius)."""
    n = len(mst['U'])
    parent = list(range(n))
    size = [1] * n
    tags = {}
    for g, rows in groups.items():
        rows = np.asarray(rows, np.int64)
        if rows.size == 0:
            raise ValueError('Empty group: ' + str(g))
        for u in np.unique(mst['inv'][rows]).tolist():
            tags.setdefault(u, set()).add(g)
    heights, events = {}, []
    for s in tags.values():
        for x in s:
            for y in s:
                if x < y:
                    heights[(x, y)] = 0.0
    total = len(groups) * (len(groups) - 1) // 2

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    E, W = mst['E'], mst['W']
    for e in np.argsort(W, kind='stable').tolist():
        if len(heights) == total:
            break
        ra, rb = find(int(E[e, 0])), find(int(E[e, 1]))
        if ra == rb:
            continue
        ta, tb = tags.get(ra), tags.get(rb)
        if size[ra] < size[rb]:
            ra, rb, ta, tb = rb, ra, tb, ta
        parent[rb] = ra
        size[ra] += size[rb]
        if ta and tb:
            new = sorted({tuple(sorted((x, y))) for x in ta for y in tb if x != y} - set(heights))
            if new:
                h = float(W[e])
                for key in new:
                    heights[key] = h
                events.append({'radius': h, 'a': sorted(ta), 'b': sorted(tb),
                               'merged_size_distinct_points': size[ra]})
        if ta or tb:
            tags[ra] = (ta or set()) | (tb or set())
            tags.pop(rb, None)
    return heights, events


def direct_gap(A, B):
    """Smallest distance between point sets A and B (cKDTree on A, k = 1 queries of B).
    Returns (distance, index in A, index in B). O(|A| + |B|) memory."""
    d, j = cKDTree(A).query(B, k=1)
    i = int(np.argmin(d))
    return float(d[i]), int(j[i]), i


def pairwise_gaps(X, labels, ids):
    """Direct gap between every pair of the given cluster ids: one tree per cluster,
    queried with the points of the higher-numbered clusters. Returns {(a, b): gap}."""
    ids = sorted(int(i) for i in ids)
    ida = np.array(ids)
    out = {}
    for i, a in enumerate(ids[:-1]):
        later = np.isin(labels, ids[i + 1:])
        d = cKDTree(X[labels == a]).query(X[later], k=1)[0]
        best = np.full(len(ids), np.inf)
        np.minimum.at(best, np.searchsorted(ida, labels[later]), d)
        for j in range(i + 1, len(ids)):
            out[(a, ids[j])] = float(best[j])
    return out


# ----------------------------------------------------------------------------- guard

class PairBudgetExceeded(RuntimeError):
    pass


def estimate_pairs(tree, X, r, sample=1000, seed=0):
    """Estimated unordered distinct pairs within r: mean neighbor count of a random
    sample (excluding the point itself) times N / 2. Builds no pair list."""
    n = len(X)
    idx = np.random.default_rng(seed).choice(n, min(sample, n), replace=False)
    cnt = tree.query_ball_point(X[idx], r, return_length=True)
    return float((cnt.mean() - 1) * n / 2)


def guarded_query_pairs(tree, X, r, budget=PAIR_BUDGET, sample=1000):
    """cKDTree.query_pairs at r only if the estimated pair count is within budget.
    A pair list grows with the square of the points inside r; large radii must use the MST."""
    est = estimate_pairs(tree, X, r, sample)
    if est > budget:
        raise PairBudgetExceeded('r=%g: estimated %.3g pairs exceeds the budget of %.3g; '
                                 'use the spanning tree (join_heights) instead' % (r, est, budget))
    return tree.query_pairs(r, output_type='ndarray')


# ----------------------------------------------------------------------------- analysis

def _group_key(a, b):
    return '%s-%s' % (a, b)


def analyze(XA, kept, params, meter=None):
    """Cluster the opacity-kept centers and measure separations. Reports every named
    cluster symmetrically; selects none. Returns (report dict, arrays dict, state)."""
    m = meter or StepMeter()
    X = XA[kept]
    with m.step('median_spacing'):
        d1 = median_spacing(X)
    if not d1 > 0:
        raise ValueError('Median nearest-neighbor spacing is zero (most centers are duplicates)')
    r = params['radius_mult'] * d1
    mst = build_mst(X, params['knn_fallback'], m)
    with m.step('sweep'):
        sw = sweep(mst, d1, params['sweep_mults'], params['sweep_min_size'], params['sweep_top'])
    with m.step('clusters_at_r'):
        comp, sizes = relabel_by_size(components_at(mst, r))
        L = named_clusters(comp, sizes, params['min_cluster_size'])
        named = sorted(set(L[L >= 0].tolist()))
    with m.step('join_radii_opacity_kept'):
        jh, ev = join_heights(mst, {'c%d' % k: np.flatnonzero(L == k) for k in named})
    with m.step('direct_gaps'):
        gap_ids = named[:params['max_gap_clusters']]
        gaps = pairwise_gaps(X, L, gap_ids)
    clusters = {}
    for k in named:
        P = X[L == k]
        clusters['c%d' % k] = {'size': int(len(P)), 'centroid': P.mean(0).tolist(),
                               'bbox_min': P.min(0).tolist(), 'bbox_max': P.max(0).tolist()}
    report = {
        'units': 'scale_free_scene_units',
        'rows': int(len(XA)), 'opacity_kept_rows': int(len(kept)), 'faint_rows': int(len(XA) - len(kept)),
        'median_nn_spacing_d1': d1, 'radius': r, 'radius_mult': params['radius_mult'],
        'min_cluster_size': params['min_cluster_size'],
        'mst_opacity_kept': mst['info'], 'sweep': sw,
        'components_at_radius': int(len(sizes)), 'unclustered_centers': int((L < 0).sum()),
        'clusters': clusters,
        'cluster_numbering': 'by decreasing size at this radius; numbering is not a target choice',
        'join_radius_opacity_kept': {_group_key(*k): v for k, v in sorted(jh.items())},
        'merge_events_opacity_kept': ev,
        'direct_gap_opacity_kept': {_group_key('c%d' % a, 'c%d' % b): v for (a, b), v in sorted(gaps.items())},
        'direct_gap_clusters_included': ['c%d' % k for k in gap_ids],
    }
    arrays = {'kept_rows': kept.astype(np.int64), 'labels': L.astype(np.int32), 'components_at_radius': comp.astype(np.int32)}
    if params['all_rows']:
        mstA = build_mst(XA, params['knn_fallback'], m, 'all_rows_')
        with m.step('all_rows_join_radii'):
            jhA, evA = join_heights(mstA, {'c%d' % k: kept[L == k] for k in named})
        report['all_rows'] = {'mst': mstA['info'],
                              'join_radius': {_group_key(*k): v for k, v in sorted(jhA.items())},
                              'merge_events': evA}
        del mstA
    return report, arrays, {'X': X, 'labels': L, 'radius': r, 'd1': d1, 'mst': mst}


def expand_unclustered(X, labels, base, within):
    """Unclustered centers (label -1) not in base whose nearest base center is within
    `within`. Also counts, per named cluster outside base, its centers within that
    distance (reported, never added). Returns (added mask, {cluster: count})."""
    d = cKDTree(X[base]).query(X, k=1)[0]
    near = (d <= within) & ~base
    added = near & (labels == -1)
    others = {'c%d' % k: int(c) for k, c in zip(*np.unique(labels[near & (labels >= 0)], return_counts=True))}
    return added, others


def nearest_kept(XA, kept):
    """For every faint row (not opacity-kept): its index, the distance to the nearest
    opacity-kept center and that center's position in kept."""
    faint = np.setdiff1d(np.arange(len(XA)), kept)
    if len(faint) == 0:
        return faint, np.zeros(0), np.zeros(0, np.int64)
    d, j = cKDTree(XA[kept]).query(XA[faint], k=1)
    return faint, d, j


def attach_faint(faint, dist, nearest, selected_kept, r):
    """Faint rows attached to a selection: nearest opacity-kept center within r and selected
    (the rule of the October 2026 Welzo derivatives)."""
    return faint[(dist <= r) & selected_kept[nearest]]


def select_rows(state, kept, faint_info, XA, spec, source_rows):
    """Resolve one explicit derivative spec into source rows. spec keys: clusters (ids),
    rows (opacity-kept source rows), expand_unclustered_within (distance)."""
    L, X, r = state['labels'], state['X'], state['radius']
    named = set(L[L >= 0].tolist())
    unknown = sorted(set(spec.get('clusters', [])) - named)
    if unknown:
        raise ValueError('Unknown cluster ids for %s: %s' % (spec['name'], unknown))
    base = np.isin(L, spec.get('clusters', []))
    if source_rows is not None:
        pos = np.searchsorted(kept, source_rows)
        if np.any(pos >= len(kept)) or np.any(kept[np.minimum(pos, len(kept) - 1)] != source_rows):
            raise ValueError('Rows for %s include rows that are not opacity-kept centers' % spec['name'])
        base[pos] = True
    result = {'clusters_selected': sorted(spec.get('clusters', [])),
              'rows_supplied': 0 if source_rows is None else int(len(source_rows))}
    sel = base.copy()
    if spec.get('expand_unclustered_within') is not None:
        added, others = expand_unclustered(X, L, base, spec['expand_unclustered_within'])
        sel |= added
        result['expansion'] = {'within': spec['expand_unclustered_within'], 'unclustered_centers_added': int(added.sum()),
                               'unselected_named_cluster_centers_within': others}
    faint, dist, nearest = faint_info
    low = attach_faint(faint, dist, nearest, sel, r)
    rows = np.sort(np.concatenate([kept[sel], low]))
    result.update(opacity_kept_rows=int(sel.sum()), faint_rows_attached=int(len(low)),
                  faint_attach_radius=r, rows=int(len(rows)))
    # all-row isolation of the derivative from every other source row
    others = np.setdiff1d(np.arange(len(XA)), rows)
    if len(others):
        d = cKDTree(XA[rows]).query(XA[others], k=1)[0]
        result['isolation_all_rows'] = {'nearest_unselected_row': float(d.min()),
                                        'unselected_rows_within': {'%gr' % k: int((d <= k * r).sum()) for k in (1, 2, 4, 8)}}
    return rows, result
