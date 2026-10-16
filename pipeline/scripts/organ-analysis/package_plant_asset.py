#!/usr/bin/env python3
"""Emit a versioned plant-asset package from a checked branch-review folder plus explicit human decisions.

See README.md ("Plant asset package") for the inputs and DESIGN.md for the rules.
Sources are only read; the package is written to a fresh output directory.
"""
import argparse
import copy
import csv
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import brentq

from analyze_organs import VERSION, read, sha, source_revision, validate_evidence, write
from branches import graph_digest
import plant_asset as pa
from plant_asset import require

DECISIONS_SCHEMA = 'plant-asset-decisions'
DECISIONS_VERSION = 1
INFERENCE_METHOD = 'nearest-point-mean-reference-angle'
REJECT_CATEGORY = 'false-branch'
DEFAULT_LIMITS = ['No metric scale or radii', 'No MTG/Helios suitability claim',
                  'No anatomical validation of inferred geometry',
                  'Blade assets are Gaussian centers, not fitted surfaces',
                  'Internodes are attachment intervals; uncertain basal ordering is not validated']
ID_SCOPE = 'this exact source capture and graph; not cross-date identity'


def angle(a, b):
    return float(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))


def project_points(points, stem):
    """Closest (point, stem segment) pair; returns point row, segment, fraction and projection."""
    starts, vectors = stem[:-1], np.diff(stem, axis=0)
    length2 = np.sum(vectors ** 2, axis=1)
    t = np.clip(np.einsum('nsi,si->ns', points[:, None, :] - starts, vectors) / length2, 0, 1)
    q = starts + t[:, :, None] * vectors
    dist2 = np.sum((points[:, None, :] - q) ** 2, axis=2)
    row, seg = np.unravel_index(np.argmin(dist2), dist2.shape)
    return int(row), int(seg), float(t[row, seg]), q[row, seg]


def check_decisions(decisions, digest, endpoints, clusters, apex):
    allowed = {'schema', 'schema_version', 'plant_id', 'source', 'source_graph_sha256', 'rejected_endpoints',
               'leaves', 'cotyledon_pairs', 'pair_insertions', 'review_summary', 'additional_limits'}
    require(decisions.get('schema') == DECISIONS_SCHEMA and decisions.get('schema_version') == DECISIONS_VERSION,
            'Decisions must be ' + DECISIONS_SCHEMA + ' schema_version ' + str(DECISIONS_VERSION))
    require(not set(decisions) - allowed, 'Unknown decision keys: ' + str(sorted(set(decisions) - allowed)))
    require(decisions.get('source_graph_sha256') == digest, 'Decisions were made for a different source graph')
    plant = decisions.get('plant_id')
    require(isinstance(plant, str) and plant.strip() and ':' not in plant, 'plant_id must be a nonempty string without ":"')
    require(isinstance(decisions.get('source'), str) and decisions['source'].strip(), 'Record who made the decisions in "source"')
    for row in decisions.get('rejected_endpoints', []):
        require(set(row) == {'endpoint_id', 'reason'} and row['endpoint_id'] in endpoints and str(row['reason']).strip(),
                'Each rejected endpoint needs a known endpoint_id and a reason')
    leaves = decisions.get('leaves')
    require(isinstance(leaves, dict) and leaves, 'Decisions need a leaves object keyed by cluster ID')
    for key, decision in leaves.items():
        require(key.isdigit() and int(key) in clusters, 'Decision for unknown or empty leaf cluster ' + key)
        require(not set(decision) - {'role', 'petiole', 'attachment', 'petiole_unresolved_reason'},
                'Unknown keys in leaf decision ' + key)
        require(decision.get('role') in pa.ROLES, 'Leaf ' + key + ' role must be one of ' + ', '.join(pa.ROLES))
        if decision['role'] == 'combined-apex-region':
            require(int(key) == apex, 'Only the skeleton apex_leaf can be the combined apex region')
        petiole = decision.get('petiole')
        if petiole is None:
            require(decision['role'] == 'combined-apex-region' or str(decision.get('petiole_unresolved_reason', '')).strip(),
                    'Leaf ' + key + ' has no petiole: give petiole_unresolved_reason')
            require('attachment' not in decision, 'Leaf ' + key + ' has no petiole, so no attachment notes')
            continue
        require(decision['role'] != 'combined-apex-region', 'The combined apex region has no separate petiole in contract v1')
        source = petiole.get('source')
        if source == 'inferred':
            require(not set(petiole) - {'source', 'method', 'reference_leaf_ids', 'human_review'}, 'Unknown inferred keys ' + key)
            require(petiole.get('method') == INFERENCE_METHOD, 'Supported inference method: ' + INFERENCE_METHOD)
            refs = petiole.get('reference_leaf_ids')
            require(refs is None or (isinstance(refs, list) and len(set(refs)) == 2 and len(refs) == 2),
                    'reference_leaf_ids must name two leaves')
        else:
            require(source in ('algorithm', 'endpoint'), 'Leaf ' + key + ' petiole source must be algorithm, endpoint or inferred')
            extra = {'endpoint_id'} if source == 'endpoint' else set()
            require(not set(petiole) - {'source', 'geometry_status', 'biological_insertion_status'} - extra,
                    'Unknown petiole decision keys for leaf ' + key)
            if source == 'endpoint':
                require(petiole.get('endpoint_id') in endpoints, 'Unknown endpoint for leaf ' + key)
            status = petiole.get('geometry_status', 'extracted')
            require(status.startswith('extracted') and not pa.claims_measured(status),
                    'An extracted petiole status must start with "extracted" and never claim measurement')
        require(not set(decision.get('attachment', {})) - {'biological_insertion_status'}, 'Unknown attachment notes ' + key)
    for pair in decisions.get('cotyledon_pairs', []):
        require(len(pair) == 2 and all(leaves.get(str(c), {}).get('role') == 'cotyledon' for c in pair),
                'cotyledon_pairs must name decided cotyledons')


def read_endpoint_review(path, digest, capture, endpoints, positions):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    ids = [row['endpoint_id'] for row in rows]
    require(rows and len(ids) == len(set(ids)), 'Endpoint review CSV needs unique endpoint rows')
    for row in rows:
        require(row['source_graph_sha256'] == digest and row['capture_id'] == capture,
                'Endpoint review belongs to a different graph or capture')
        node = endpoints.get(row['endpoint_id'])
        require(node is not None and int(row['source_node_id']) == node, 'Endpoint ID/node mismatch: ' + row['endpoint_id'])
        require(positions[node] == [float(row[k]) for k in 'xyz'], 'Endpoint coordinate mismatch: ' + row['endpoint_id'])
    return {row['endpoint_id'] for row in rows if row['human_category'] == REJECT_CATEGORY}


def build(review, decisions_path, endpoint_review=None):
    """Return the in-memory package plan; nothing is written."""
    review = Path(review).resolve()
    bundle = read(review / 'organ-evidence.json')
    validate_evidence(bundle, review)
    graph = bundle['source_graph']
    require(graph, 'Review folder has no skeleton graph')
    digest = graph_digest(graph)
    report = read(review / 'branch-candidates.json')
    require(report['source_graph_sha256'] == digest, 'branch-candidates.json belongs to a different graph')
    endpoints = {e['endpoint_id']: e['source_node_id'] for e in report['endpoint_records']}
    node_endpoint = {node: eid for eid, node in endpoints.items()}
    labels = pa.load_npz(review / 'leaf_ids2.npz')
    X, label = labels['X'], labels['label']
    clusters = {int(c) for c in np.unique(label) if c >= 0}
    decisions = read(decisions_path)
    apex = graph['apex_leaf']
    check_decisions(decisions, digest, endpoints, clusters, apex)
    plant = decisions['plant_id']
    decided = {int(k): v for k, v in decisions['leaves'].items()}
    raw = {leaf['leaf_id']: leaf for leaf in graph['leaves']}
    require(not set(raw) - set(decided), 'Every skeleton leaf record needs a decision: ' + str(sorted(set(raw) - set(decided))))

    positions = {n['id']: n['xyz'] for n in graph['skeleton_nodes']}
    adjacency = {n: set() for n in positions}
    for e in graph['skeleton_edges']:
        adjacency[e['a']].add(e['b'])
        adjacency[e['b']].add(e['a'])
    main_ids = []
    for xyz in graph['main_stem']:
        match = [n for n, p in positions.items() if p == xyz]
        require(len(match) == 1, 'Main-stem coordinate must match exactly one graph node')
        main_ids.append(match[0])
    root = main_ids[0]
    parent, stack = {root: None}, [root]
    while stack:
        n = stack.pop()
        for other in adjacency[n]:
            if other not in parent:
                parent[other] = n
                stack.append(other)
    require(len(parent) == len(positions) == len(graph['skeleton_edges']) + 1, 'Source graph must be one tree')

    rejected = {r['endpoint_id']: r['reason'] for r in decisions.get('rejected_endpoints', [])}
    if endpoint_review is not None:
        for eid in read_endpoint_review(endpoint_review, digest, bundle['capture_id'], endpoints, positions):
            rejected.setdefault(eid, 'endpoint review CSV: ' + REJECT_CATEGORY)
    rejected_nodes = {endpoints[e] for e in rejected}
    # Retain the union of root paths to every endpoint not explicitly rejected,
    # so a rejected tip removes only edges no retained endpoint depends on.
    keep = {root}
    for node in endpoints.values():
        if node in rejected_nodes:
            continue
        while node is not None and node not in keep:
            keep.add(node)
            node = parent[node]
    require(set(main_ids) <= keep, 'A rejected endpoint would cut the main axis')
    reviewed = copy.deepcopy(graph)
    reviewed.pop('petiole_points', None)  # algorithm diagnostic, not recomputed after review
    reviewed['skeleton_nodes'] = [n for n in graph['skeleton_nodes'] if n['id'] in keep]
    reviewed['skeleton_edges'] = [e for e in graph['skeleton_edges'] if e['a'] in keep and e['b'] in keep]
    degree = {n: len(adjacency[n] & keep) for n in keep}
    main_set = set(main_ids)
    terminals = sorted(n for n in keep if degree[n] == 1 and n not in main_set)

    nodes = {n: np.array(positions[n]) for n in keep}
    stem = np.array(graph['main_stem'])
    lengths = np.linalg.norm(np.diff(stem, axis=0), axis=1)
    arcs = np.r_[0, np.cumsum(lengths)]
    tangents = np.array([stem[min(i + 1, len(stem) - 1)] - stem[max(0, i - 1)] for i in range(len(stem))])
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]

    def to_axis(node):
        path = [node]
        while path[-1] not in main_set:
            path.append(parent[path[-1]])
        return path[::-1]

    def nearest_leaf_point(cluster, xyz):
        points = X[label == cluster]
        i = int(np.argmin(np.linalg.norm(points - xyz, axis=1)))
        return points[i], float(np.linalg.norm(points[i] - xyz))

    def attachment_on_node(cluster, node):
        idx = main_ids.index(node)
        seg, fraction = (idx - 1, 1.0) if idx == len(main_ids) - 1 else (idx, 0.0)
        return {'id': plant + ':attachment:' + str(cluster), 'leaf_cluster_id': cluster, 'xyz': stem[idx].tolist(),
                'arc_length': float(arcs[idx]), 'stem_segment_index': seg, 'segment_fraction': fraction,
                'geometry_status': 'extracted-candidate', 'source_node_id': node}

    attachments, petioles, known, graph_records = {}, {}, [], {}
    for cluster, decision in sorted(decided.items()):
        spec = decision.get('petiole')
        if spec is None or spec['source'] == 'inferred':
            continue
        if spec['source'] == 'algorithm':
            row = raw.get(cluster)
            require(row is not None and row['base_rule'] == 'tip' and row['petiole_network_length'] > 0,
                    'Leaf ' + str(cluster) + ' has no extracted algorithm path; decide endpoint, inferred or unresolved')
            base = np.array(row['base_xyz'])
            end = min(terminals, key=lambda n: (float(np.linalg.norm(nodes[n] - base)), n))
            require(math.isclose(float(np.linalg.norm(nodes[end] - base)), row['gap'], rel_tol=1e-6, abs_tol=1e-9),
                    'Leaf ' + str(cluster) + ' algorithm tip is no longer in the reviewed graph')
            path = to_axis(end)
            require(positions[path[0]] == row['insertion_xyz'], 'Leaf ' + str(cluster) + ' insertion disagrees with the graph')
            status = spec.get('geometry_status', 'extracted-candidate')
        else:
            end = endpoints[spec['endpoint_id']]
            require(end in keep and end in terminals, 'Endpoint ' + spec['endpoint_id'] + ' was rejected or is not a side tip')
            path = to_axis(end)
            status = spec.get('geometry_status', 'extracted; human-selected endpoint')
        require(len(path) >= 2, 'Leaf ' + str(cluster) + ' petiole path has no length')
        base_point, gap = nearest_leaf_point(cluster, nodes[end])
        attachment = attachment_on_node(cluster, path[0])
        attachment.update(decision.get('attachment', {}))
        attachments[cluster] = attachment
        petiole = {'id': plant + ':petiole:' + str(cluster), 'leaf_id': plant + ':leaf:' + str(cluster),
                   'attachment_id': attachment['id'], 'polyline': [positions[n] for n in path], 'source_node_ids': path,
                   'classification': pa.EXTRACTED, 'geometry_status': status, 'radius': None, 'inference': None,
                   'source_endpoint_id': node_endpoint.get(end), 'source_endpoint_node_id': end,
                   'blade_to_path_gap_scene_units': gap}
        if 'biological_insertion_status' in spec:
            petiole['biological_insertion_status'] = spec['biological_insertion_status']
        petioles[cluster] = petiole
        idx = main_ids.index(path[0])
        known.append({'leaf': cluster, 'arc': float(arcs[idx]), 'path': path,
                      'angle': angle(nodes[path[1]] - nodes[path[0]], tangents[idx])})
        if spec['source'] == 'endpoint':
            graph_records[cluster] = {
                'leaf_id': cluster, 'insertion_along_stem': float(arcs[idx]), 'insertion_xyz': positions[path[0]],
                'petiole_network_length': float(sum(np.linalg.norm(nodes[a] - nodes[b]) for a, b in zip(path, path[1:]))),
                'base_xyz': base_point.tolist(), 'gap': gap, 'base_rule': 'human_selected_existing_endpoint',
                'source_endpoint_id': spec['endpoint_id'], 'source_node_id': end, 'path_node_ids': path,
                'review_status': 'human_selected_existing_geometry', 'units': pa.UNITS}

    for cluster, decision in sorted(decided.items()):
        spec = decision.get('petiole')
        if spec is None or spec['source'] != 'inferred':
            continue
        points = X[label == cluster]
        row, seg, fraction, _ = project_points(points, stem)
        anchor = points[row]
        nearest_arc = float(arcs[seg] + fraction * lengths[seg])
        pool = known
        if spec.get('reference_leaf_ids') is not None:
            pool = [k for k in known if k['leaf'] in spec['reference_leaf_ids']]
            require(len(pool) == 2, 'Inference references must be two leaves with extracted petioles')
        refs = sorted(pool, key=lambda k: (abs(k['arc'] - nearest_arc), k['leaf']))[:2]
        require(len(refs) == 2, 'Inference needs two extracted reference petioles')
        theta = float(np.mean([k['angle'] for k in refs]))
        require(1e-6 < theta < math.pi - 1e-6, 'Degenerate reference angle')
        candidates = []
        for j, length in enumerate(lengths):
            def residual(t):
                q = stem[j] + t * (stem[j + 1] - stem[j])
                return angle(anchor - q, (1 - t) * tangents[j] + t * tangents[j + 1]) - theta
            grid = np.linspace(0, 1, 33)
            roots = []
            for low, high in zip(grid, grid[1:]):
                a, b = residual(low), residual(high)
                if abs(a) < 1e-10:
                    roots.append(float(low))
                if a * b < 0:
                    roots.append(float(brentq(residual, low, high, xtol=1e-14)))
                if abs(b) < 1e-10:
                    roots.append(float(high))
            for t in roots:
                q = stem[j] + t * (stem[j + 1] - stem[j])
                achieved = angle(anchor - q, (1 - t) * tangents[j] + t * tangents[j + 1])
                candidates.append((abs(achieved - theta), abs(float(arcs[j] + t * length) - nearest_arc), j, t, q,
                                   float(arcs[j] + t * length), achieved))
        # Exact-angle solutions only; nearest along the stem. Measured stem points never move.
        exact = [c for c in candidates if c[0] < 1e-9]
        require(exact, 'No exact stem solution for inferred petiole of leaf ' + str(cluster))
        _, shift, j, t, q, arc, achieved = min(exact, key=lambda c: (c[1], c[2]))
        source_row = int(np.flatnonzero(label == cluster)[row])
        attachments[cluster] = {'id': plant + ':attachment:' + str(cluster), 'leaf_cluster_id': cluster, 'xyz': q.tolist(),
                                'arc_length': arc, 'stem_segment_index': j, 'segment_fraction': t,
                                'geometry_status': 'inferred', 'source_node_id': None}
        inference = {'method': 'nearest leaf point to stem; mean of two nearest existing petiole initial-segment angles; root-solved attachment on existing stem',
                     'method_id': INFERENCE_METHOD,
                     'nearest_defined_by': 'along-stem distance from nearest leaf-point projection',
                     'angle_convention': 'directed root-to-apex; normalized centered tangents at stem nodes interpolated continuously along segments',
                     'reference_leaf_ids': [k['leaf'] for k in refs],
                     'reference_selection': 'human-specified' if spec.get('reference_leaf_ids') is not None else 'two nearest extracted petioles',
                     'reference_angles_degrees': [math.degrees(k['angle']) for k in refs],
                     'mean_reference_angle_degrees': math.degrees(theta), 'achieved_angle_degrees': math.degrees(achieved),
                     'nearest_projection_arc': nearest_arc, 'attachment_shift_scene_units': shift,
                     'selected_source_point_row_id': source_row, 'confidence': 'heuristic; no measured accuracy'}
        if 'human_review' in spec:
            inference['human_review'] = spec['human_review']
        petioles[cluster] = {'id': plant + ':petiole:' + str(cluster), 'leaf_id': plant + ':leaf:' + str(cluster),
                             'attachment_id': attachments[cluster]['id'], 'polyline': [q.tolist(), anchor.tolist()],
                             'source_node_ids': None, 'classification': pa.INFERRED, 'geometry_status': 'inferred',
                             'radius': None, 'inference': inference}

    leaves_out = []
    for cluster, decision in sorted(decided.items()):
        leaf = {'id': plant + ':leaf:' + str(cluster), 'source_cluster_id': cluster, 'role': decision['role'],
                'petiole_id': petioles[cluster]['id'] if cluster in petioles else None,
                'point_geometry': {'path': 'leaves/leaf-' + str(cluster) + '.npz', 'sha256': None,
                                   'point_count': int((label == cluster).sum()), 'status': 'extracted Gaussian centers'},
                'surface_mesh': None}
        if decision.get('petiole_unresolved_reason'):
            leaf['petiole_unresolved_reason'] = decision['petiole_unresolved_reason']
        leaves_out.append(leaf)
    leaf_records = []
    for row in graph['leaves']:
        spec = decided[row['leaf_id']].get('petiole')
        if row['leaf_id'] in graph_records:
            leaf_records.append(graph_records.pop(row['leaf_id']))
        elif (spec and spec['source'] == 'algorithm') or (spec is None and decided[row['leaf_id']]['role'] == 'combined-apex-region'):
            leaf_records.append(row)
    reviewed['leaves'] = leaf_records + [graph_records[c] for c in sorted(graph_records)]

    ordered = sorted(attachments.values(), key=lambda n: (n['arc_length'], n['leaf_cluster_id']))
    internodes = []
    for i, (a, b) in enumerate(zip(ordered, ordered[1:])):
        inner = [stem[k].tolist() for k in range(len(stem)) if a['arc_length'] < arcs[k] < b['arc_length']]
        statuses = [a['geometry_status'], b['geometry_status']]
        internodes.append({'id': plant + ':internode:' + str(i), 'start_attachment_id': a['id'], 'end_attachment_id': b['id'],
                           'polyline': [a['xyz']] + inner + [b['xyz']], 'length_scene_units': b['arc_length'] - a['arc_length'],
                           'geometry_status': 'unresolved zero sampling interval; not measured biological internode'
                           if a['arc_length'] == b['arc_length'] else 'derived interval; inferred boundary'
                           if 'inferred' in statuses else 'derived from extracted geometry', 'radius': None})
    owners = {pa_edge(e['a'], e['b']): [] for e in reviewed['skeleton_edges']}
    for a, b in zip(main_ids, main_ids[1:]):
        owners[pa_edge(a, b)].append('main-axis')
    for cluster in sorted(petioles):
        ids = petioles[cluster]['source_node_ids'] or []
        for a, b in zip(ids, ids[1:]):
            owners[pa_edge(a, b)].append(petioles[cluster]['leaf_id'])
    require(all(len(who) <= 1 for who in owners.values()), 'An extracted edge is assigned to more than one organ')
    edge_counts = {'main_axis': sum(who == ['main-axis'] for who in owners.values()),
                   'extracted_petiole': sum(bool(who) and who != ['main-axis'] for who in owners.values()),
                   'unassigned': sum(not who for who in owners.values())}
    asset = {'schema': pa.SCHEMA, 'schema_version': pa.SCHEMA_VERSION, 'plant_id': plant, 'capture_id': bundle['capture_id'],
             'id_scope': ID_SCOPE, 'source_graph_sha256': digest, 'units': pa.UNITS, 'metric_scale': None,
             'generator': {'name': 'package_plant_asset.py', 'version': VERSION},
             'point_labels': {'path': 'source-evidence/leaf_ids2.npz', 'sha256': None, 'point_count': int(len(X)),
                              'row_id_scope': pa.ROW_ID_SCOPE},
             'reviewed_graph': {'path': 'skeleton-extracted-reviewed.json', 'sha256': None,
                                'note': 'Retained extracted graph after human endpoint review. Edges are main axis, one extracted petiole, or unassigned. Biological insertion positions remain provisional.',
                                'edge_classification': edge_counts},
             'main_axis': {'id': plant + ':axis:main', 'polyline': stem.tolist(), 'source_node_ids': main_ids,
                           'geometry_status': 'algorithm-extracted; human-reviewed' if rejected else 'algorithm-extracted',
                           'radius': None},
             'attachment_nodes': [attachments[c] for c in sorted(attachments)], 'internodes': internodes,
             'petioles': [petioles[c] for c in sorted(petioles)], 'leaves': leaves_out,
             'ordered_attachment_ids': [n['id'] for n in ordered],
             'cotyledon_pairs': decisions.get('cotyledon_pairs', []), 'pair_insertions': decisions.get('pair_insertions'),
             'source_files': {}, 'source_evidence': [], 'limits': DEFAULT_LIMITS + decisions.get('additional_limits', []),
             'review_update': decisions.get('review_summary')}
    edge_rows = [(key, who[0] if who else 'unassigned') for key, who in owners.items()]
    return {'asset': asset, 'graph': reviewed, 'bundle': bundle, 'labels': labels, 'rejected': rejected,
            'edge_rows': edge_rows, 'decisions': decisions}


def pa_edge(a, b):
    return '-'.join(map(str, sorted([a, b])))


def viewer_scene(plan):
    """Display copy of the evidence: reviewed graph plus visibly separate inferred connections."""
    asset, scene = plan['asset'], copy.deepcopy(plan['bundle'])
    scene['source_graph'] = copy.deepcopy(plan['graph'])
    by_cluster = {int(p['leaf_id'].rsplit(':', 1)[1]): p for p in asset['petioles']}
    for cluster, petiole in sorted(by_cluster.items()):
        if petiole['inference'] is None:
            continue
        start, end = 'inferred-' + str(cluster) + '-stem', 'inferred-' + str(cluster) + '-blade'
        scene['source_graph']['skeleton_nodes'] += [{'id': start, 'xyz': petiole['polyline'][0]}, {'id': end, 'xyz': petiole['polyline'][-1]}]
        scene['source_graph']['skeleton_edges'].append({'a': start, 'b': end, 'main': False, 'classification': pa.INFERRED})
        scene['source_graph']['leaves'].append({'leaf_id': cluster, 'insertion_xyz': petiole['polyline'][0], 'classification': pa.INFERRED})
    for leaf in scene['leaves']:
        petiole = by_cluster.get(leaf['id'])
        if petiole is None:
            continue
        if petiole['inference'] is not None:
            leaf['geometry_status'] = pa.INFERRED
            leaf['annotation'] = {'classification': pa.INFERRED, 'not_measured': True}
        elif plan['decisions']['leaves'][str(leaf['id'])]['petiole']['source'] == 'endpoint':
            leaf['geometry_status'] = 'human_reviewed_existing_route'
            leaf['annotation'] = {'petiole': 'existing ' + petiole['source_endpoint_id'] + ' / node ' + str(petiole['source_endpoint_node_id'])}
    return scene


def package(review, decisions_path, output, endpoint_review=None, evidence=()):
    from viewer import write_viewer
    review, output = Path(review).resolve(), Path(output).resolve()
    sources = [review / 'leaf_ids2.npz', review / 'organ-evidence.json', review / 'branch-candidates.json',
               Path(decisions_path).resolve()] + ([Path(endpoint_review).resolve()] if endpoint_review else [])
    sources += [Path(p).resolve() for p in evidence]
    require(not output.exists(), 'Output already exists; choose a fresh directory: ' + str(output))
    require(review != output and review not in output.parents, 'Write the package outside the review folder')
    names = ['leaf_ids2.npz', 'organ-evidence-original.json', 'branch-candidates.json'] + [s.name for s in sources[3:]]
    require(len(names) == len(set(names)), 'Evidence file names must be unique')
    before = {str(s): sha(s) for s in sources}
    plan = build(review, decisions_path, endpoint_review)
    asset, folder = plan['asset'], output / 'plant-asset'
    record = {'version': VERSION, 'status': 'running', 'command': sys.argv, 'repository': source_revision(),
              'started_utc': datetime.now(timezone.utc).isoformat(), 'input_sha256': before,
              'code_sha256': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}}
    (folder / 'leaves').mkdir(parents=True)
    (folder / 'source-evidence').mkdir()
    try:
        for source, name in zip(sources, names):
            shutil.copyfile(source, folder / 'source-evidence' / name)
            asset['source_evidence'].append({'path': 'source-evidence/' + name, 'sha256': sha(folder / 'source-evidence' / name)})
        asset['source_files'] = before
        asset['point_labels']['sha256'] = sha(folder / asset['point_labels']['path'])
        labels = plan['labels']
        for leaf in asset['leaves']:
            rows = np.flatnonzero(labels['label'] == leaf['source_cluster_id'])
            data = {name: values[rows] for name, values in labels.items() if values.shape[:1] == labels['X'].shape[:1]}
            data['source_point_row_ids'] = rows
            pa.save_npz(folder / leaf['point_geometry']['path'], data)
            leaf['point_geometry']['sha256'] = sha(folder / leaf['point_geometry']['path'])
        (folder / asset['reviewed_graph']['path']).write_text(json.dumps(plan['graph'], indent=2, allow_nan=False) + '\n')
        asset['reviewed_graph']['sha256'] = sha(folder / asset['reviewed_graph']['path'])
        with (folder / 'plant.json').open('x') as stream:
            stream.write(json.dumps(asset, indent=2, allow_nan=False) + '\n')
        summary = pa.validate_asset(folder)
        require(read(folder / 'plant.json') == asset, 'plant.json round trip failed')
        with tempfile.TemporaryDirectory() as temporary:
            moved = Path(temporary) / 'copied-asset'
            shutil.copytree(folder, moved)
            pa.validate_asset(moved)
            for leaf in asset['leaves']:
                document = pa.export_leaf(moved, leaf['id'], Path(temporary) / ('leaf-' + str(leaf['source_cluster_id'])))
                require(document['petiole'] == next((p for p in asset['petioles'] if p['id'] == leaf['petiole_id']), None),
                        'Relocated leaf export changed its petiole')
        plant_sha = sha(folder / 'plant.json')
        exports = {str(leaf['source_cluster_id']): pa.leaf_export_document(folder, leaf['id'], asset, inline=True)
                   for leaf in asset['leaves']}
        write_viewer(output, viewer_scene(plan), labels_path=folder / asset['point_labels']['path'],
                     plant_asset={'plant_id': asset['plant_id'], 'capture_id': asset['capture_id'],
                                  'schema': asset['schema'], 'schema_version': asset['schema_version'],
                                  'plant_json_sha256': plant_sha, 'exports': exports})
        with (output / 'edge-classifications.csv').open('x', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['source_edge_id', 'classification'])
            writer.writerows(plan['edge_rows'])
        nonapex = [leaf for leaf in asset['leaves'] if leaf['role'] != 'combined-apex-region']
        checks = {'schema': asset['schema'], 'schema_version': asset['schema_version'], 'validation': summary,
                  'nonapex_leaf_ids': [leaf['source_cluster_id'] for leaf in nonapex],
                  'nonapex_leaves_without_petiole': [leaf['source_cluster_id'] for leaf in nonapex if leaf['petiole_id'] is None],
                  'rejected_endpoints': plan['rejected'], 'exact_json_round_trip': True,
                  'relocated_asset_valid': True, 'relocated_leaf_exports_reload': len(asset['leaves']),
                  'sources_unchanged': all(sha(Path(p)) == h for p, h in before.items())}
        require(checks['sources_unchanged'], 'A source file changed while packaging')
        write(output / 'checks.json', checks)
        write(output / 'plant-asset-manifest.json',
              {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob('*')) if p.is_file()})
        record['status'] = 'complete'
        record['output_sha256'] = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*'))
                                   if p.is_file() and p.name != 'package-record.json'}
    except Exception as exc:
        record.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        record['ended_utc'] = datetime.now(timezone.utc).isoformat()
        write(output / 'package-record.json', record)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, required=True,
                        help='Branch-review folder with organ-evidence.json, leaf_ids2.npz and branch-candidates.json')
    parser.add_argument('--decisions', type=Path, required=True, help='plant-asset-decisions JSON; see README.md')
    parser.add_argument('--endpoint-review', type=Path, help='Endpoint review CSV downloaded from the review viewer')
    parser.add_argument('--evidence', type=Path, action='append', default=[], help='Extra review record copied with its hash')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; existing paths are refused')
    parser.add_argument('--validate-only', action='store_true', help='Check inputs and decisions without writing')
    args = parser.parse_args(argv)
    if args.validate_only:
        plan = build(args.review, args.decisions, args.endpoint_review)
        print(json.dumps({'status': 'validated', 'leaves': len(plan['asset']['leaves']),
                          'petioles': len(plan['asset']['petioles']), 'rejected_endpoints': sorted(plan['rejected'])}))
        return 0
    summary = package(args.review, args.decisions, args.output, args.endpoint_review, args.evidence)
    print(json.dumps(summary))
    print('Complete: ' + str(args.output.resolve() / 'plant-asset' / 'plant.json'))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print('package-plant-asset: ' + str(exc), file=sys.stderr)
        sys.exit(2)
