"""Conservative graph candidate agreement, not a biological stem detector."""
import json
from pathlib import Path

import networkx as nx
import numpy as np


def orientation(spec):
    """Validate a completed orientation and its current, hashed source frame."""
    from analyze_organs import read, sha
    path = Path(spec['orientation'])
    for key in ['images', 'plant_masks', 'leaf_instances']:
        source = Path(spec['inputs'][key])
        if source == path or source in path.parents:
            raise ValueError('Orientation must be outside protected source directories')
    artifact = read(path)
    record = read(path.parent / 'run-record.json')
    if (artifact.get('schema') != 'plant-orientation' or type(artifact.get('schema_version')) is not int
            or artifact['schema_version'] != 1 or artifact.get('status') != 'complete'
            or record.get('status') != 'complete'):
        raise ValueError('Orientation requires a completed plant-orientation schema version 1 artifact')
    if artifact.get('capture_id') != spec['capture_id'] or record.get('spec', {}).get('capture_id') != spec['capture_id']:
        raise ValueError('Orientation capture_id differs')
    if record.get('output_sha256', {}).get(path.name) != sha(path):
        raise ValueError('Orientation artifact changed since completion')
    hashes = artifact.get('input_sha256', {})
    if not hashes or hashes != record.get('input_sha256'):
        raise ValueError('Orientation source hashes differ from completed record')
    for p, digest in hashes.items():
        if sha(p) != digest:
            raise ValueError('Orientation source changed: ' + p)
    prior = record.get('spec', {})
    required = {k: prior.get('inputs', {}).get(k) for k in ['transforms', 'dataparser_transforms']}
    if prior.get('soil'):
        required['frame_map'] = prior['soil'].get('frame_map')
    for key, p in required.items():
        if p is None or hashes.get(p) != sha(spec['inputs'][key]):
            raise ValueError('Orientation coordinate-frame hash differs: ' + key)
    try:
        up = np.asarray(artifact['up']['up'], dtype=float)
        if up.shape != (3,) or not np.isfinite(up).all() or abs(np.linalg.norm(up) - 1) > 1e-6:
            raise ValueError('Orientation up must be a finite unit vector')
        choice = artifact['up']
        plane = None
        if choice.get('method') == 'soil_plane' and choice.get('status') == 'primary':
            soil = artifact['up_soil_plane']
            normal = np.asarray(soil['normal'], dtype=float)
            center = np.asarray(soil['point'], dtype=float)
            if (soil.get('status') != 'ok' or normal.shape != (3,) or center.shape != (3,)
                    or not np.isfinite(normal).all() or not np.isfinite(center).all()
                    or not np.allclose(normal, up, rtol=0, atol=1e-6)):
                raise ValueError('Orientation soil plane is invalid')
            plane = {'normal': normal.tolist(), 'center': center.tolist()}
        elif not (spec.get('allow_camera_fallback', False) and choice.get('method') == 'camera_mean_image_up'
                  and choice.get('status') == 'fallback'):
            raise ValueError('Morphology requires soil-primary orientation; camera fallback must be explicitly allowed')
    except (KeyError, TypeError) as exc:
        raise ValueError('Malformed orientation up or soil plane') from exc
    # Include the completed record too: both are immutable inputs to this decision.
    return {'up': up.tolist(), 'soil_plane': plane, 'method': choice['method'],
            'status': choice['status'], 'artifact_sha256': sha(path)}, dict(hashes, **{
                str(path): sha(path), str(path.parent / 'run-record.json'): sha(path.parent / 'run-record.json')})


def candidates(output, X, labels, Q, idxQ, root, nodes, tree, rootn, info, up, orientation, override, bin_width):
    """Require three unique extrema to agree. Ties are ambiguity, never ID order."""
    supported = {r['leaf_id'] for r in json.loads((Path(output) / 'leaf_ids2.json').read_text())['clusters']}
    scale = float(np.linalg.norm(np.ptp(Q, axis=0)))
    tolerance = max(float(bin_width), scale * 1e-9)
    distances = nx.single_source_dijkstra_path_length(tree, rootn, weight='weight')
    leaves = []
    for leaf in sorted(supported):
        d = info.get(leaf)
        points = X[labels == leaf]
        leaves.append({'leaf_id': int(leaf),
                       'median_height': float(np.median(points @ up)) if len(points) else None,
                       'endpoint': int(d['node']) if d else None,
                       'endpoint_is_terminal': bool(d and tree.degree[d['node']] == 1 and d['node'] != rootn),
                       'endpoint_geodesic': float(distances[d['node']]) if d else None,
                       'association_status': 'available' if d else 'unavailable_derived_leaf_info'})
    terminals = [{'endpoint': int(n), 'height': float(nodes[n]['c'] @ up),
                  'geodesic': float(distances[n])} for n in tree if tree.degree[n] == 1 and n != rootn]

    def rank(rows, value, rank_key):
        valid = [r for r in rows if r[value] is not None]
        for row in rows:
            row[rank_key] = None if row[value] is None else 1 + sum(other[value] > row[value] + tolerance for other in valid)
        return [r for r in rows if r[rank_key] == 1]

    height = rank(leaves, 'median_height', 'median_height_rank')
    farthest = rank(leaves, 'endpoint_geodesic', 'endpoint_geodesic_rank')
    highest = rank(terminals, 'height', 'terminal_height_rank')
    agree = (len(height) == len(farthest) == len(highest) == 1
             and height[0]['endpoint'] == farthest[0]['endpoint'] == highest[0]['endpoint']
             and height[0]['endpoint_is_terminal'])
    selected = override if override is not None else (height[0]['leaf_id'] if agree else None)
    base = Q[root]
    plane = orientation['soil_plane']
    report = {'schema': 'apex-candidates-v1', 'orientation': orientation,
              'tolerance': {'absolute': tolerance, 'skeleton_bin_width': float(bin_width),
                            'rule': 'maximum of skeleton BIN and 1e-9 times retained network bounding-box diagonal'},
              'base': {'method': 'lowest retained network point along orientation up',
                       'interpretation': 'candidate network root; anatomical stem base unvalidated',
                       'point_label_array_row': int(idxQ[root]), 'xyz': base.tolist(),
                       'height': float(base @ up),
                       'signed_plane_distance': None if plane is None else float((base - plane['center']) @ up)},
              'supported_leaves': leaves, 'terminals': terminals,
              'extrema': {'highest_median_leaves': [r['leaf_id'] for r in height],
                          'farthest_associated_leaves': [r['leaf_id'] for r in farthest],
                          'highest_terminals': [r['endpoint'] for r in highest]},
              'graph': {'root_sample': rootn,
                        'samples': [{'id': int(n), 'xyz': nodes[n]['c'].tolist()} for n in tree],
                        'edges': [{'a': int(a), 'b': int(b), 'length': float(d['weight'])}
                                  for a, b, d in tree.edges(data=True)]},
              'consensus': bool(agree), 'apex_leaf': selected,
              'decision_basis': 'explicit_apex_leaf_override' if override is not None else 'three_unique_extrema_agree' if agree else 'ambiguous_refused',
              'limits': ['Graph endpoints are geometric samples, not biological nodes',
                         'Leaf association and graph can include strings, tendrils and missing connections',
                         'Height-slice medians are not a traced stem']}
    with (Path(output) / 'apex-candidates.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    if selected is None:
        raise ValueError('Morphology apex ambiguous: candidate extrema disagree or tie; see apex-candidates.json; provide apex_leaf override')
    return selected, report
