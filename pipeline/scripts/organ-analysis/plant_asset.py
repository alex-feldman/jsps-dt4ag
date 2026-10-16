"""Versioned, capture-scoped plant-asset contract: validation and single-leaf export.

The contract is described in README.md ("Plant asset package") and DESIGN.md.
Nothing in this module edits an existing asset; exports are written to fresh
directories and the source asset is only read.
"""
import copy
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import zipfile

import numpy as np

from analyze_organs import read, sha

SCHEMA = 'plant-asset'
SCHEMA_VERSION = 1
LEAF_EXPORT_SCHEMA = 'plant-asset-leaf-export'
LEAF_EXPORT_VERSION = 1
UNITS = 'scale_free_scene_units'
ROLES = ('true-leaf', 'cotyledon', 'combined-apex-region')
ROW_ID_SCOPE = 'row of the opacity-filtered analysis array named by point_labels; not a raw PLY vertex index'
TOP_LEVEL = ['schema', 'schema_version', 'plant_id', 'capture_id', 'id_scope', 'source_graph_sha256', 'units',
             'metric_scale', 'generator', 'point_labels', 'reviewed_graph', 'main_axis', 'attachment_nodes',
             'internodes', 'petioles', 'leaves', 'ordered_attachment_ids', 'cotyledon_pairs', 'pair_insertions',
             'source_files', 'source_evidence', 'limits', 'review_update']
LEAF_KEYS = ['id', 'source_cluster_id', 'role', 'petiole_id', 'point_geometry', 'surface_mesh']
PETIOLE_KEYS = ['id', 'leaf_id', 'attachment_id', 'polyline', 'source_node_ids', 'classification',
                'geometry_status', 'radius', 'inference']
ATTACHMENT_KEYS = ['id', 'leaf_cluster_id', 'xyz', 'arc_length', 'stem_segment_index', 'segment_fraction',
                   'geometry_status', 'source_node_id']
INTERNODE_KEYS = ['id', 'start_attachment_id', 'end_attachment_id', 'polyline', 'length_scene_units',
                  'geometry_status', 'radius']
INFERENCE_KEYS = ['method', 'reference_leaf_ids', 'mean_reference_angle_degrees', 'achieved_angle_degrees',
                  'selected_source_point_row_id', 'confidence']
EXTRACTED = 'extracted-petiole-candidate'
INFERRED = 'inferred-petiole'
TOL = 1e-12


class PlantAssetError(ValueError):
    """The asset or leaf export does not satisfy the contract."""


def require(condition, message):
    if not condition:
        raise PlantAssetError(message)


def claims_measured(status):
    """True when a status text asserts measurement ("not measured" is a disclaimer, not a claim)."""
    return re.search(r'(?<!not )\bmeasured\b', str(status).lower()) is not None


def inside(folder, relative):
    """Resolve a contract path; only plain relative POSIX paths inside the folder are allowed."""
    require(isinstance(relative, str) and relative and '\\' not in relative, 'Bad relative path: ' + repr(relative))
    pure = PurePosixPath(relative)
    require(not pure.is_absolute() and '..' not in pure.parts and ':' not in pure.parts[0],
            'Path must stay inside the asset folder: ' + relative)
    folder = Path(folder).resolve()
    full = folder.joinpath(*pure.parts).resolve()
    require(folder in full.parents, 'Path escapes the asset folder: ' + relative)
    return full


def save_npz(path, arrays):
    """Deterministic compressed NPZ: fixed entry order and timestamps, refuses overwrite."""
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(value), allow_pickle=False)
            info = zipfile.ZipInfo(name + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, buffer.getvalue())


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def stem_arcs(stem):
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(stem, axis=0), axis=1))]


def finite_polyline(value, name, minimum=2):
    array = np.asarray(value, dtype=float)
    require(array.ndim == 2 and array.shape[1] == 3 and len(array) >= minimum and np.isfinite(array).all(),
            name + ' must be a finite polyline of xyz points')
    return array


def load_asset(folder):
    return read(Path(folder) / 'plant.json')


def by_id(rows, kind):
    result = {}
    for row in rows:
        require(row['id'] not in result, 'Duplicate ' + kind + ' ID ' + str(row['id']))
        result[row['id']] = row
    return result


def validate_asset(folder, asset=None):
    """Check every contract rule against the files in ``folder``; return a summary or raise PlantAssetError."""
    folder = Path(folder).resolve()
    if asset is None:
        asset = load_asset(folder)
    require(asset.get('schema') == SCHEMA, 'Not a plant-asset package')
    require(asset.get('schema_version') == SCHEMA_VERSION,
            'Unsupported plant-asset schema_version ' + repr(asset.get('schema_version')))
    missing = [key for key in TOP_LEVEL if key not in asset]
    require(not missing, 'Missing required fields: ' + ', '.join(missing))
    plant = asset['plant_id']
    require(isinstance(plant, str) and plant.strip() and ':' not in plant, 'plant_id must be a nonempty string without ":"')
    require(asset['units'] == UNITS, 'Contract v1 supports only scale_free_scene_units')
    require(asset['metric_scale'] is None, 'metric_scale must stay null: no metric scale is established')
    labels_ref = asset['point_labels']
    labels_path = inside(folder, labels_ref['path'])
    require(sha(labels_path) == labels_ref['sha256'], 'point_labels hash mismatch')
    labels = load_npz(labels_path)
    require(len(labels['X']) == labels_ref['point_count'], 'point_labels point_count mismatch')
    for ref in asset['source_evidence']:
        require(sha(inside(folder, ref['path'])) == ref['sha256'], 'Evidence hash mismatch: ' + ref['path'])
    graph_ref = asset['reviewed_graph']
    graph_path = inside(folder, graph_ref['path'])
    require(sha(graph_path) == graph_ref['sha256'], 'Reviewed graph hash mismatch')
    graph = read(graph_path)
    positions = {n['id']: n['xyz'] for n in graph['skeleton_nodes']}
    graph_edges = {frozenset((e['a'], e['b'])) for e in graph['skeleton_edges']}

    for kind, rows, keys in [('leaf', asset['leaves'], LEAF_KEYS), ('petiole', asset['petioles'], PETIOLE_KEYS),
                             ('attachment', asset['attachment_nodes'], ATTACHMENT_KEYS),
                             ('internode', asset['internodes'], INTERNODE_KEYS)]:
        for row in rows:
            absent = [k for k in keys if k not in row]
            require(not absent, kind + ' ' + str(row.get('id')) + ' lacks ' + ', '.join(absent))
            require(isinstance(row['id'], str) and row['id'].startswith(plant + ':' + kind + ':'),
                    'ID ' + str(row['id']) + ' is not scoped as ' + plant + ':' + kind + ':*')
    all_ids = [row['id'] for group in ['leaves', 'petioles', 'attachment_nodes', 'internodes'] for row in asset[group]]
    all_ids.append(asset['main_axis']['id'])
    require(len(all_ids) == len(set(all_ids)), 'Organ IDs must be unique')
    require(asset['main_axis']['id'] == plant + ':axis:main', 'Main axis ID must be ' + plant + ':axis:main')
    for group in ['petioles', 'attachment_nodes', 'internodes']:
        for row in asset[group]:
            require(not claims_measured(row['geometry_status']),
                    'geometry_status must not claim measured geometry: ' + row['id'])
    leaves, petioles = by_id(asset['leaves'], 'leaf'), by_id(asset['petioles'], 'petiole')
    attachments = by_id(asset['attachment_nodes'], 'attachment')

    axis = asset['main_axis']
    stem = finite_polyline(axis['polyline'], 'main_axis.polyline')
    require(axis['radius'] is None, 'Radii are not established; main_axis.radius must be null')
    require(len(axis['source_node_ids']) == len(stem), 'main_axis source_node_ids must match polyline')
    for node, xyz in zip(axis['source_node_ids'], axis['polyline']):
        require(positions.get(node) == xyz, 'main_axis node ' + str(node) + ' differs from reviewed graph')
    arcs = stem_arcs(stem)

    for node in attachments.values():
        i, t = node['stem_segment_index'], node['segment_fraction']
        require(isinstance(i, int) and 0 <= i < len(stem) - 1 and 0 <= t <= 1, 'Attachment off the stem: ' + node['id'])
        require(np.allclose(node['xyz'], stem[i] + t * (stem[i + 1] - stem[i]), atol=TOL, rtol=0),
                'Attachment coordinate is not on its stem segment: ' + node['id'])
        expected_arc = arcs[i] + t * (arcs[i + 1] - arcs[i])
        require(math.isclose(node['arc_length'], expected_arc, rel_tol=0, abs_tol=1e-9), 'Attachment arc mismatch: ' + node['id'])
        if node['geometry_status'] == 'inferred':
            require(node['source_node_id'] is None, 'Inferred attachment cannot cite a source node: ' + node['id'])
        else:
            require(node['geometry_status'] == 'extracted-candidate', 'Unknown attachment status: ' + node['id'])
            index = i + int(t == 1)
            require(t in (0, 1) and node['source_node_id'] == axis['source_node_ids'][index],
                    'Extracted attachment must sit on its cited stem node: ' + node['id'])

    owners = {}
    for a, b in zip(axis['source_node_ids'], axis['source_node_ids'][1:]):
        require(frozenset((a, b)) in graph_edges, 'Main-axis edge absent from reviewed graph: ' + str((a, b)))
        owners.setdefault(frozenset((a, b)), []).append(axis['id'])
    leaf_ids_by_cluster = {leaf['source_cluster_id']: leaf['id'] for leaf in leaves.values()}
    for p in petioles.values():
        require(p['leaf_id'] in leaves, 'Petiole leaf missing: ' + p['id'])
        require(leaves[p['leaf_id']]['petiole_id'] == p['id'], 'Leaf does not point back to petiole ' + p['id'])
        require(p['attachment_id'] in attachments, 'Dangling attachment reference: ' + p['id'])
        attachment = attachments[p['attachment_id']]
        require(leaf_ids_by_cluster.get(attachment['leaf_cluster_id']) == p['leaf_id'],
                'Attachment belongs to another leaf: ' + p['id'])
        line = finite_polyline(p['polyline'], p['id'] + '.polyline')
        require(np.allclose(line[0], attachment['xyz'], atol=TOL, rtol=0), 'Petiole does not start at its attachment: ' + p['id'])
        require(p['radius'] is None, 'Radii are not established: ' + p['id'])
        if p['inference'] is not None:
            require(p['classification'] == INFERRED and p['geometry_status'] == 'inferred',
                    'Inferred geometry must keep inferred-petiole status: ' + p['id'])
            require(attachment['geometry_status'] == 'inferred', 'Inferred petiole needs an inferred attachment: ' + p['id'])
            require(p['source_node_ids'] is None, 'Inferred petiole cannot cite extracted source nodes: ' + p['id'])
            inference = p['inference']
            absent = [k for k in INFERENCE_KEYS if k not in inference]
            require(not absent, 'Inference record lacks ' + ', '.join(absent))
            row = inference['selected_source_point_row_id']
            cluster = leaves[p['leaf_id']]['source_cluster_id']
            require(0 <= row < len(labels['X']) and int(labels['label'][row]) == cluster
                    and np.array_equal(line[-1], labels['X'][row]), 'Inferred anchor is not the cited leaf point: ' + p['id'])
            require(abs(inference['achieved_angle_degrees'] - inference['mean_reference_angle_degrees']) < 1e-7,
                    'Inferred angle does not match its stated construction: ' + p['id'])
        else:
            require(p['classification'] == EXTRACTED and str(p['geometry_status']).startswith('extracted'),
                    'Extracted petiole status is inconsistent: ' + p['id'])
            require(attachment['geometry_status'] == 'extracted-candidate', 'Extracted petiole needs extracted attachment: ' + p['id'])
            nodes = p['source_node_ids']
            require(isinstance(nodes, list) and len(nodes) == len(line) and nodes[0] == attachment['source_node_id'],
                    'Extracted petiole source nodes are inconsistent: ' + p['id'])
            for node, xyz in zip(nodes, p['polyline']):
                require(positions.get(node) == xyz, 'Petiole node differs from reviewed graph: ' + p['id'])
            for a, b in zip(nodes, nodes[1:]):
                require(frozenset((a, b)) in graph_edges, 'Petiole edge absent from reviewed graph: ' + p['id'])
                owners.setdefault(frozenset((a, b)), []).append(p['id'])
    shared = [sorted(edge) for edge, who in owners.items() if len(who) > 1]
    require(not shared, 'Extracted graph edges assigned to more than one organ: ' + str(shared))

    for leaf in leaves.values():
        require(leaf['role'] in ROLES, 'Unknown leaf role: ' + str(leaf['role']))
        require(leaf['id'] == plant + ':leaf:' + str(leaf['source_cluster_id']), 'Leaf ID must cite its source cluster: ' + leaf['id'])
        require(leaf['surface_mesh'] is None, 'Surface meshes are not part of contract v1')
        if leaf['petiole_id'] is None:
            require(leaf['role'] == 'combined-apex-region' or leaf.get('petiole_unresolved_reason'),
                    'A leaf without a petiole needs the apex role or petiole_unresolved_reason: ' + leaf['id'])
        else:
            require(leaf['petiole_id'] in petioles and petioles[leaf['petiole_id']]['leaf_id'] == leaf['id'],
                    'Leaf petiole reference is dangling: ' + leaf['id'])
        geometry = leaf['point_geometry']
        path = inside(folder, geometry['path'])
        require(sha(path) == geometry['sha256'], 'Leaf point hash mismatch: ' + leaf['id'])
        data = load_npz(path)
        rows = data.get('source_point_row_ids')
        require(rows is not None and data['X'].ndim == 2 and data['X'].shape[1] == 3 and np.isfinite(data['X']).all(),
                'Leaf points need finite X and source_point_row_ids: ' + leaf['id'])
        require(len(data['X']) == len(rows) == geometry['point_count'], 'Leaf point_count mismatch: ' + leaf['id'])
        expected = np.flatnonzero(labels['label'] == leaf['source_cluster_id'])
        require(np.array_equal(rows, expected), 'Leaf rows are not the complete source cluster: ' + leaf['id'])
        for name, values in data.items():
            if name != 'source_point_row_ids':
                require(name in labels and np.array_equal(values, labels[name][rows]),
                        'Leaf array differs from source rows: ' + leaf['id'] + '.' + name)

    ordered = asset['ordered_attachment_ids']
    require(sorted(ordered) == sorted(attachments), 'ordered_attachment_ids must list every attachment once')
    keys = [(attachments[a]['arc_length'], attachments[a]['leaf_cluster_id']) for a in ordered]
    require(keys == sorted(keys), 'ordered_attachment_ids must sort by (arc_length, leaf_cluster_id)')
    require(len(asset['internodes']) == max(0, len(ordered) - 1), 'One internode interval per adjacent attachment pair')
    for inter, a, b in zip(asset['internodes'], ordered, ordered[1:]):
        require(inter['start_attachment_id'] == a and inter['end_attachment_id'] == b, 'Internode order mismatch: ' + inter['id'])
        length = attachments[b]['arc_length'] - attachments[a]['arc_length']
        require(math.isclose(inter['length_scene_units'], length, rel_tol=0, abs_tol=1e-12), 'Internode length mismatch: ' + inter['id'])
        require(inter['radius'] is None, 'Radii are not established: ' + inter['id'])
        if length == 0:
            require('unresolved' in inter['geometry_status'], 'Zero interval must be marked unresolved: ' + inter['id'])
    cotyledons = {leaf['source_cluster_id'] for leaf in leaves.values() if leaf['role'] == 'cotyledon'}
    for pair in asset['cotyledon_pairs']:
        require(len(pair) == 2 and set(pair) <= cotyledons, 'cotyledon_pairs must name two cotyledon clusters')
    edge_counts = {'main_axis': sum(1 for who in owners.values() if who == [axis['id']]),
                   'extracted_petiole': sum(1 for who in owners.values() if who != [axis['id']])}
    edge_counts['unassigned'] = len(graph_edges) - len(owners)
    require(graph_ref.get('edge_classification') == edge_counts, 'reviewed_graph.edge_classification is stale')
    return {'plant_id': plant, 'schema_version': SCHEMA_VERSION, 'leaves': len(leaves), 'petioles': len(petioles),
            'extracted_petioles': sorted(p['leaf_id'] for p in petioles.values() if p['inference'] is None),
            'inferred_petioles': sorted(p['leaf_id'] for p in petioles.values() if p['inference'] is not None),
            'attachments': len(attachments), 'internodes': len(asset['internodes']),
            'graph_nodes': len(positions), 'graph_edges': len(graph_edges), 'edge_classification': edge_counts}


def find_leaf(asset, leaf):
    """Accept a full leaf ID or a source cluster integer (or its decimal string)."""
    for row in asset['leaves']:
        if row['id'] == leaf or (str(leaf).lstrip('-').isdigit() and row['source_cluster_id'] == int(leaf)):
            return row
    raise PlantAssetError('No leaf ' + str(leaf) + ' in ' + asset['plant_id'])


def petiole_status(petiole):
    if petiole is None:
        return 'none'
    return INFERRED if petiole['inference'] is not None else 'extracted'


def leaf_export_document(folder, leaf, asset=None, inline=False):
    """Build the single-leaf export record; ``inline`` embeds X and row IDs instead of naming points.npz."""
    folder = Path(folder).resolve()
    asset = asset or load_asset(folder)
    row = find_leaf(asset, leaf)
    petiole = next((p for p in asset['petioles'] if p['id'] == row['petiole_id']), None)
    attachment = None if petiole is None else next(a for a in asset['attachment_nodes'] if a['id'] == petiole['attachment_id'])
    source = row['point_geometry']
    exported = copy.deepcopy(row)
    geometry = {'point_count': source['point_count'], 'status': source['status'],
                'source_path': source['path'], 'source_sha256': source['sha256']}
    if inline:
        data = load_npz(inside(folder, source['path']))
        geometry.update(arrays=['X', 'source_point_row_ids'],
                        inline={'X': data['X'].tolist(), 'source_point_row_ids': data['source_point_row_ids'].tolist()})
    else:
        geometry.update(path='points.npz', sha256=source['sha256'],
                        arrays=sorted(load_npz(inside(folder, source['path']))))
    exported['point_geometry'] = geometry
    return {'schema': LEAF_EXPORT_SCHEMA, 'schema_version': LEAF_EXPORT_VERSION,
            'export_route': 'viewer-download' if inline else 'cli-folder',
            'source_asset': {'schema': asset['schema'], 'schema_version': asset['schema_version'],
                             'plant_id': asset['plant_id'], 'capture_id': asset['capture_id'], 'id_scope': asset['id_scope'],
                             'plant_json_sha256': sha(folder / 'plant.json'),
                             'source_graph_sha256': asset['source_graph_sha256'], 'point_labels': asset['point_labels']},
            'units': asset['units'], 'metric_scale': asset['metric_scale'],
            'leaf': exported, 'petiole_status': petiole_status(petiole),
            'petiole': copy.deepcopy(petiole), 'attachment': copy.deepcopy(attachment),
            'limits': list(asset['limits'])}


def export_leaf(folder, leaf, output):
    """Validate the asset, then write one leaf, its petiole and attachment to a fresh folder."""
    folder, output = Path(folder).resolve(), Path(output).resolve()
    require(not output.exists(), 'Output already exists; choose a fresh directory: ' + str(output))
    require(folder != output and folder not in output.parents, 'Write exports outside the source asset')
    asset = load_asset(folder)
    validate_asset(folder, asset)
    document = leaf_export_document(folder, leaf, asset)
    output.mkdir(parents=True)
    shutil.copyfile(inside(folder, document['leaf']['point_geometry']['source_path']), output / 'points.npz')
    with (output / 'leaf.json').open('x') as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write('\n')
    load_leaf_export(output)
    return document


def load_leaf_export(path):
    """Reload a CLI export folder or a viewer JSON download; returns (document, arrays)."""
    path = Path(path)
    folder = path if path.is_dir() else path.parent
    document = read(path / 'leaf.json' if path.is_dir() else path)
    require(document.get('schema') == LEAF_EXPORT_SCHEMA and document.get('schema_version') == LEAF_EXPORT_VERSION,
            'Not a supported plant-asset leaf export')
    require(document['units'] == UNITS and document['metric_scale'] is None, 'Leaf exports stay scale-free')
    leaf, petiole, attachment = document['leaf'], document['petiole'], document['attachment']
    plant = document['source_asset']['plant_id']
    require(leaf['id'] == plant + ':leaf:' + str(leaf['source_cluster_id']) and leaf['role'] in ROLES, 'Bad leaf identity')
    status = document['petiole_status']
    require(status == petiole_status(petiole), 'petiole_status disagrees with the petiole record')
    if petiole is None:
        require(attachment is None and leaf['petiole_id'] is None, 'Leaf without petiole cannot carry an attachment')
    else:
        require(petiole['id'] == leaf['petiole_id'] and petiole['leaf_id'] == leaf['id'], 'Petiole belongs to another leaf')
        require(attachment is not None and attachment['id'] == petiole['attachment_id'], 'Attachment record missing')
        require(np.allclose(petiole['polyline'][0], attachment['xyz'], atol=TOL, rtol=0), 'Petiole does not start at attachment')
        if status == INFERRED:
            require(petiole['classification'] == INFERRED and petiole['geometry_status'] == 'inferred'
                    and attachment['geometry_status'] == 'inferred', 'Inferred status was lost in the export')
        else:
            require(petiole['classification'] == EXTRACTED and attachment['geometry_status'] == 'extracted-candidate',
                    'Extracted status is inconsistent in the export')
    geometry = leaf['point_geometry']
    if 'inline' in geometry:
        arrays = {'X': np.asarray(geometry['inline']['X'], dtype=float),
                  'source_point_row_ids': np.asarray(geometry['inline']['source_point_row_ids'], dtype=np.int64)}
    else:
        points = inside(folder, geometry['path'])
        require(sha(points) == geometry['sha256'], 'Exported point file hash mismatch')
        arrays = load_npz(points)
    X, rows = arrays['X'], arrays['source_point_row_ids']
    require(X.ndim == 2 and X.shape[1] == 3 and np.isfinite(X).all(), 'Exported X must be finite xyz points')
    require(len(X) == len(rows) == geometry['point_count'] and len(np.unique(rows)) == len(rows) and (rows >= 0).all(),
            'Exported points and row IDs disagree')
    return document, arrays
