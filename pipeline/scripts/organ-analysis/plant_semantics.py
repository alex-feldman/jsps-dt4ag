"""Experimental partial plant semantics; source observations never imply accepted anatomy."""
import copy
import json
import re
from pathlib import Path
import shutil
import uuid

import plant_asset as pa
from analyze_organs import VERSION, sha

SCHEMA = 'plant-partial-semantics'
REGISTRY_SCHEMA = 'plant-semantic-identity-registry'
VERSION_SCHEMA = 1


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def tree_hashes(folder):
    folder = Path(folder)
    pa.require(not any(p.is_symlink() for p in folder.rglob('*')), 'Source/package symlinks are unsupported')
    return {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def binding(folder, asset):
    return {'plant_json_sha256': sha(Path(folder) / 'plant.json'),
            'capture_id': asset['capture_id'], 'source_graph_sha256': asset['source_graph_sha256'],
            'tree_sha256': tree_hashes(folder)}


def identity_keys(asset):
    keys = {'plant': 'plant', 'axis:' + asset['main_axis']['id']: 'axis'}
    for leaf in asset['leaves']:
        kind = 'region' if leaf['role'] == 'combined-apex-region' else 'leaf'
        keys[kind + ':' + leaf['id']] = kind
    keys.update({'candidate:' + row['id']: 'candidate' for row in asset['attachment_nodes']})
    return keys


def registry_for(folder, asset, registry=None):
    keys = identity_keys(asset)
    if registry is None:
        return {'schema': REGISTRY_SCHEMA, 'schema_version': VERSION_SCHEMA,
                'source': binding(folder, asset),
                'identities': {key: {'kind': kind, 'id': kind + ':' + str(uuid.uuid4())}
                               for key, kind in sorted(keys.items())}}
    pa.require(set(registry) == {'schema', 'schema_version', 'source', 'identities'}, 'Bad registry fields')
    pa.require(registry['schema'] == REGISTRY_SCHEMA and type(registry['schema_version']) is int
               and registry['schema_version'] == VERSION_SCHEMA,
               'Unsupported identity registry')
    pa.require(registry['source'] == binding(folder, asset), 'Registry source mismatch; explicit correspondence required')
    rows = registry['identities']
    pa.require(set(rows) == set(keys), 'Registry source mapping mismatch')
    ids = []
    for key, kind in keys.items():
        row = rows[key]
        pa.require(set(row) == {'id', 'kind'} and row['kind'] == kind, 'Registry identity kind mismatch')
        value = row['id']
        pa.require(isinstance(value, str) and value.startswith(kind + ':'), 'Invalid opaque identity kind')
        try:
            parsed = uuid.UUID(value.split(':', 1)[1])
        except (ValueError, AttributeError):
            raise pa.PlantAssetError('Identity must contain an opaque UUID')
        pa.require(str(parsed) == value.split(':', 1)[1], 'Noncanonical UUID')
        ids.append(value)
    pa.require(len(ids) == len(set(ids)), 'Duplicate registry identity')
    return copy.deepcopy(registry)


def document(asset, registry):
    ids = registry['identities']
    def ident(kind, source):
        return ids[kind + ':' + source]['id']
    axis_id = ident('axis', asset['main_axis']['id'])
    leaves, regions, candidates = [], [], []
    leaf_by_source = {}
    for source in asset['leaves']:
        kind = 'region' if source['role'] == 'combined-apex-region' else 'leaf'
        row = {'id': ident(kind, source['id']), 'source_leaf_id': source['id'], 'role': source['role']}
        if kind == 'leaf':
            row.update(accepted_node_id=None)
            leaves.append(row)
            leaf_by_source[source['source_cluster_id']] = row['id']
        else:
            regions.append(row)
    for source in asset['attachment_nodes']:
        candidates.append({'id': ident('candidate', source['id']), 'axis_id': axis_id,
                           'source_attachment_id': source['id'], 'leaf_ids': [leaf_by_source[source['leaf_cluster_id']]],
                           'status': 'unresolved-insertion-candidate', 'node_rank': None})
    # Equal source samples establish ambiguous observations, never common anatomical nodes.
    ambiguous = []
    for index, a in enumerate(asset['attachment_nodes']):
        for b in asset['attachment_nodes'][index + 1:]:
            if a['source_node_id'] is not None and a['source_node_id'] == b['source_node_id']:
                ambiguous.append({'candidate_ids': [ident('candidate', a['id']), ident('candidate', b['id'])],
                                  'source_node_id': a['source_node_id'], 'relationship': 'shared-sample; anatomical relationship unresolved'})
    apex = [r['id'] for r in regions]
    return {'schema': SCHEMA, 'schema_version': VERSION_SCHEMA, 'generator_version': VERSION,
            'source': registry['source'], 'registry_path': 'identity-registry.json',
            'source_path': 'plant-asset', 'plant_id': ids['plant']['id'],
            'scope': 'exact reviewed source asset only; no cross-date correspondence',
            'axes': [{'id': axis_id, 'source_axis_id': asset['main_axis']['id'],
                      'designation': 'reviewed geometric main-axis path; anatomical axis/base unvalidated',
                      'parent_axis_id': None, 'attachment_node_id': None, 'basal_node_known': False,
                      'basal_node_id': None, 'apex_region_ids': apex, 'botanical_order_known': False}],
            'leaves': leaves, 'regions': regions, 'insertion_candidates': candidates,
            'ambiguous_insertions': ambiguous,
            'cotyledon_pairs': [[leaf_by_source[c] for c in pair] for pair in asset['cotyledon_pairs']],
            'accepted_nodes': [], 'accepted_internodes': [],
            'limits': ['Candidate groups are geometric observations, not botanical order or count',
                       'Unseen nodes and unknown basal/apex boundaries remain possible',
                       'Source units, inference labels and human-review limits remain authoritative']}


def validate_topology(doc):
    """Validate explicitly accepted topology, also usable on small synthetic fixtures.

    This is structural consistency, not scientific acceptance. Source-package
    validation separately refuses any anatomy absent from its builder inputs.
    """
    axes = pa.by_id(doc['axes'], 'axis')
    leaves = pa.by_id(doc['leaves'], 'leaf')
    nodes = pa.by_id(doc['accepted_nodes'], 'accepted node')
    incoming, outgoing = {}, {}
    for node in nodes.values():
        pa.require(node['axis_id'] in axes and node.get('status') == 'accepted', 'Bad accepted node axis/status')
        pa.require(isinstance(node.get('evidence'), str) and node['evidence'].strip(), 'Accepted node needs evidence')
        pa.require(isinstance(node['leaf_ids'], list) and len(set(node['leaf_ids'])) == len(node['leaf_ids'])
                   and all(x in leaves for x in node['leaf_ids']), 'Bad accepted leaf references')
        rank = node['node_rank']
        pa.require(rank is None or (type(rank) is int and rank >= 0), 'Bad node rank')
    attached = [leaf for n in nodes.values() for leaf in n['leaf_ids']]
    pa.require(len(attached) == len(set(attached)), 'Leaf attached to multiple accepted nodes')
    for leaf in leaves.values():
        expected = next((n['id'] for n in nodes.values() if leaf['id'] in n['leaf_ids']), None)
        pa.require(leaf['accepted_node_id'] == expected, 'Accepted leaf backlink mismatch')
    inters = pa.by_id(doc['accepted_internodes'], 'internode')
    for inter in inters.values():
        a, b = inter['basal_node_id'], inter['apical_node_id']
        pa.require(a != b and a in nodes and b in nodes, 'Internode needs distinct accepted endpoints')
        pa.require(nodes[a]['axis_id'] == nodes[b]['axis_id'] == inter['axis_id'], 'Cross-axis internode')
        pa.require(inter.get('status') == 'accepted' and isinstance(inter.get('evidence'), str)
                   and inter['evidence'].strip(), 'Internode needs explicit accepted evidence')
        pa.require(a not in outgoing and b not in incoming, 'Contradictory accepted succession')
        outgoing[a], incoming[b] = b, a
        ra, rb = nodes[a]['node_rank'], nodes[b]['node_rank']
        pa.require(ra is None or rb is None or rb == ra + 1, 'Ranks disagree with accepted succession')
    for start in nodes:
        seen, current = set(), start
        while current in outgoing:
            pa.require(current not in seen, 'Accepted succession cycle')
            seen.add(current)
            current = outgoing[current]
    for axis in axes.values():
        own = [n for n in nodes.values() if n['axis_id'] == axis['id']]
        ranks = [n['node_rank'] for n in own if n['node_rank'] is not None]
        pa.require(len(ranks) == len(set(ranks)), 'Duplicate axis-local rank')
        if axis['basal_node_known']:
            basal = nodes.get(axis['basal_node_id'])
            pa.require(basal is not None and basal['axis_id'] == axis['id'] and basal['node_rank'] == 0
                       and basal['id'] not in incoming, 'Known basal node must be rank zero without predecessor')
        else:
            pa.require(axis['basal_node_id'] is None and not ranks, 'Unknown basal identity requires null absolute ranks')
    return {'accepted_nodes': len(nodes), 'accepted_internodes': len(inters)}


def validate(folder):
    folder = Path(folder)
    doc = json.loads((folder / 'semantic.json').read_text())
    registry = json.loads((folder / 'identity-registry.json').read_text())
    source = folder / 'plant-asset'
    asset = pa.load_asset(source)
    summary = pa.validate_asset(source, asset)
    registry_for(source, asset, registry)
    expected = document(asset, registry)
    provenance = doc.get('generator_version')
    pa.require(isinstance(provenance, str) and re.fullmatch(r'\d+\.\d+\.\d+(?:-alpha\.\d+)?', provenance),
               'Bad generator version provenance')
    expected['generator_version'] = provenance
    expected.update(source_tree_sha256=tree_hashes(source), registry_sha256=sha(folder / 'identity-registry.json'))
    pa.require(json.dumps(doc, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False),
               'Semantic document differs from source/registry; promotion or tampering refused')
    validate_topology(doc)
    return {'schema_version': VERSION_SCHEMA, 'leaves': len(doc['leaves']), 'regions': len(doc['regions']),
            'insertion_candidates': len(doc['insertion_candidates']), 'accepted_nodes': 0,
            'accepted_internodes': 0, 'source_validation': summary}


def build(source, output, registry_path=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    pa.require(not output.exists(), 'Output already exists')
    pa.require(source != output and source not in output.parents, 'Output must be outside source')
    asset = pa.load_asset(source)
    pa.validate_asset(source, asset)
    before = tree_hashes(source)
    old = None if registry_path is None else json.loads(Path(registry_path).read_text())
    registry = registry_for(source, asset, old)
    output.mkdir(parents=True)
    shutil.copytree(source, output / 'plant-asset')
    write(output / 'identity-registry.json', registry)
    doc = document(asset, registry)
    doc.update(source_tree_sha256=before, registry_sha256=sha(output / 'identity-registry.json'))
    write(output / 'semantic.json', doc)
    result = validate(output)
    pa.require(before == tree_hashes(source) == tree_hashes(output / 'plant-asset'), 'Source tree changed during build')
    return result


def leaves_at(doc, identity, accepted=False):
    collection = doc['accepted_nodes'] if accepted else doc['insertion_candidates']
    row = next((r for r in collection if r['id'] == identity), None)
    pa.require(row is not None, 'Unknown ' + ('accepted node' if accepted else 'candidate'))
    by_id = pa.by_id(doc['leaves'], 'leaf')
    return [by_id[x] for x in row['leaf_ids']]


def below_node_phytomer(doc, node_id):
    nodes = pa.by_id(doc['accepted_nodes'], 'node')
    pa.require(node_id in nodes, 'Unknown accepted node')
    incoming = next((r for r in doc['accepted_internodes'] if r['apical_node_id'] == node_id), None)
    return {'convention': 'below-node', 'node': nodes[node_id], 'leaves': leaves_at(doc, node_id, True),
            'internode': incoming, 'basal_boundary_complete': incoming is not None}


def query(folder, mode, identity=None):
    validate(folder)
    folder = Path(folder)
    doc = json.loads((folder / 'semantic.json').read_text())
    asset = pa.load_asset(folder / 'plant-asset')
    if mode == 'candidate-leaves':
        return leaves_at(doc, identity)
    if mode == 'topology':
        return {k: doc[k] for k in ('axes', 'accepted_nodes', 'accepted_internodes')}
    pa.require(mode == 'candidate-groups', 'Unknown query')
    axis = identity or doc['axes'][0]['id']
    pa.require(any(r['id'] == axis for r in doc['axes']), 'Unknown axis')
    attachments = pa.by_id(asset['attachment_nodes'], 'attachment')
    groups = {}
    for row in doc['insertion_candidates']:
        if row['axis_id'] == axis:
            source = attachments[row['source_attachment_id']]
            groups.setdefault(source['arc_length'], []).append({**row, 'source_attachment': source,
                'leaves': leaves_at(doc, row['id']),
                'source_petioles': [p for p in asset['petioles'] if p['attachment_id'] == source['id']]})
    return {'axis_id': axis, 'order_basis': 'source geometric root-to-apex arc',
            'botanical_order_known': False, 'possible_unseen_nodes': True,
            'groups': [{'arc_length_scene_units': arc, 'candidates': rows} for arc, rows in sorted(groups.items())]}
