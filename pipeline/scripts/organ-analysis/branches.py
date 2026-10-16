"""Conservative leaf-free terminal-subtree candidates, not confirmed clutter.

See DESIGN.md. Original graph and point labels are never changed by detection.
"""
import copy
import hashlib
import json
import math

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree


def edge_key(a,b):
    return '-'.join(str(i) for i in sorted((int(a),int(b))))


def graph_digest(graph):
    return hashlib.sha256(json.dumps(graph,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def identify_candidates(graph, points, labels, tip_gap):
    if not math.isfinite(tip_gap) or tip_gap<=0:
        raise ValueError('tip_gap must be positive and finite')
    points,labels=np.asarray(points),np.asarray(labels)
    if points.ndim!=2 or points.shape[1]!=3 or labels.shape!=(len(points),) or not np.isfinite(points).all():
        raise ValueError('Point coordinates and labels must align and be finite')
    tree=nx.Graph()
    positions={}
    for row in graph['skeleton_nodes']:
        node=row['id']
        xyz=np.asarray(row['xyz'],dtype=float)
        if node in positions or xyz.shape!=(3,) or not np.isfinite(xyz).all():
            raise ValueError('Invalid or duplicate skeleton node')
        positions[node]=xyz
        tree.add_node(node)
    for edge in graph['skeleton_edges']:
        a,b=edge['a'],edge['b']
        if a not in positions or b not in positions or a==b or tree.has_edge(a,b):
            raise ValueError('Invalid or duplicate skeleton edge')
        tree.add_edge(a,b,weight=float(np.linalg.norm(positions[a]-positions[b])))
    if not positions or not nx.is_tree(tree):
        raise ValueError('Expected a connected skeleton tree')
    ids=sorted(positions)
    coords=np.array([positions[i] for i in ids])
    def match(xyz):
        distances=np.linalg.norm(coords-np.asarray(xyz),axis=1)
        i=int(np.argmin(distances))
        if distances[i]>1e-9:
            raise ValueError('Saved main/insertion coordinate has no graph node')
        return ids[i]
    main=[match(xyz) for xyz in graph['main_stem']]
    if len(main)<2 or len(set(main))!=len(main) or any(not tree.has_edge(a,b) for a,b in zip(main,main[1:])):
        raise ValueError('Invalid main path')
    root=main[0]
    protected=set()
    guarded=set()
    associations=[]
    def protect(node,kind,leaf_id=None,distance=None):
        path=nx.shortest_path(tree,root,node)
        destination=guarded if kind in ['uncertain_leaf_base_adjacency','near_leaf_terminal_guard'] else protected
        destination.update(edge_key(a,b) for a,b in zip(path,path[1:]))
        associations.append({'node_id':node,'reason':kind,'leaf_id':leaf_id,'distance_scene_units':distance})
    protect(main[-1],'main_axis')
    tips=sorted(n for n in tree if tree.degree(n)==1 and n!=root)
    for leaf in graph['leaves']:
        base=np.asarray(leaf['base_xyz'])
        protect(match(leaf['insertion_xyz']),'saved_leaf_insertion',leaf['leaf_id'])
        if leaf['base_rule']=='tip':
            node=min(tips,key=lambda n:(float(np.linalg.norm(positions[n]-base)),n))
            gap=float(np.linalg.norm(positions[node]-base))
            if not math.isclose(gap,leaf['gap'],rel_tol=1e-6,abs_tol=1e-9):
                raise ValueError('Cannot recover saved tip association without ambiguity')
            protect(node,'saved_leaf_tip',leaf['leaf_id'],gap)
        elif leaf['base_rule']=='nearest':
            # Raw base_xyz is a blade point, not a graph-node coordinate. Protect
            # its nearest graph path conservatively, without claiming attachment.
            node=min(ids,key=lambda n:(float(np.linalg.norm(positions[n]-base)),n))
            protect(node,'uncertain_leaf_base_adjacency',leaf['leaf_id'],float(np.linalg.norm(positions[node]-base)))
        else:
            raise ValueError('Unknown leaf-base association rule')
    supported={leaf['leaf_id'] for leaf in graph['leaves']}
    mask=np.isin(labels,list(supported))
    if not mask.any():
        raise ValueError('No source leaf centers available for endpoint support check')
    leaf_tree=cKDTree(points[mask])
    leaf_labels=labels[mask]
    endpoint_support=[]
    for node in tips:
        gap,index=leaf_tree.query(positions[node])
        item={'node_id':node,'nearest_leaf_id':int(leaf_labels[index]),'distance_scene_units':float(gap),
              'within_tip_gap':bool(gap<=tip_gap)}
        endpoint_support.append(item)
        if gap<=tip_gap:
            protect(node,'near_leaf_terminal_guard',int(leaf_labels[index]),float(gap))

    remaining=nx.Graph()
    for a,b in tree.edges:
        if edge_key(a,b) not in protected:
            remaining.add_edge(a,b)
    protected_nodes=set(main)
    for a,b in tree.edges:
        if edge_key(a,b) in protected:
            protected_nodes.update([a,b])
    candidates=[]
    for component in nx.connected_components(remaining):
        edges=sorted([sorted([a,b]) for a,b in remaining.subgraph(component).edges])
        attachments=sorted(component & protected_nodes)
        terminals=sorted(n for n in component if tree.degree(n)==1 and n!=root)
        if len(attachments)!=1 or not terminals:
            raise ValueError('Unprotected component is not a terminal subtree with one attachment')
        candidate={'id':'branch-'+','.join(edge_key(*e) for e in edges),
                   'attachment_node_id':attachments[0],'terminal_node_ids':terminals,
                   'node_ids':sorted(component),'edges':edges,'edge_ids':[edge_key(*e) for e in edges],
                   'length_scene_units':sum(tree[a][b]['weight'] for a,b in edges),
                   'status':'candidate_leaf_free_subtree','human_disposition':None,
                   'reason':'No saved leaf-tip association downstream; main/known-leaf paths protected',
                   'guarded_edge_ids':[edge_key(*e) for e in edges if edge_key(*e) in guarded],
                   'eligible_for_pruning':not any(edge_key(*e) in guarded for e in edges),
                   'nearest_leaf_distance_at_terminals':{str(n):next(s['distance_scene_units'] for s in endpoint_support if s['node_id']==n) for n in terminals}}
        candidates.append(candidate)
    candidates.sort(key=lambda c:(c['attachment_node_id'],c['terminal_node_ids']))
    # Short stable review labels, with the full edge-derived identity retained.
    for i,c in enumerate(candidates,1):
        c['review_label']='B'+str(i)
    return {'schema':'leaf-free-branch-candidates-v1','source_graph_sha256':graph_digest(graph),
            'tip_gap_scene_units':tip_gap,'protected_edge_ids':sorted(protected|guarded),
            'known_leaf_path_edge_ids':sorted(protected),'uncertainty_guard_edge_ids':sorted(guarded-protected),
            'leaf_support_records':associations,'terminal_leaf_proximity':endpoint_support,'candidates':candidates,
                   'limits':['Candidate topology is not confirmed clutter','Tendrils or missed tissue can be candidates',
                      'Nearest-base guards preserve uncertainty, not measured attachment','No graph or point was deleted']}


def prune_candidates(graph, report, candidate_ids):
    """Return a derivative graph for explicitly selected candidates; never mutate.

    This helper is not invoked by candidate detection or review preview.
    """
    if graph_digest(graph)!=report['source_graph_sha256']:
        raise ValueError('Candidate report belongs to a different graph')
    by_id={c['id']:c for c in report['candidates']}
    selected=set(candidate_ids)
    if not selected.issubset(by_id):
        raise ValueError('Unknown candidate selection')
    removed={key for cid in selected for key in by_id[cid]['edge_ids']}
    if removed & set(report['protected_edge_ids']):
        raise ValueError('Refusing to remove a protected main/leaf path')
    result=copy.deepcopy(graph)
    result['skeleton_edges']=[e for e in result['skeleton_edges'] if edge_key(e['a'],e['b']) not in removed]
    retained={e[k] for e in result['skeleton_edges'] for k in ['a','b']}
    result['skeleton_nodes']=[n for n in result['skeleton_nodes'] if n['id'] in retained]
    return result
