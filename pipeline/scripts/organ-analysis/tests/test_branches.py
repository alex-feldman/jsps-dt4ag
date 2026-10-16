import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from branches import identify_candidates, prune_candidates, graph_digest, edge_key, apply_review


def fixture(extra=None):
    positions={0:[0,0,0],1:[0,0,1],2:[0,0,2],3:[1,0,1],4:[2,0,1],5:[0,1,1],6:[0,2,1]}
    edges=[(0,1),(1,2),(1,3),(3,4),(1,5),(5,6)]
    if extra:
        positions.update(extra[0]);edges.extend(extra[1])
    graph={'skeleton_nodes':[{'id':n,'xyz':p} for n,p in positions.items()],
           'skeleton_edges':[{'a':a,'b':b,'main':(a,b) in [(0,1),(1,2)]} for a,b in edges],
           'main_stem':[positions[n] for n in [0,1,2]],
           'leaves':[{'leaf_id':0,'insertion_xyz':positions[2],'base_xyz':[.001,0,2],'base_rule':'tip','gap':.001,'petiole_network_length':0},
                     {'leaf_id':1,'insertion_xyz':positions[1],'base_xyz':[2.001,0,1],'base_rule':'tip','gap':.001,'petiole_network_length':2}]}
    points=np.array([[.001,0,2],[2.001,0,1]])
    return graph,points,np.array([0,1])


class BranchTests(unittest.TestCase):
    def test_leaf_paths_protected_and_source_unchanged(self):
        graph,points,labels=fixture()
        original=copy.deepcopy(graph)
        report=identify_candidates(graph,points,labels,.01)
        self.assertEqual(graph,original)
        self.assertEqual(len(report['candidates']),1)
        self.assertEqual(report['candidates'][0]['edge_ids'],['1-5','5-6'])
        self.assertEqual(report['candidates'][0]['attachment_node_id'],1)
        self.assertEqual(set(report['protected_edge_ids']),{'0-1','1-2','1-3','3-4'})

    def test_shared_axis_with_descendant_leaf_not_false(self):
        graph,points,labels=fixture(({7:[1,1,1]},[(3,7)]))
        report=identify_candidates(graph,points,labels,.01)
        candidates={e for c in report['candidates'] for e in c['edge_ids']}
        self.assertNotIn('1-3',candidates)
        self.assertNotIn('3-4',candidates)
        self.assertIn('3-7',candidates)

    def test_leaf_near_terminal_guard(self):
        graph,points,labels=fixture()
        points=np.concatenate([points,[[0,2.001,1]]])
        report=identify_candidates(graph,points,np.array([0,1,1]),.01)
        self.assertEqual(len(report['candidates']),1)
        self.assertFalse(report['candidates'][0]['eligible_for_pruning'])
        with self.assertRaisesRegex(ValueError,'protected'):
            prune_candidates(graph,report,[report['candidates'][0]['id']])
        self.assertTrue(any(r['reason']=='near_leaf_terminal_guard' and r['node_id']==6 for r in report['leaf_support_records']))

    def test_nearest_base_uncertainty_preserves_connection(self):
        graph,points,labels=fixture()
        graph['leaves'].append({'leaf_id':7,'insertion_xyz':[0,0,1],'base_xyz':[0,2.01,1],
                                'base_rule':'nearest','gap':.01,'petiole_network_length':0})
        report=identify_candidates(graph,points,labels,.005)
        self.assertEqual(len(report['candidates']),1)
        self.assertFalse(report['candidates'][0]['eligible_for_pruning'])
        self.assertTrue(any(r['reason']=='uncertain_leaf_base_adjacency' for r in report['leaf_support_records']))

    def test_multi_terminal_subtree_merged_once(self):
        graph,points,labels=fixture(({7:[1,2,1]},[(5,7)]))
        report=identify_candidates(graph,points,labels,.01)
        self.assertEqual(len(report['candidates']),1)
        c=report['candidates'][0]
        self.assertEqual(c['terminal_node_ids'],[6,7])
        self.assertEqual(c['edge_ids'],['1-5','5-6','5-7'])

    def test_pruning_is_selected_derivative_not_source_mutation(self):
        graph,points,labels=fixture()
        digest=graph_digest(graph)
        report=identify_candidates(graph,points,labels,.01)
        pruned=prune_candidates(graph,report,[report['candidates'][0]['id']])
        self.assertEqual(graph_digest(graph),digest)
        self.assertEqual(len(pruned['skeleton_edges']),4)
        self.assertEqual(pruned['leaves'],graph['leaves'])
        self.assertEqual(pruned['main_stem'],graph['main_stem'])
        self.assertEqual(prune_candidates(graph,report,[]),graph)

    def test_changed_graph_and_unknown_selection_refused(self):
        graph,points,labels=fixture()
        report=identify_candidates(graph,points,labels,.01)
        with self.assertRaisesRegex(ValueError,'Unknown candidate'):
            prune_candidates(graph,report,['nonexistent'])
        graph['skeleton_nodes'][0]['xyz'][0]=.1
        with self.assertRaisesRegex(ValueError,'different graph'):
            prune_candidates(graph,report,[report['candidates'][0]['id']])

    def test_invalid_tree_and_gap_refused(self):
        graph,points,labels=fixture()
        with self.assertRaises(ValueError):identify_candidates(graph,points,labels,0)
        graph['skeleton_edges'].append({'a':4,'b':6,'main':False})
        with self.assertRaisesRegex(ValueError,'tree'):
            identify_candidates(graph,points,labels,.01)

    def test_every_endpoint_has_unique_stable_id(self):
        graph,points,labels=fixture(({7:[1,2,1]},[(5,7)]))
        report=identify_candidates(graph,points,labels,.01)
        endpoints=report['endpoint_records']
        self.assertEqual(len(endpoints),5)
        self.assertEqual(len({e['endpoint_id'] for e in endpoints}),5)
        self.assertEqual({e['endpoint_id'] for e in endpoints if e['review_group']=='B1'},{'b1-a','b1-b'})

    def test_retained_leaf_connected_branch_not_candidate_and_id_stable(self):
        graph,points,labels=fixture()
        report=identify_candidates(graph,points,labels,.01)
        reviewed=apply_review(graph,report,{'candidate_dispositions':{'B1':'retain_leaf_connected'}})
        self.assertEqual(reviewed['candidates'],[])
        self.assertEqual(reviewed['retained_branches'][0]['review_label'],'B1')
        self.assertEqual(next(e for e in reviewed['endpoint_records'] if e['source_node_id']==6)['endpoint_id'],'b1-a')
        self.assertIn('5-6',reviewed['protected_edge_ids'])
        self.assertEqual(report['candidates'][0]['human_disposition'],None)

    def test_human_upstream_candidate_can_overlap_saved_leaf_route(self):
        graph,points,labels=fixture(({7:[1,1,1]},[(3,7)]))
        report=identify_candidates(graph,points,labels,.01)
        reviewed=apply_review(graph,report,{'candidate_dispositions':{'B1':'confirmed_false'},'upstream_candidate_labels':['B2']})
        extra=reviewed['extra_review_paths'][0]
        self.assertEqual(extra['edge_ids'],['1-3'])
        self.assertTrue(extra['intersects_saved_leaf_path'])
        self.assertFalse(extra['eligible_for_pruning'])
        self.assertEqual(reviewed['candidates'][0]['status'],'human_confirmed_false')


if __name__=='__main__':unittest.main()
