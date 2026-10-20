import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import analyze_organs as cli
from geometry import Geometry, load_ply
from viewer import write_viewer


class OrganAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ['images','plants','instances/photo']:
            (self.root/name).mkdir(parents=True)
        Image.new('RGB',(32,32),(30,160,30)).save(self.root/'images/photo.png')
        for name in ['plants/photo.png','instances/photo/leaf_01.png']:
            Image.new('L',(32,32),255).save(self.root/name)
        (self.root/'instances/photo/instances.json').write_text(json.dumps({'instances':[{'file':'leaf_01.png'}]}))
        props = ['x','y','z','opacity','scale_0','scale_1','scale_2','f_dc_0','f_dc_1','f_dc_2']
        data = np.zeros(24,dtype=[(p,'<f4') for p in props])
        data['x'] = np.arange(24)*.0001
        data['z'],data['opacity'] = -3,3
        for p in ['scale_0','scale_1','scale_2']: data[p] = -4.6
        data['f_dc_1'] = .8
        self.ply = self.root/'model.ply'
        with self.ply.open('wb') as f:
            f.write(('ply\nformat binary_little_endian 1.0\nelement vertex 24\n'+
                     ''.join('property float '+p+'\n' for p in props)+'end_header\n').encode())
            f.write(data.tobytes())
        self.transform={'fl_x':10,'fl_y':10,'cx':16,'cy':16,'w':32,'h':32,
                        'frames':[{'file_path':'images/frame_00001.png','transform_matrix':np.eye(4).tolist()}]}
        (self.root/'transforms.json').write_text(json.dumps(self.transform))
        (self.root/'dataparser.json').write_text(json.dumps({'transform':np.eye(4)[:3].tolist(),'scale':1}))
        (self.root/'frame_map.json').write_text(json.dumps({'frame_00001':'photo'}))
        self.spec={'capture_id':'tiny-plant','inputs':{'ply':'model.ply','transforms':'transforms.json',
                   'dataparser_transforms':'dataparser.json','frame_map':'frame_map.json','images':'images',
                   'plant_masks':'plants','leaf_instances':'instances','image_extension':'.png'}}
        self.config=self.root/'spec.json'
        self.save()

    def save(self):
        self.config.write_text(json.dumps(self.spec))

    def invoke(self,output,*args):
        with contextlib.redirect_stdout(io.StringIO()):
            return cli.main(['--config',str(self.config),'--output',str(output),*args])

    def test_leaf_pass_and_verified_reuse_failure_record(self):
        out=self.root/'leaves'
        self.assertEqual(self.invoke(out,'--stage','leaves'),0)
        self.assertEqual(cli.read(out/'run-record.json')['status'],'complete')
        self.assertEqual(len(cli.read(out/'organ-evidence.json')['leaves']),1)
        self.assertEqual(cli.read(out/'organ-evidence.json')['leaves'][0]['point_count'],24)
        self.assertEqual(cli.read(out/'organ-evidence.json')['leaves'][0]['association_support'],'supported')
        self.assertIn('<canvas', (out/'viewer.html').read_text())
        self.spec['apex_leaf']=0
        self.save()
        # No non-leaf tissue exists in this all-leaf fixture: refuse fake skeleton.
        with self.assertRaisesRegex(ValueError,'No plant-minus-leaf'):
            self.invoke(self.root/'skeleton','--stage','skeleton','--labels-dir',str(out))
        self.assertEqual(cli.read(self.root/'skeleton/run-record.json')['status'],'failed')

    def test_viewer_places_cluster_ids_at_rendered_points(self):
        output=self.root/'viewer-labels'
        output.mkdir()
        points=np.array([[0.0,0.0,0.0],[0.1,0.0,0.0],[0.0,0.1,0.0],
                         [0.005,0.0,0.0],[0.105,0.0,0.0],[0.005,0.1,0.0]])
        labels=np.array([16,16,16,17,17,17])
        labels_path=self.root/'viewer-points.npz'
        np.savez(labels_path,X=points,label=labels)
        bundle={'capture_id':'tiny-plant','leaves':[{'id':16},{'id':17}], 'source_graph':{}}
        write_viewer(output,bundle,labels_path=labels_path)
        page=(output/'viewer.html').read_text()
        data=json.JSONDecoder().raw_decode(page.split('const D=',1)[1])[0]
        for cluster in ('16','17'):
            self.assertIn(data['pointLabels'][cluster],data['points'][cluster])
        self.assertIn('const clusterAnchors=Object.entries(D.pointLabels||{})',page)
        self.assertIn("const anchor=project(item.point),text='C'+item.id",page)
        self.assertIn('drawClusterLabels()',page)

    def test_reused_labels_tampering_refused_before_output(self):
        out=self.root/'leaves'
        self.invoke(out,'--stage','leaves')
        with (out/'leaf_ids2.npz').open('ab') as f: f.write(b'changed')
        self.spec['apex_leaf']=0
        self.save()
        target=self.root/'skeleton'
        with self.assertRaisesRegex(ValueError,'Reused labels changed'):
            self.invoke(target,'--stage','skeleton','--labels-dir',str(out))
        self.assertFalse(target.exists())

    def test_changed_mask_refuses_reused_labels(self):
        out=self.root/'leaves'
        self.invoke(out,'--stage','leaves')
        Image.new('L',(32,32),0).save(self.root/'plants/photo.png')
        self.spec['apex_leaf']=0
        self.save()
        with self.assertRaisesRegex(ValueError,'source input changed'):
            self.invoke(self.root/'skeleton','--stage','skeleton','--labels-dir',str(out))

    def test_apex_explicit_and_validate_only_nonmutating(self):
        with self.assertRaisesRegex(ValueError,'apex_leaf'):
            self.invoke(self.root/'all')
        self.assertEqual(self.invoke(self.root/'leaves','--stage','leaves','--validate-only'),0)
        self.assertFalse((self.root/'leaves').exists())

    def test_existing_output_and_source_tree_protected(self):
        with self.assertRaisesRegex(ValueError,'already exists'):
            self.invoke(self.root/'images','--stage','leaves')
        with self.assertRaisesRegex(ValueError,'outside source'):
            self.invoke(self.root/'images/new-output','--stage','leaves')

    def test_bad_domain_and_missing_instance_refused(self):
        Image.new('L',(16,16),255).save(self.root/'plants/photo.png')
        with self.assertRaisesRegex(ValueError,'dimension'):
            self.invoke(self.root/'leaves','--stage','leaves')
        Image.new('L',(32,32),255).save(self.root/'plants/photo.png')
        (self.root/'instances/photo/leaf_01.png').unlink()
        with self.assertRaises(FileNotFoundError):
            self.invoke(self.root/'leaves','--stage','leaves')

    def test_distortion_and_frame_mismatch_refused(self):
        self.transform['k2']=.01
        (self.root/'transforms.json').write_text(json.dumps(self.transform))
        with self.assertRaisesRegex(ValueError,'distortion'):
            self.invoke(self.root/'leaves','--stage','leaves')
        self.transform.pop('k2')
        (self.root/'transforms.json').write_text(json.dumps(self.transform))
        (self.root/'frame_map.json').write_text(json.dumps({'wrong-frame':'photo'}))
        with self.assertRaises(KeyError):
            self.invoke(self.root/'leaves','--stage','leaves')

    def test_truncated_ply_fails_and_normal_ply_readable(self):
        self.assertEqual(len(load_ply(self.ply)),24)
        self.ply.write_bytes(self.ply.read_bytes()[:-10])
        with self.assertRaisesRegex(ValueError,'Truncated'):
            load_ply(self.ply)

    def test_parameters_and_instance_traversal_refused(self):
        self.spec['parameters']={'BIN':0}
        self.save()
        with self.assertRaisesRegex(ValueError,'positive'):
            self.invoke(self.root/'leaves','--stage','leaves')
        self.spec.pop('parameters')
        self.save()
        (self.root/'instances/photo/instances.json').write_text(json.dumps({'instances':[{'file':'../escape.png'}]}))
        with self.assertRaisesRegex(ValueError,'escape'):
            self.invoke(self.root/'leaves','--stage','leaves')

    def test_evidence_checks_real_references(self):
        out=self.root/'leaves'
        self.invoke(out,'--stage','leaves')
        bundle=cli.read(out/'organ-evidence.json')
        bundle['source_graph']={'skeleton_nodes':[{'id':0,'xyz':[0,0,0]}],
                                'skeleton_edges':[{'a':0,'b':99}]}
        with self.assertRaisesRegex(ValueError,'graph references'):
            cli.validate_evidence(bundle,out)

    def test_skeleton_result_has_json_stable_keys(self):
        import skeleton
        out=self.root/'synthetic-network'
        out.mkdir()
        stem=np.column_stack([np.zeros(41),np.zeros(41),np.linspace(0,.2,41)])
        blade=np.column_stack([np.linspace(.002,.004,20),np.zeros(20),np.full(20,.2)])
        np.savez(out/'leaf_ids2.npz',X=np.concatenate([stem,blade]),
                 label=np.array([-2]*41+[0]*20),rgb=np.array([[.1,.8,.1]]*41+[[.4,.4,.4]]*20))
        with contextlib.redirect_stdout(io.StringIO()):
            graph=skeleton.run(None,out,dict(cli.DEFAULTS),0,False)
        self.assertGreater(len(graph['skeleton_edges']),0)
        self.assertEqual(graph,json.loads(json.dumps(graph)))
        self.assertTrue(all(isinstance(key,str) for key in graph['petiole_points']))


if __name__=='__main__':
    unittest.main()
