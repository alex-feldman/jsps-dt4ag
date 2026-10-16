#!/usr/bin/env python3
"""CPU leaf-instance lifting and provisional skeleton extraction. See README.md."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

VERSION = '0.3.0-alpha.7'
DEFAULTS = {'opacity_cutoff':0.3,'RD':8,'RM':4,'ZTOL':0.006,'MAXR':8,'MIN_G':15,
            'JOIN':0.35,'MERGE':0.5,'TOUCH':0.004,'TOUCH_SHARE':0.1,
            'LINK':0.009,'BIN':0.008,'LINP':0.55,'GREEN':0.03,'TIPGAP':0.006}
PATH_KEYS = ['ply','transforms','dataparser_transforms','frame_map','images','plant_masks','leaf_instances']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    with Path(path).open() as f:
        return json.load(f)


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def source_revision():
    root = Path(__file__).resolve().parents[3]
    def git(*args):
        result = subprocess.run(['git','-C',str(root),*args],capture_output=True,text=True,timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    return {'root':str(root),'commit':git('rev-parse','HEAD'),'status':git('status','--short')}


def spec_from(path, stage):
    path = Path(path).resolve()
    spec = read(path)
    unknown = set(spec)-{'capture_id','inputs','parameters','apex_leaf','plots','annotations'}
    if unknown or not isinstance(spec.get('capture_id'), str) or not spec['capture_id'].strip():
        raise ValueError('Expected capture_id and known config keys; unknown: '+str(sorted(unknown)))
    inputs = dict(spec['inputs'])
    if set(inputs)-set(PATH_KEYS)-{'image_extension','mask_extension'}:
        raise ValueError('Unknown input keys')
    for key in PATH_KEYS:
        p = Path(inputs[key])
        inputs[key] = str((path.parent/p).resolve())
        if not Path(inputs[key]).exists():
            raise ValueError('Missing input '+key+': '+inputs[key])
    inputs.setdefault('image_extension','.jpg')
    inputs.setdefault('mask_extension','.png')
    for key in ['image_extension','mask_extension']:
        if not isinstance(inputs[key],str) or not inputs[key].startswith('.') or '/' in inputs[key] or '\\' in inputs[key]:
            raise ValueError('Expected filename extension for '+key)
    params = dict(DEFAULTS)
    if set(spec.get('parameters',{}))-set(params):
        raise ValueError('Unknown analysis parameter')
    params.update(spec.get('parameters',{}))
    for key,value in params.items():
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value):
            raise ValueError('Invalid parameter '+key)
        if key in ['RD','RM','MIN_G','MAXR'] and (not isinstance(value,int) or value < 1):
            raise ValueError('Expected positive integer '+key)
        if key not in ['GREEN'] and value <= 0:
            raise ValueError('Expected positive '+key)
        if key in ['opacity_cutoff','JOIN','MERGE','TOUCH_SHARE','LINP'] and not 0 < value < 1:
            raise ValueError('Expected fraction in (0,1): '+key)
    if not isinstance(spec.get('plots',False),bool):
        raise ValueError('plots must be boolean')
    if stage != 'leaves' and (type(spec.get('apex_leaf')) is not int or spec['apex_leaf'] < 0):
        raise ValueError('Select apex_leaf explicitly after inspecting this model\'s leaf IDs, or run --stage leaves first')
    spec['inputs'], spec['parameters'] = inputs, params
    return spec


def preflight(spec, labels_dir=None):
    from PIL import Image
    from geometry import Geometry
    inputs = spec['inputs']
    geom = Geometry(inputs)
    fmap = read(inputs['frame_map'])
    if not fmap or len(set(fmap.values())) != len(fmap):
        raise ValueError('Frame map must be nonempty and one-to-one')
    if any(not isinstance(n,str) or Path(n).name != n or n in ['.','..'] or '\\' in n for n in fmap.values()):
        raise ValueError('Frame map names must be plain source-image stems')
    cams = geom.cameras(fmap)
    if len(cams) != len(fmap) or {c['frame'] for c in cams} != set(fmap):
        raise ValueError('Camera and frame-map memberships differ')
    files = {Path(inputs[k]) for k in PATH_KEYS if Path(inputs[k]).is_file()}
    for cam in cams:
        expected = (cam['w'],cam['h'])
        if min(expected) < max(spec['parameters']['RD'],spec['parameters']['RM']) or any(v % spec['parameters']['RD'] or v % spec['parameters']['RM'] for v in expected):
            raise ValueError('Camera image dimensions must be divisible by RD and RM')
        paths = [geom.image_path(cam['name']),geom.plant_path(cam['name'])]
        directory = Path(inputs['leaf_instances'])/cam['name']
        sidecar = directory/'instances.json'
        files.add(sidecar)
        for item in read(sidecar)['instances']:
            filename = item['file']
            if Path(filename).name != filename or '\\' in filename:
                raise ValueError('Instance filename must not escape its directory')
            paths.append(directory/filename)
        for p in paths:
            with Image.open(p) as image:
                if image.size != expected:
                    raise ValueError('Photo/mask dimension differs from raw camera domain: '+str(p))
            files.add(p)
    if labels_dir is not None:
        for name in ['leaf_ids2.npz','leaf_ids2.json','frame_map.json','run-record.json']:
            files.add(labels_dir/name)
        prior = read(labels_dir/'run-record.json')
        if prior['status'] != 'complete' or prior['spec']['inputs'] != inputs or prior['spec']['parameters'] != spec['parameters']:
            raise ValueError('Reused labels must come from a completed command with the same inputs and parameters')
        for p,digest in prior['input_sha256'].items():
            if sha(p) != digest:
                raise ValueError('A reused-label source input changed: '+p)
        for name in ['leaf_ids2.npz','leaf_ids2.json','frame_map.json']:
            if sha(labels_dir/name) != prior['output_sha256'][name]:
                raise ValueError('Reused labels changed: '+name)
    return geom, {str(p):sha(p) for p in sorted(files)}


def evidence(spec, out, graph):
    import numpy as np
    data = read(out/'leaf_ids2.json')
    with np.load(out/'leaf_ids2.npz') as arrays:
        leaf_ids = sorted(int(v) for v in np.unique(arrays['label']) if v>=0)
        counts = {i:int((arrays['label']==i).sum()) for i in leaf_ids}
    supported_ids = {row['leaf_id'] for row in data['clusters']}
    annotations = spec.get('annotations',{})
    if not isinstance(annotations,dict) or set(annotations)-{str(i) for i in leaf_ids}:
        raise ValueError('Annotation IDs must refer to this output\'s leaf clusters')
    result = {'schema':'organ-evidence-v1','capture_id':spec['capture_id'],'units':'scale_free_scene_units',
        'point_labels':{'file':'leaf_ids2.npz','sha256':sha(out/'leaf_ids2.npz'),
                        'id_scope':'array row in this hashed model and filtering output; not a cross-run ID'},
        'source_ply_sha256':sha(spec['inputs']['ply']), 'leaf_instance_associations':data,
        'leaves':[{'id':i,'point_count':counts[i],'association_support':'supported' if i in supported_ids else 'small_unreviewed_cluster',
                   'annotation':annotations.get(str(i)),
                   'annotation_status':'user_supplied' if str(i) in annotations else 'unreviewed',
                   'geometry_status':'candidate'} for i in leaf_ids],
        'source_graph':graph,'camera_inputs':{k:sha(spec['inputs'][k]) for k in ['transforms','dataparser_transforms','frame_map']},
        'limits':['No measured scale/radii','Sampling nodes are not phytomer nodes',
                  'Zero-length fallback does not establish absent petiole',
                  'Apex zero-length petiole follows from construction','Strings/tendrils remain unresolved']}
    if graph:
        for leaf in result['leaves']:
            row = next((r for r in graph['leaves'] if r['leaf_id']==leaf['id']),None)
            leaf['geometry_status'] = 'unresolved_no_path' if row is None or (row['petiole_network_length']==0 and leaf['id']!=spec['apex_leaf']) else 'algorithm_candidate'
    return result


def validate_evidence(bundle, out):
    if sha(out/bundle['point_labels']['file']) != bundle['point_labels']['sha256']:
        raise ValueError('Point-label integrity failure')
    graph = bundle['source_graph']
    if graph:
        ids = [n['id'] for n in graph['skeleton_nodes']]
        if len(ids)!=len(set(ids)) or any(e['a'] not in ids or e['b'] not in ids for e in graph['skeleton_edges']):
            raise ValueError('Invalid graph references')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True,help='Explicit input/parameter JSON; see example.json')
    parser.add_argument('--output',type=Path,required=True,help='New output directory; existing paths are refused')
    parser.add_argument('--stage',choices=['all','leaves','skeleton'],default='all')
    parser.add_argument('--labels-dir',type=Path,help='Completed leaf-stage output for skeleton-only execution')
    parser.add_argument('--validate-only',action='store_true',help='Validate/hash source inputs without creating output')
    args = parser.parse_args(argv)
    if (args.stage=='skeleton') != (args.labels_dir is not None):
        parser.error('--labels-dir is required only with --stage skeleton')
    out = args.output.resolve()
    if out.exists():
        raise ValueError('Output already exists; choose a fresh directory: '+str(out))
    spec = spec_from(args.config,args.stage)
    # Never allow analysis outputs to become children of protected source trees.
    for key in ['images','plant_masks','leaf_instances']:
        if Path(spec['inputs'][key]) in out.parents or out == Path(spec['inputs'][key]):
            raise ValueError('Output must be outside source image/mask directories')
    labels_dir = args.labels_dir.resolve() if args.labels_dir else None
    geom, hashes = preflight(spec, labels_dir)
    if args.validate_only:
        print(json.dumps({'status':'validated','source_files':len(hashes),'stage':args.stage}))
        return 0
    out.mkdir(parents=True,exist_ok=False)
    record = {'version':VERSION,'status':'running','stage':args.stage,'spec':spec,'command':sys.argv,
              'working_directory':str(Path.cwd()),'repository':source_revision(),
              'started_utc':datetime.now(timezone.utc).isoformat(),'input_sha256':hashes,
              'code_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
              'versions':{k:importlib.metadata.version(k) for k in ['numpy','scipy','networkx','Pillow']}}
    write(out/'input-manifest.json',record)
    try:
        shutil.copyfile(spec['inputs']['frame_map'],out/'frame_map.json')
        if labels_dir:
            for name in ['leaf_ids2.npz','leaf_ids2.json']:
                shutil.copyfile(labels_dir/name,out/name)
        else:
            import leaf_lift
            leaf_lift.run(geom,out,spec['parameters'])
        graph = None
        if args.stage!='leaves':
            import skeleton
            graph = skeleton.run(geom,out,spec['parameters'],spec['apex_leaf'],spec.get('plots',False))
        bundle = evidence(spec,out,graph)
        validate_evidence(bundle,out)
        write(out/'organ-evidence.json',bundle)
        restored = read(out/'organ-evidence.json')
        validate_evidence(restored,out)
        if restored != bundle:
            raise ValueError('Evidence JSON round trip failed')
        from viewer import write_viewer
        write_viewer(out,bundle)
        if any(sha(p)!=digest for p,digest in hashes.items()):
            raise ValueError('Source input changed during analysis')
        record['status']='complete'
        record['output_sha256']={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    except Exception as exc:
        record.update(status='failed',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        record['ended_utc']=datetime.now(timezone.utc).isoformat()
        write(out/'run-record.json',record)
    print('Complete: '+str(out/'organ-evidence.json'))
    return 0


if __name__=='__main__':
    try:
        sys.exit(main())
    except (ValueError,KeyError,OSError,TypeError) as exc:
        print('organ-analysis: '+str(exc),file=sys.stderr)
        sys.exit(2)
