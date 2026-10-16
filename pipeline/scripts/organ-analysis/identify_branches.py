#!/usr/bin/env python3
"""Identify leaf-free terminal-subtree candidates from an existing organ output."""
import argparse
import csv
from pathlib import Path
import shutil
import sys

import numpy as np
from analyze_organs import read, write, sha, validate_evidence, source_revision
from branches import identify_candidates
from viewer import write_viewer


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset',type=Path,required=True,help='Completed organ-analysis output directory')
    p.add_argument('--output',type=Path,required=True,help='Fresh additive review directory')
    p.add_argument('--tip-gap',type=float,help='Leaf proximity guard, scene units; default from recorded TIPGAP')
    args=p.parse_args(argv)
    asset,out=args.asset.resolve(),args.output.resolve()
    if out.exists() or asset in out.parents:
        raise ValueError('Choose a fresh output directory outside the source asset')
    record=read(asset/'run-record.json')
    if record['status']!='complete':
        raise ValueError('Source organ analysis is not complete')
    bundle=read(asset/'organ-evidence.json')
    if bundle['source_graph'] is None:
        raise ValueError('Source asset has no skeleton')
    validate_evidence(bundle,asset)
    for name in ['organ-evidence.json','leaf_ids2.npz']:
        if sha(asset/name)!=record['output_sha256'][name]:
            raise ValueError('Source asset changed since completion: '+name)
    with np.load(asset/'leaf_ids2.npz') as arrays:
        report=identify_candidates(bundle['source_graph'],arrays['X'],arrays['label'],
                    args.tip_gap if args.tip_gap is not None else record['spec']['parameters']['TIPGAP'])
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(asset/'leaf_ids2.npz',out/'leaf_ids2.npz')
    write(out/'organ-evidence.json',bundle)
    write(out/'branch-candidates.json',report)
    with (out/'branch-candidates.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['review_label','id','attachment_node_id','terminal_node_ids','edge_ids','length_scene_units','eligible_for_pruning','guarded_edge_ids','status','human_disposition'])
        writer.writeheader()
        for c in report['candidates']:
            writer.writerow({k:c[k] for k in writer.fieldnames})
    write_viewer(out,bundle,candidates=report)
    write(out/'review-provenance.json',{'source_asset':str(asset),'source_graph_sha256':report['source_graph_sha256'],
          'source_evidence_sha256':sha(asset/'organ-evidence.json'),'source_points_sha256':sha(asset/'leaf_ids2.npz'),
          'code_sha256':{f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},
          'repository':source_revision(),'command':sys.argv,'original_graph_preserved':True})
    print(str(len(report['candidates']))+' candidate leaf-free subtrees; original preserved. Review: '+str(out/'viewer.html'))
    return 0


if __name__=='__main__':
    try:
        sys.exit(main())
    except (ValueError,KeyError,OSError) as exc:
        print('identify-branches: '+str(exc),file=sys.stderr)
        sys.exit(2)
