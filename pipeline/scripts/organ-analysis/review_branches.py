#!/usr/bin/env python3
"""Apply explicit human dispositions and endpoint IDs in a new review folder."""
import argparse
import csv
from pathlib import Path
import shutil
import sys

from analyze_organs import read,write,sha,source_revision,validate_evidence
from branches import apply_review,graph_digest,prune_candidates
from viewer import write_viewer


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--decisions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--export-confirmed-pruned',action='store_true',help='Export separate derivative for explicitly confirmed-false, unguarded groups')
    args=p.parse_args(argv)
    source,out=args.review.resolve(),args.output.resolve()
    if out.exists() or source in out.parents:raise ValueError('Choose a fresh output outside the source review')
    bundle=read(source/'organ-evidence.json')
    validate_evidence(bundle,source)
    original=read(source/'branch-candidates.json')
    report=apply_review(bundle['source_graph'],original,read(args.decisions))
    pruned=None
    if args.export_confirmed_pruned:
        selected=[c['id'] for c in report['candidates'] if c['human_disposition']=='confirmed_false']
        pruned=prune_candidates(bundle['source_graph'],report,selected)
    out.mkdir(parents=True)
    shutil.copyfile(source/'leaf_ids2.npz',out/'leaf_ids2.npz')
    shutil.copyfile(source/'organ-evidence.json',out/'organ-evidence.json')
    shutil.copyfile(args.decisions,out/'human-review.json')
    write(out/'branch-candidates.json',report)
    if pruned is not None:write(out/'skeleton-confirmed-pruned.json',pruned)
    with (out/'endpoints.csv').open('x',newline='') as f:
        fields=['endpoint_id','source_node_id','review_group','status','human_disposition']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(report['endpoint_records'])
    with (out/'branch-candidates.csv').open('x',newline='') as f:
        fields=['review_label','status','human_disposition','eligible_for_pruning','edge_ids']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:c.get(k) for k in fields} for c in report['candidates']+report['retained_branches'])
    write_viewer(out,bundle,candidates=report)
    write(out/'review-provenance.json',{'source_review':str(source),'source_report_sha256':sha(source/'branch-candidates.json'),
          'source_graph_sha256':graph_digest(bundle['source_graph']),'decisions_sha256':sha(args.decisions),
          'repository':source_revision(),'code_sha256':{f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},
          'graph_unchanged':True,'command':sys.argv})
    print(str(len(report['endpoint_records']))+' uniquely identified endpoints; '+str(len(report['candidates']))+' candidate groups; '+str(out/'viewer.html'))
    return 0


if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,KeyError,OSError) as exc:
        print('review-branches: '+str(exc),file=sys.stderr);sys.exit(2)
