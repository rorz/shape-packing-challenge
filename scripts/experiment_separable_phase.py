"""Same cached phase runner, with conservative separable support constraints."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import experiment_conflict_master as master
import experiment_phase_bundle_resume as runner
from separable_support import Inserter,audit


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--domains',required=True)
    parser.add_argument('--anchors',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--factor-preflight',action='store_true')
    args,_=parser.parse_known_args()
    archive=np.load(args.domains);domains=[archive[f'p{i}']for i in range(len(archive.files))]
    archive=np.load(args.anchors);cuts=runner.all_phase_supports(domains,archive['anchors'],archive['anchor_circles'])
    start=time.monotonic();report=audit(domains,cuts);report['audit_seconds']=time.monotonic()-start
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).with_suffix('.factor-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('FACTOR_AUDIT',json.dumps(report),flush=True)
    if args.factor_preflight:return
    del cuts
    insert=Inserter(domains);original=master.add_support_cut;master.add_support_cut=insert
    try:runner.main()
    finally:
        master.add_support_cut=original
        Path(args.output).with_suffix('.factor-model.json').write_text(json.dumps(insert.stats,indent=2)+'\n')


if __name__=='__main__':main()
