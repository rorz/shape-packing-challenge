"""Bounded fixed-XY exact angular rescue, with an explicitly external control."""
from __future__ import annotations
import argparse,hashlib,itertools,json
from pathlib import Path
import numpy as np
import experiment_conflict_master as master
from interval_pair_relation import add_allowed_interval_pair
from semicircle_packing.geometry import canonicalize_theta
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.polish import load,polygon_areas,save


def normalized(path):
    p=load(path);p[:,:2]=np.round(p[:,:2],6)
    p[:,2]=[canonicalize_theta(t)for t in p[:,2]]
    if any(canonicalize_theta(t)!=t for t in p[:,2]):
        raise ValueError('This rescue requires stable canonical angle representatives')
    return p


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--control',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--jitter',type=int,default=20)
    parser.add_argument('--seconds',type=float,default=45)
    parser.add_argument('--target',type=float,default=2.9485269629)
    args=parser.parse_args();output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    source=Path(args.input).read_bytes();output.with_suffix('.raw-input.json').write_bytes(source)
    p=normalized(output.with_suffix('.raw-input.json'));save(output.with_suffix('.canonical-input.json'),p)
    control=normalized(args.control);control_domains=[row[None]for row in control]
    control_tables,_=master.exact_tables(control_domains,band_screen=True,local_loss=True)
    control_result=master.positive_control(control_domains,control_tables)
    metadata=dict(source_sha256=hashlib.sha256(source).hexdigest(),source=str(args.input),
                  external_positive_control=control_result,external_control_input=str(args.control),
                  initial_mec=compute_mec_fast(p),fixed_xy=True,jitter=args.jitter,target=args.target,
                  initial_bad_pairs=[dict(pair=[i,j],area=float(a))for(i,j),a in zip(itertools.combinations(range(15),2),polygon_areas(p))if a>1e-6],
                  note='Invalid included seed is not itself a positive control; the separately checked valid input verifies the same engine')
    output.with_suffix('.audit.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('AUDIT',json.dumps(metadata),flush=True)
    domains,_=master.correlated_domains(p,directions=0,jitter=args.jitter,max_domain=2*args.jitter+1)
    original=master.add_pair_constraint;master.add_pair_constraint=add_allowed_interval_pair
    try:
        master.solve(domains,args.target,output,seconds=args.seconds,support_cuts=True,
                     check_positive=False,band_screen=True,local_loss=True)
    finally:master.add_pair_constraint=original


if __name__=='__main__':main()
