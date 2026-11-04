"""Replay the independent algebra, angle-seam, integerization, and 2-SAT probes."""
from __future__ import annotations

import os
for variable in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from angular_rounder import _pair_indices, solve_binary_constraints
from experiment_oblique_kbest import continuous_seed
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.polish import AreaConstraints, load


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",default="solution.json")
    parser.add_argument("--output",type=Path,default=Path("experiments/2026-11-04/model-audit-replay.json"))
    args=parser.parse_args();p=load(args.input)
    report={"input":args.input,"input_sha256":hashlib.sha256(Path(args.input).read_bytes()).hexdigest()}
    seed,radius,initial_info=continuous_seed(p)
    angle=1.;cs,sn=np.cos(angle),np.sin(angle);rotation=np.array([[cs,sn],[-sn,cs]])
    seed[:,:2]=seed[:,:2]@rotation
    seed[:,2]=np.arctan2(np.sin(seed[:,2]+angle),np.cos(seed[:,2]+angle))
    known=p.copy();known[:,:2]-=compute_mec_fast(known)[:2]
    known[:,:2]=known[:,:2]@rotation;known[:,2]+=angle
    c=AreaConstraints(seed);g,j=c.evaluate(np.r_[seed.ravel(),2.94852695]);active=g<1e-7;J=j[active,:45]
    raw=known-seed;wrapped=raw.copy();wrapped[:,2]=np.arctan2(np.sin(wrapped[:,2]),np.cos(wrapped[:,2]))
    old=g[active]+J@raw.ravel();correct=g[active]+J@wrapped.ravel()
    report["angle_seam"]={"angle":angle,"crossed":np.flatnonzero(abs(raw[:,2])>np.pi).tolist(),
        "max_guidance_error":float(abs(old-correct).max()),"wrapped_range":[float(correct.min()),float(correct.max())],
        "continuous_radius":radius,"continuous_info":initial_info}
    cx,cy,r=compute_mec_fast(p);q=p.copy();q[:,:2]-=[cx,cy]
    _,j=AreaConstraints(p).evaluate(np.r_[q.ravel(),r])
    C=-np.column_stack([j[:,:45:3].sum(1),j[:,1:45:3].sum(1)])*1e-10
    coefficient_error=np.abs(np.rint(C*1e16)-C*1e16)
    report["cpsat_rounding"]={"center_grid_nm":.1,"max_center_grid_error_nm":float(np.sqrt(2)*.05),
        "rows":[dict(movement_micro=movement,max_row_rounding_error_nm=float(coefficient_error.sum(1).max()*movement*10000/1e16*1e9))for movement in (50,100,200)]}
    q=load("experiments/branch-3.141616-3.141592.json");q[:,:2]-=compute_mec_fast(q)[:2]
    g,j=AreaConstraints(q).evaluate(np.r_[q.ravel(),2.9485269238]);active=g<1e-6;J=j[active,:45]
    U,_,_=np.linalg.svd(J,full_matrices=True);w=U[:,-1];w*=np.sign(w.sum())
    C=-w[:,None]*np.column_stack([J[:,0::3].sum(1),J[:,1::3].sum(1)]);inverse=np.linalg.pinv(C);P=np.eye(len(w))-C@inverse
    rng=np.random.default_rng(20261104);n=rng.integers(-10,11,43);x=rng.normal(0,.1,len(w));WJ=w[:,None]*J[:,2:]
    center=inverse@(x-WJ@n);projected=WJ@n+C@center
    report["difference_map"]={"seed":20261104,"center_rank":int(np.linalg.matrix_rank(C)),
        "orthogonality_error":float(np.linalg.norm(C.T@(projected-x))),"projector_error":float(np.linalg.norm(P@P-P))}
    rng=np.random.default_rng(110430);cases=0
    for count in range(1,7):
        for _ in range(300):
            unary=rng.random((count,2))>.15;compat=rng.random((count,count,2,2))>.15
            answer=solve_binary_constraints(unary,compat);witness=None
            for bits in itertools.product((0,1),repeat=count):
                if all(unary[i,bits[i]]for i in range(count)) and all(compat[i,k,bits[i],bits[k]]for i in range(count)for k in range(i+1,count)):
                    witness=bits;break
            assert (answer is None)==(witness is None)
            if answer is not None:
                assert all(unary[i,answer[i]]for i in range(count))
                assert all(compat[i,k,answer[i],answer[k]]for i in range(count)for k in range(i+1,count))
            cases+=1
    for count in range(2,16):
        pairs,tables=_pair_indices(count);lookup={pair:i for i,pair in enumerate(itertools.combinations(range(2*count),2))}
        for (i,k),table in zip(pairs,tables):
            for a,b in itertools.product((0,1),repeat=2):assert table[a,b]==lookup[(2*i+a,2*k+b)]
    report["angular_rounder"]={"seed":110430,"bruteforce_cases":cases,"sizes":[1,2,3,4,5,6],"pair_index_sizes":list(range(2,16)),"all_passed":True}
    report["source_sha256"]={str(path):hashlib.sha256(path.read_bytes()).hexdigest()for path in map(Path,[
        "scripts/audit_search_models.py","scripts/experiment_cpsat.py","scripts/experiment_oblique_kbest.py",
        "scripts/experiment_difference_map.py","scripts/experiment_position_projection.py","scripts/angular_rounder.py"])}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
