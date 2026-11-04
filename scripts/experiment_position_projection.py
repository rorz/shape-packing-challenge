"""28-dimensional integer-center projection with continuous angles, then 2-SAT.

The two global integer translations are fixed. Fifteen angles and the enclosing
center are eliminated continuously from the lattice distance. Adjacent-angle
2-SAT makes the final angle grid choices; full validation remains mandatory.
"""
from __future__ import annotations
import os
for key in ('VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import argparse
import json
from pathlib import Path
import time
import numpy as np
from fpylll import BKZ,GSO,IntegerMatrix,LLL
from semicircle_packing.polish import AreaConstraints,checked,load,save
from semicircle_packing.fast_mec import compute_mec_fast
from angular_rounder import round_angles


def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--input',default='experiments/branch-3.141616-3.141592.json')
    pa.add_argument('--output',type=Path,default=Path('experiments/2026-11-04/positions/best.json'))
    pa.add_argument('--seconds',type=float,default=180)
    pa.add_argument('--seed',type=int,default=110430)
    pa.add_argument('--target',type=float,default=2.94852695)
    pa.add_argument('--iterations',type=int,default=1000)
    pa.add_argument('--beta',type=float,default=.8)
    pa.add_argument('--movement',type=float,default=100)
    pa.add_argument('--angle-reg',type=float,default=1e-5)
    pa.add_argument('--position-reg',type=float,default=1e-5)
    pa.add_argument('--round-every',type=int,default=1)
    pa.add_argument('--frame-angle',type=float)
    pa.add_argument('--snap-angular',action='store_true',help='Project reflected angular coordinates onto their integer grid before residual feedback')
    pa.add_argument('--integer-boundary-angle',action='store_true',help='Keep piece14 orientation integer in the joint lattice because it controls endpoint containment')
    pa.add_argument('--integer-angle-indices',default='',help='Comma-separated additional piece indices whose angles stay in the integer lattice')
    pa.add_argument('--mincost-projection',action='store_true',help='Complete center proposals using the globally least-violating adjacent-angle assignment')
    pa.add_argument('--max-restarts',type=int,default=20,help='Stop after repeated divergent local projections')
    args=pa.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.mincost_projection:
        from angular_min_cost import minimize_angles
    rng=np.random.default_rng(args.seed);original=load(args.input)
    original[:,:2]-=compute_mec_fast(original)[:2]
    best_score=checked(load('solution.json')).score;best_violation=float('inf')
    started=time.monotonic();frames=trials=rounded=assignments=restarts=0
    integer_angles={int(v) for v in args.integer_angle_indices.split(',') if v}
    if args.integer_boundary_angle:integer_angles.add(14)
    if not integer_angles.issubset(set(range(15))):pa.error('integer angle indices must be from0to14')
    position=np.array([i for i in range(3,45) if i%3!=2]+[3*i+2 for i in sorted(integer_angles)])
    angles=np.array([i for i in range(2,45,3) if i//3 not in integer_angles],dtype=int)
    na=len(angles)
    d=len(position);scale=1e8;identity=max(1,round(args.position_reg*scale));reg=identity/scale
    while time.monotonic()-started<args.seconds:
        angle=rng.uniform(-np.pi,np.pi) if args.frame_angle is None else args.frame_angle
        co,si=np.cos(angle),np.sin(angle);p=original.copy()
        p[:,:2]=p[:,:2]@np.array([[co,si],[-si,co]])
        p[:,2]=np.arctan2(np.sin(p[:,2]+angle),np.cos(p[:,2]+angle))
        c=AreaConstraints(p);g,j=c.evaluate(np.r_[p.ravel(),args.target]);active=g<1e-6
        J=j[active,:45];U,_,_=np.linalg.svd(J,full_matrices=True)
        weights=U[:,-1];weights*=np.sign(weights.sum())
        if weights.min()<=0:continue
        m=len(weights);anchor=p.copy();anchor[:,:2]=np.round(anchor[:,:2],6)
        for piece in integer_angles:anchor[piece,2]=np.round(anchor[piece,2],6)
        error=(anchor-p).ravel()/1e-6;offset=weights*(g[active]/1e-6+J@error)
        W=weights[:,None]*J
        center=-weights[:,None]*np.column_stack([J[:,0::3].sum(axis=1),J[:,1::3].sum(axis=1)])
        flat=np.zeros((2,15));flat[0,1]=.1;flat[0,5]=-.1;flat[1,6]=.1;flat[1,10]=-.1
        full_flat=np.zeros((2,45));full_flat[:,2::3]=flat
        K=np.vstack([np.zeros((d,na+2)),np.column_stack([np.eye(na)*args.angle_reg,np.zeros((na,2))]),
            np.column_stack([W[:,angles],center]),np.column_stack([flat[:,angles//3],np.zeros((2,2))])])
        inverse=np.linalg.pinv(K,rcond=1e-12)
        embed=np.vstack([np.eye(d)*reg,np.zeros((na,d)),W[:,position],full_flat[:,position]])
        projected=embed-K@(inverse@embed)
        B=IntegerMatrix.from_matrix(np.rint(projected.T*scale).astype(np.int64).tolist())
        LLL.reduction(B,delta=.99);BKZ.reduction(B,BKZ.Param(block_size=20,max_loops=2))
        M=GSO.Mat(B,update=True)
        moves=np.rint(np.array([list(row) for row in B],dtype=float)[:,:d]/identity).astype(np.int64)
        sl=slice(d+na,d+na+m)
        x=np.r_[rng.normal(0,args.movement*.1,d)*reg,np.zeros(na),np.maximum(-offset,0),np.zeros(2)]
        seen=set();frames+=1
        for iteration in range(args.iterations):
            if time.monotonic()-started>args.seconds:break
            t=x-K@(inverse@x)
            coefficients=M.babai(np.rint(t*scale).astype(np.int64).tolist())
            n=np.rint(np.array(coefficients,dtype=float)@moves)
            if not np.isfinite(n).all() or np.abs(n).max()>20*args.movement:
                restarts+=1
                print(json.dumps(dict(event='restart_diverged',frames=frames,trials=trials,
                    maximum_step=float(np.abs(n).max()),seconds=time.monotonic()-started)),flush=True)
                break
            lp=embed@n;floating=inverse@(x-lp);lp+=K@floating
            candidate=anchor.copy();flat_p=candidate.ravel();flat_p[position]+=n*1e-6
            flat_p[angles]+=floating[:na]*1e-6
            candidate[:,:2]=np.round(candidate[:,:2],6)
            candidate[:,2]=np.arctan2(np.sin(candidate[:,2]),np.cos(candidate[:,2]))
            if args.mincost_projection:
                candidate,cost_info=minimize_angles(candidate,args.target,area_scale=c.scale)
                diff=candidate.ravel()[angles]-anchor.ravel()[angles]
                floating[:na]=np.arctan2(np.sin(diff),np.cos(diff))/1e-6
                floating[na:]=np.array(cost_info['center'])/1e-6
                lp=embed@n+K@floating
            centered=candidate.copy();centered[:,:2]-=floating[na:]*1e-6
            actual,_=c.evaluate(np.r_[centered.ravel(),args.target]);trials+=1
            local_offset=weights*actual[active]/1e-6-lp[sl]
            reflected=2*lp-x;projection=reflected.copy()
            projection[:d]=np.clip(projection[:d],-args.movement*reg,args.movement*reg)
            projection[d:d+na]=np.clip(projection[d:d+na],-args.movement*args.angle_reg,args.movement*args.angle_reg)
            if args.snap_angular:
                angular_units=p.ravel()[angles]/1e-6+projection[d:d+na]/args.angle_reg
                projection[d:d+na]=(np.rint(angular_units)-p.ravel()[angles]/1e-6)*args.angle_reg
            projection[sl]=np.maximum(projection[sl],-local_offset)
            projection[-2:]=np.clip(projection[-2:],-.15,.15)
            x+=args.beta*(projection-lp)
            violation=-float(actual.min())
            if violation<best_violation:
                best_violation=violation
                args.output.with_name(args.output.stem+'_continuous.json').write_text(json.dumps([dict(zip(('x','y','theta'),map(float,row))) for row in candidate],indent=2)+'\n')
                print(json.dumps(dict(event='nearest_continuous',frames=frames,trials=trials,violation=violation,seconds=time.monotonic()-started)),flush=True)
            if iteration%args.round_every:continue
            if args.mincost_projection:
                if actual[:105].min() < -1e-9:continue
            elif violation>2e-6:continue
            rounding_center=np.array(compute_mec_fast(candidate)[:2])
            key=(tuple(n.astype(int))+tuple(np.floor(candidate[:,2]*1e6).astype(int))+
                 tuple(np.rint(rounding_center*1e13).astype(np.int64)))
            if key in seen:continue
            seen.add(key);rounded+=1
            integer_candidate=(candidate if args.mincost_projection else
                round_angles(candidate,min(best_score-1e-12,args.target)))
            if integer_candidate is None:continue
            if args.mincost_projection and compute_mec_fast(integer_candidate)[2]>=best_score:continue
            assignments+=1
            result=checked(integer_candidate)
            print(json.dumps(dict(event='reference',valid=result.valid,score=result.score,errors=result.errors,rounded=rounded,seconds=time.monotonic()-started)),flush=True)
            if result.valid and result.score<best_score:
                best_score=result.score;save(args.output,integer_candidate)
                if best_score<2.9485269238856113:break
        if best_score<2.9485269238856113 or restarts>=args.max_restarts:break
    print(json.dumps(dict(event='finished',frames=frames,trials=trials,rounded=rounded,assignments=assignments,restarts=restarts,
        best_score=best_score,best_violation=best_violation,seconds=time.monotonic()-started)),flush=True)


if __name__=='__main__':main()
