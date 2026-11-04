"""Residual-feedback projection between an integer lattice and contact inequalities.

Douglas–Rachford uses x <- x + beta*(P_C(2*P_L(x)-x)-P_L(x)).
P_L is approximate Babai decoding; P_C clips contact violations. This is a
heuristic for a nonconvex feasibility problem, with exact geometric acceptance.
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
from fpylll import BKZ, GSO, IntegerMatrix, LLL
from semicircle_packing.polish import AreaConstraints, checked, load, save
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.geometry import canonicalize_theta


def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--input',default='experiments/branch-3.141616-3.141592.json')
    pa.add_argument('--output',type=Path,default=Path('experiments/2026-11-04/difference-map/best.json'))
    pa.add_argument('--incumbent',default='solution.json')
    pa.add_argument('--seconds',type=float,default=120)
    pa.add_argument('--seed',type=int,default=110411)
    pa.add_argument('--target',type=float,default=2.9485269238)
    pa.add_argument('--iterations',type=int,default=1000)
    pa.add_argument('--beta',type=float,default=.8)
    pa.add_argument('--regularization',type=float,default=1e-5)
    pa.add_argument('--movement',type=float,default=100)
    pa.add_argument('--frame-angle',type=float)
    pa.add_argument('--block-size',type=int,default=20)
    pa.add_argument('--flat-width',type=float,default=.15)
    pa.add_argument('--nonlinear',action='store_true',help='Correct the halfspace offsets at every decoded placement')
    pa.add_argument('--lock-flat',action='store_true',help='Snap frame-dependent flat differences continuously, then eliminate their integer freedoms')
    args=pa.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed);original=load(args.input)
    original[:,:2]-=compute_mec_fast(original)[:2]
    started=time.monotonic();trials=actuals=frames=0;best_violation=float('inf')
    incumbent=checked(load(args.incumbent));best_score=incumbent.score
    variable=np.array([i for i in range(2,45) if not args.lock_flat or i not in (17,32)])
    d=len(variable);reg=args.regularization
    A=np.zeros((45,d));A[variable,np.arange(d)]=1
    if args.lock_flat:
        A[17,np.flatnonzero(variable==5)[0]]=1
        A[32,np.flatnonzero(variable==20)[0]]=1
    D=np.zeros((2,46));D[0,5]=1;D[0,17]=-1;D[1,20]=1;D[1,32]=-1
    scale=1e8;identity=max(1,round(reg*scale));reg=identity/scale
    while time.monotonic()-started<args.seconds:
        angle=rng.uniform(-np.pi,np.pi) if args.frame_angle is None else args.frame_angle
        co,si=np.cos(angle),np.sin(angle);p=original.copy()
        p[:,:2]=p[:,:2]@np.array([[co,si],[-si,co]])
        p[:,2]=np.arctan2(np.sin(p[:,2]+angle),np.cos(p[:,2]+angle))
        constraint=AreaConstraints(p)
        if args.lock_flat:
            continuous_radius=compute_mec_fast(p)[2]
            z=np.r_[p.ravel(),continuous_radius]
            cg,cj=constraint.evaluate(z);contacts=cg<1e-7
            flat_values=np.round(D@z,6)
            for _ in range(8):
                cg,cj=constraint.evaluate(z)
                residual=np.r_[cg[contacts],D@z-flat_values]
                if np.abs(residual).max()<2e-12:break
                z-=np.linalg.lstsq(np.vstack([cj[contacts],D]),residual,rcond=1e-12)[0]
            if np.abs(residual).max()>1e-9 or np.abs(z[:-1].reshape(15,3)[:,2]).max()>np.pi:continue
            p=z[:-1].reshape(15,3)
            constraint=AreaConstraints(p)
        g,j=constraint.evaluate(np.r_[p.ravel(),args.target])
        active=g<1e-6;J=j[active,:45]
        U,_,_=np.linalg.svd(J,full_matrices=True);weights=U[:,-1];weights*=np.sign(weights.sum())
        if weights.min()<=0:continue
        m=len(weights);anchor=np.round(p,6)
        anchor[:,2]=[canonicalize_theta(t) for t in anchor[:,2]]
        if args.lock_flat:
            anchor[5,2]=np.round(anchor[1,2]-flat_values[0],6)
            anchor[10,2]=np.round(anchor[6,2]-flat_values[1],6)
            if np.abs(anchor[:,2]).max()>np.pi:continue
        error=(anchor-p).ravel()/1e-6;error[2::3]=np.arctan2(np.sin((anchor-p)[:,2]),np.cos((anchor-p)[:,2]))/1e-6
        offset=weights*(g[active]/1e-6+J@error)
        WJ=weights[:,None]*(J@A)
        C=-weights[:,None]*np.column_stack([J[:,0::3].sum(axis=1),J[:,1::3].sum(axis=1)])
        inverse=np.linalg.pinv(C);P=np.eye(m)-C@inverse
        flat=np.zeros((2,45))
        for row,(i,k) in enumerate([(1,5),(6,10)]):flat[row,3*i+2]=.1;flat[row,3*k+2]=-.1
        flat_target=-flat@error;F=flat@A
        basis=np.column_stack([np.eye(d)*identity,np.round((P@WJ).T*scale),np.round(F.T*scale)]).astype(np.int64)
        B=IntegerMatrix.from_matrix(basis.tolist());LLL.reduction(B,delta=.99)
        if args.block_size:BKZ.reduction(B,BKZ.Param(block_size=args.block_size,max_loops=2))
        M=GSO.Mat(B,update=True)
        moves=np.rint(np.array([list(row) for row in B],dtype=float)[:,:d]/identity).astype(np.int64)
        x=np.r_[rng.normal(0,args.movement*.2,d)*reg,np.maximum(-offset,0),flat_target]
        seen=set();frames+=1
        for iteration in range(args.iterations):
            if time.monotonic()-started>args.seconds:break
            target=x.copy();target[d:d+m]=P@target[d:d+m]
            coeff=M.babai(np.rint(target*scale).astype(np.int64).tolist())
            n=np.array(coeff,dtype=float)@moves
            lp=np.r_[reg*n,WJ@n,F@n]
            center=inverse@(x[d:d+m]-lp[d:d+m]);lp[d:d+m]+=C@center
            delta=A@n
            candidate=anchor+delta.reshape(15,3)*1e-6
            candidate[:,2]=[canonicalize_theta(t) for t in candidate[:,2]]
            centered=candidate.copy();centered[:,:2]-=center*1e-6
            actual=None
            local_offset=offset
            if args.nonlinear:
                actual,_=constraint.evaluate(np.r_[centered.ravel(),args.target]);actuals+=1
                local_offset=weights*actual[active]/1e-6-lp[d:d+m]
            reflected=2*lp-x;projected=reflected.copy()
            projected[:d]=np.clip(projected[:d],-args.movement*reg,args.movement*reg)
            projected[d:d+m]=np.maximum(projected[d:d+m],-local_offset)
            projected[d+m:]=np.clip(projected[d+m:],flat_target-args.flat_width,flat_target+args.flat_width)
            x+=args.beta*(projected-lp);trials+=1
            key=tuple(n.astype(int))
            if key in seen:continue
            seen.add(key)
            if actual is None:
                actual,_=constraint.evaluate(np.r_[centered.ravel(),args.target]);actuals+=1
            violation=-float(actual.min())
            if violation<best_violation:
                best_violation=violation;save(args.output.with_name(args.output.stem+'_nearest.json'),candidate)
                joint=np.rint(moves@A.T).astype(int)
                np.savez(args.output.with_suffix('.npz'),placements=candidate,origin=p,moves=joint.reshape(d,15,3),target=args.target)
                print(json.dumps(dict(event='nearest',frames=frames,trials=trials,actuals=actuals,violation=violation,
                    seconds=time.monotonic()-started)),flush=True)
            if actual[:105].min()>-1e-9:
                radius=compute_mec_fast(candidate)[2]
                if radius<best_score:
                    result=checked(candidate)
                    print(json.dumps(dict(event='reference',valid=result.valid,score=result.score,errors=result.errors)),flush=True)
                    if result.valid and result.score<best_score:
                        best_score=result.score;save(args.output,candidate)
                        if best_score<args.target:break
            # Repeated lattice points can stagnate even with changing residuals.
            if iteration and iteration%200==0 and len(seen)<iteration//10:
                x[:d]+=rng.normal(0,args.movement*.1,d)*reg
        if best_score<args.target:break
    print(json.dumps(dict(event='finished',frames=frames,trials=trials,actuals=actuals,
        best_score=best_score,best_violation=best_violation,seconds=time.monotonic()-started)),flush=True)


if __name__=='__main__':main()
