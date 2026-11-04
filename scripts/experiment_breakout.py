"""Constraint-priced discrete search: raise the price of blocked contacts.

Unlike a fixed violation energy, breakout penalties change at each local trap.
All proposal geometry is nonlinear; candidates still require the reference
validator before becoming improvements. Neutral angle/translation moves are
not removed except for exact rigid translations.
"""
from __future__ import annotations
import os
for key in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse
from collections import deque
import json
from pathlib import Path
import time
import numpy as np
from fpylll import BKZ,IntegerMatrix,LLL
from breakout_geometry import evaluate_batch
from experiment_lll import flat_contact_pairs
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.geometry import canonicalize_theta
from semicircle_packing.polish import AreaConstraints,checked,load,save

def basis_at(p,regularization=1e-5):
    center=compute_mec_fast(p);q=p.copy();q[:,:2]-=center[:2]
    c=AreaConstraints(q);g,j=c.evaluate(np.r_[q.ravel(),center[2]])
    J=j[g<1e-6,:45];U,_,_=np.linalg.svd(J,full_matrices=True)
    w=U[:,-1];w*=np.sign(w.sum())
    if w.min()<=0:w=np.abs(w)
    C=-w[:,None]*np.column_stack([J[:,0::3].sum(1),J[:,1::3].sum(1)])
    Q,_=np.linalg.qr(C);P=np.eye(len(w))-Q@Q.T
    pairs=flat_contact_pairs(q,c,g)
    flat=np.zeros((len(pairs),45))
    for row,(a,b) in enumerate(pairs):flat[row,3*a+2]=.1;flat[row,3*b+2]=-.1
    W=np.vstack([P@(w[:,None]*J),flat]);scale=1e8;identity=round(scale*regularization)
    B=IntegerMatrix.from_matrix(np.column_stack([np.eye(45)*identity,np.rint(W.T*scale)]).astype(np.int64).tolist())
    LLL.reduction(B,delta=.99);BKZ.reduction(B,BKZ.Param(block_size=20,max_loops=2))
    moves=np.rint(np.array([list(row)for row in B])[:,:45]/identity).astype(np.int64).reshape(45,15,3)
    moves[:,:,:2]-=moves[:,0:1,:2]
    moves=np.unique(moves.reshape(-1,45),axis=0)
    return moves[np.any(moves,axis=1)].reshape(-1,15,3),c.scale

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',default='solution.json');ap.add_argument('--checkpoint')
    ap.add_argument('--output',type=Path,default=Path('experiments/2026-11-04/round2/breakout/best.json'))
    ap.add_argument('--seconds',type=float,default=180);ap.add_argument('--seed',type=int,default=110451)
    ap.add_argument('--target',type=float,default=2.9485269238);ap.add_argument('--movement',type=int,default=1000)
    ap.add_argument('--combinations',type=int,default=300);ap.add_argument('--penalty-step',type=float,default=2)
    ap.add_argument('--restarts',type=int,default=800);ap.add_argument('--regularization',type=float,default=1e-5)
    a=ap.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(a.seed)
    p=load(a.input);initial=checked(p);inc=checked(load('solution.json'));best=inc.score
    moves,scale=basis_at(p,a.regularization)
    if a.checkpoint:
        st=np.load(a.checkpoint);moves=np.rint(st['moves']).astype(np.int64);moves[:,:,:2]-=moves[:,0:1,:2]
        moves=np.unique(moves.reshape(-1,45),axis=0).reshape(-1,15,3);moves=moves[np.any(moves.reshape(-1,45),axis=1)]
    # Add physical one-coordinate angular moves even if LLL did not expose them.
    raw=np.eye(45,dtype=np.int64).reshape(45,15,3)[2::3]
    moves=np.concatenate([moves,raw]);directions=np.concatenate([moves,-moves])
    root=np.rint(p*1e6).astype(np.int64);current=root.copy();anchor=root.copy();center=np.array(compute_mec_fast(p)[:2])
    weights=np.ones(106);weights[-1]=2
    best_energy=float('inf');best_violation=float('inf');best_state=current.copy();best_center=center.copy()
    tabu=deque(maxlen=60);accepted=set();position_states=set();counts=dict(evaluations=0,iterations=0,traps=0,restarts=0,reference_checks=0,valid_states=0,native_fallbacks=0)
    start=time.monotonic();last_report=start
    def evaluate(units,center):
        pp=units*1e-6
        areas,circles,ok=evaluate_batch(pp,center)
        violation=np.c_[np.maximum((areas-1e-6)/scale,0),np.maximum(circles[:,2]-a.target,0)]*1e9
        # Different convex-solver starting centers can differ by ~1e-14 in R.
        # Do not count those differences as descending physical moves. This
        # quantization affects only search energy, never candidate validation.
        violation=np.round(violation,3)
        counts['evaluations']+=len(units);counts['native_fallbacks']+=int((ok==0).sum())
        return violation,circles,areas
    violation,circles,areas=evaluate(current[None],center)
    v=violation[0];center=circles[0,:2]
    while time.monotonic()-start<a.seconds:
        counts['iterations']+=1
        ids=rng.integers(len(directions),size=(a.combinations,3))
        combinations=directions[ids[:,0]]+directions[ids[:,1]]
        use3=rng.random(len(ids))<.35;combinations[use3]+=directions[ids[use3,2]]
        dd=np.concatenate([directions,combinations])
        proposed=current+dd
        proposed[:,:,2]=np.rint(np.arctan2(np.sin(proposed[:,:,2]*1e-6),np.cos(proposed[:,:,2]*1e-6))*1e6).astype(np.int64)
        proposed=np.unique(proposed.reshape(-1,45),axis=0).reshape(-1,15,3)
        proposed=proposed[np.max(np.abs(proposed-anchor),axis=(1,2))<=a.movement]
        if len(proposed)==0:break
        vv,cc,aa=evaluate(proposed,center)
        raw=np.sum(vv*vv,axis=1);radius=cc[:,2]
        score=(vv*vv+vv*.1)@weights
        current_score=float((v*v+v*.1)@weights)
        for k,q in enumerate(proposed):
            if q.tobytes() in tabu:score[k]=float('inf')
        # Keep valid candidates above the goal: they can still improve the saved score.
        # Near the tolerance, the polygon estimator needs a wider screening
        # guard; every promotion still requires the independent GEOS validator.
        for index in np.flatnonzero((aa.max(axis=1)<1e-6+1e-11)&(radius<best-1e-13)):
            counts['reference_checks']+=1;validation=checked(proposed[index]*1e-6)
            if validation.valid:
                counts['valid_states']+=1
                if validation.score<best:
                    best=validation.score;save(a.output,proposed[index]*1e-6)
                    print(json.dumps(dict(event='valid_improvement',score=best,seconds=time.monotonic()-start,**counts)),flush=True)
                    if best<a.target:break
        index=int(np.argmin(raw))
        if raw[index]<best_energy:
            best_energy=float(raw[index]);best_violation=float(vv[index].max());best_state=proposed[index].copy();best_center=cc[index,:2].copy()
            save(a.output.with_name(a.output.stem+'_nearest.json'),best_state*1e-6)
            print(json.dumps(dict(event='nearest',energy_nm2=best_energy,max_violation_nm=best_violation,radius=float(radius[index]),seconds=time.monotonic()-start,**counts)),flush=True)
        if best<a.target:break
        index=int(np.argmin(score))
        if score[index]>=current_score-1e-8 or (counts['iterations']%15==0 and current[:,:2].tobytes() in position_states):
            counts['traps']+=1
            weights+=a.penalty_step*(v>.001)
            if counts['traps']%20==0:weights=1+(weights-1)*.9
            score=(vv*vv+vv*.1)@weights
            for k,q in enumerate(proposed):
                if q.tobytes() in tabu:score[k]=float('inf')
            index=int(np.argmin(score))
        tabu.append(current.tobytes());current=proposed[index].copy();v=vv[index].copy();center=cc[index,:2].copy();accepted.add(current.tobytes());position_states.add(current[:,:2].tobytes())
        if counts['iterations']%a.restarts==0:
            counts['restarts']+=1;current=best_state.copy() if counts['restarts']%2 else root.copy();center=best_center.copy()
            weights=1+rng.exponential(.5,106);weights[-1]*=2;tabu.clear()
            v,circles,_=evaluate(current[None],center);v=v[0];center=circles[0,:2]
        if time.monotonic()-last_report>10:
            print(json.dumps(dict(event='progress',best_score=best,best_energy_nm2=best_energy,best_violation_nm=best_violation,unique_accepted=len(accepted),unique_positions=len(position_states),max_weight=float(weights.max()),seconds=time.monotonic()-start,**counts)),flush=True);last_report=time.monotonic()
    result=dict(event='done',seed=a.seed,target=a.target,best_score=best,best_energy_nm2=best_energy,best_violation_nm=best_violation,unique_accepted=len(accepted),unique_positions=len(position_states),search_energy_quantum=1e-12,seconds=time.monotonic()-start,**counts)
    a.output.with_suffix('.report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
