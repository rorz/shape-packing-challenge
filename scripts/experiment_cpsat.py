"""CP-SAT search on reduced integer moves, retaining nonlinear contact cuts.

Optional dependency: ortools. Tangent cuts guide local search, not global proofs.
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
from ortools.sat.python import cp_model
from semicircle_packing.polish import AreaConstraints, checked, load, save
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.geometry import canonicalize_theta


def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--checkpoint',default='experiments/lll_hillclimb_best_state.npz')
    pa.add_argument('--input', help='Override the checkpoint placement and rebuild its reduced basis')
    pa.add_argument('--output',type=Path,default=Path('experiments/2026-11-04/cpsat/best.json'))
    pa.add_argument('--seconds',type=float,default=120)
    pa.add_argument('--solve-seconds',type=float,default=20)
    pa.add_argument('--coefficients',type=int,default=4)
    pa.add_argument('--movement',type=int,default=100)
    pa.add_argument('--target',type=float,default=2.9485269238)
    pa.add_argument('--seed',type=int,default=110401)
    pa.add_argument('--workers',type=int,default=2)
    pa.add_argument('--slack',choices=['none','radius','all'],default='none')
    pa.add_argument('--minimum',type=float,default=2.948526966725367)
    args=pa.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    with np.load(args.checkpoint) as state:
        base=np.round(state['placements'],6)
        origin=state['origin'].copy()
        moves=np.rint(state['moves']).astype(np.int64).reshape(45,45)
    if args.input:
        from fpylll import BKZ, IntegerMatrix, LLL
        base=load(args.input)
        origin=base.copy()
        center=compute_mec_fast(base)
        origin[:,:2]-=center[:2]
        probe=AreaConstraints(origin)
        gg,jj=probe.evaluate(np.r_[origin.ravel(),args.target])
        J=jj[gg<1e-6,:45]
        U,_,_=np.linalg.svd(J,full_matrices=True)
        weights=np.abs(U[:,-1])
        cj=-np.column_stack([J[:,0::3].sum(axis=1),J[:,1::3].sum(axis=1)])
        Q,_=np.linalg.qr(weights[:,None]*cj)
        projection=np.eye(len(weights))-Q@Q.T
        angle_rows=np.zeros((2,45))
        for row,(i,j) in enumerate([(1,5),(6,10)]):
            angle_rows[row,3*i+2]=.1;angle_rows[row,3*j+2]=-.1
        WJ=np.vstack([projection@(weights[:,None]*J),angle_rows])
        basis=np.column_stack([np.eye(45)*100,np.round(WJ.T*1e7)]).astype(np.int64)
        B=IntegerMatrix.from_matrix(basis.tolist());LLL.reduction(B,delta=.99)
        BKZ.reduction(B,BKZ.Param(block_size=20,max_loops=2))
        moves=np.rint(np.array([list(row) for row in B],dtype=float)[:,:45]/100).astype(np.int64)
    constraint=AreaConstraints(origin)
    cx,cy,_=compute_mec_fast(base); initial_center=np.array([cx,cy])
    direction=moves.T*1e-6
    center_unit=1e-10;slack_unit=1e-11;scale=1e16
    model=cp_model.CpModel()
    variables=[model.new_int_var(-args.coefficients,args.coefficients,f'k{i}') for i in range(45)]
    center_bound=args.movement*10000
    variables += [model.new_int_var(-center_bound,center_bound,f'c{i}') for i in range(2)]
    for i,move in enumerate(moves.reshape(45,15,3)):
        if np.all(move[:,2]==0) and np.all(move[:,:2]==move[0,:2]):model.add(variables[i]==0)
    for coordinate,row in enumerate(moves.T):
        expression=sum(int(a)*v for a,v in zip(row,variables[:45]) if a)
        if coordinate<2:
            model.add(expression==0)  # Quotient out exact integer translations.
        else:
            model.add_linear_constraint(expression,-args.movement,args.movement)
    slack=model.new_int_var(-100000 if args.slack=='radius' else 0,100000,'slack')
    if args.slack=='none':model.add(slack==0)
    else:model.minimize(slack)
    cuts=0
    def evaluate(values):
        p=base+(values[:45]@moves).reshape(15,3)*1e-6
        p=np.round(p,6);p[:,2]=[canonicalize_theta(t) for t in p[:,2]]
        centered=p.copy();centered[:,:2]-=initial_center+values[45:47]*center_unit
        g,j=constraint.evaluate(np.r_[centered.ravel(),args.target])
        dj=np.column_stack([j[:,:45]@direction,
            -j[:,0:45:3].sum(axis=1)*center_unit,
            -j[:,1:45:3].sum(axis=1)*center_unit])
        return p,g,dj
    def add_cuts(values,g,j):
        nonlocal cuts
        active=np.flatnonzero((g<1e-3)&(np.linalg.norm(j,axis=1)>1e-15))
        for index in active:
            coeff=np.rint(j[index]*scale).astype(np.int64)
            lower=int(np.ceil((-g[index]+j[index]@values)*scale))
            expression=sum(int(a)*v for a,v in zip(coeff,variables) if a)
            if args.slack=='all' or (args.slack=='radius' and index>=105):
                expression+=int(round(slack_unit*scale))*slack
            model.add(expression>=lower)
            cuts+=1
    values=np.zeros(47);p,g,j=evaluate(values);add_cuts(values,g,j)
    started=time.monotonic();iteration=0;best_violation=float('inf');best_score=args.minimum
    while time.monotonic()-started<args.seconds:
        solver=cp_model.CpSolver();solver.parameters.max_time_in_seconds=min(args.solve_seconds,args.seconds-(time.monotonic()-started))
        solver.parameters.num_search_workers=args.workers;solver.parameters.random_seed=args.seed+iteration
        solver.parameters.randomize_search=True
        model.clear_hints()
        for v,a in zip(variables,values):model.add_hint(v,int(round(a)))
        relevant=g if args.slack=='all' else g[105:]
        hinted_slack=int(np.ceil(-relevant.min()/slack_unit))+1
        if args.slack!='none':model.add_hint(slack,max(-100000 if args.slack=='radius' else 0,min(100000,hinted_slack)))
        iteration+=1;status=solver.solve(model)
        info=dict(iteration=iteration,status=solver.status_name(status),cuts=cuts,seconds=time.monotonic()-started,
                  branches=solver.num_branches,conflicts=solver.num_conflicts)
        if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):
            print(json.dumps(info),flush=True)
            if status in (cp_model.INFEASIBLE,cp_model.MODEL_INVALID):break
            continue
        values=np.array([solver.value(v) for v in variables]);p,g,j=evaluate(values)
        violation=-float(g.min());info.update(violation=violation,slack=solver.value(slack)*slack_unit,
            movement=int(np.abs(values[:45]@moves).max()),objective_bound=solver.best_objective_bound*slack_unit)
        print(json.dumps(info),flush=True)
        if violation<best_violation:
            best_violation=violation;save(args.output.with_name(args.output.stem+'_nearest.json'),p)
            np.savez(args.output.with_suffix('.npz'),placements=p,origin=origin,moves=moves.reshape(45,15,3),target=args.target)
        if g[:105].min()>-1e-9:
            validation=checked(p)
            print(json.dumps(dict(event='reference',valid=validation.valid,score=validation.score,errors=validation.errors)),flush=True)
            if validation.valid and validation.score<best_score:
                best_score=validation.score;save(args.output,p)
                if best_score<args.target:break
        add_cuts(values,g,j)
        model.add_forbidden_assignments(variables[:45],[values[:45].tolist()])
    print(json.dumps(dict(event='finished',seconds=time.monotonic()-started,iterations=iteration,best_score=best_score,
        best_violation=best_violation if np.isfinite(best_violation) else None)),flush=True)


if __name__=='__main__':main()
