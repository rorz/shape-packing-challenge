"""Pin the endpoint-sensitive angle on-grid, then search the other 44 integers.

Optional dependencies: uv run --extra lattice python search.py lll --help
"""
from __future__ import annotations
import os
for key in ('VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse
import itertools
import json
from pathlib import Path
import time
import numpy as np
from fpylll import BKZ, CVP, GSO, IntegerMatrix, LLL
from semicircle_packing.polish import AreaConstraints, checked, load, polish, save
from semicircle_packing.geometry import canonicalize_theta
from semicircle_packing.fast_mec import compute_mec_fast


def flat_contact_pairs(p, constraints, gaps):
    """Detect opposed active flats without assuming a piece numbering."""
    return [(i,j)for row,(i,j)in enumerate(itertools.combinations(range(len(p)),2))
            if np.cos(p[i,2]-p[j,2]) < -.999999 and gaps[row] < 1e-6
            and constraints.scale[row] > .1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='solution.json')
    parser.add_argument('--output', default='experiments/2026-11-04/round3/pinned/best.json')
    parser.add_argument('--pin-piece', type=int, default=-1,
                        help='Piece whose angle defines the frame; -1 detects two active endpoints')
    parser.add_argument('--target', type=float, default=2.9485269238)
    parser.add_argument('--goal', type=float)
    parser.add_argument('--seconds', type=float, default=90)
    parser.add_argument('--seed', type=int, default=9201)
    parser.add_argument('--regularization', type=float, default=1e-5)
    parser.add_argument('--movement', type=float, default=500)
    parser.add_argument('--flat-weight', type=float, default=.1)
    parser.add_argument('--flat-pairs', default='auto',
                        help='auto detects opposed active flats; alternatively i:j,k:l or none')
    parser.add_argument('--block-size', type=int, default=0)
    parser.add_argument('--exact-every', type=int, default=0)
    parser.add_argument('--keep-seed', action='store_true')
    parser.add_argument('--fixed-center', action='store_true')
    parser.add_argument('--frame-angle', type=float)
    parser.add_argument('--samples-per-frame', type=int, default=300)
    parser.add_argument('--concentration', type=float, default=10.)
    parser.add_argument('--boundary-bias', type=float, default=1.)
    parser.add_argument('--near-valid')
    parser.add_argument('--incumbent')
    parser.add_argument('--checkpoint-prefix',default='lll')
    args = parser.parse_args()
    goal = args.target if args.goal is None else args.goal
    rng = np.random.default_rng(args.seed)
    original = load(args.input)
    if args.keep_seed:
        cx,cy,initial_radius = compute_mec_fast(original)
        original[:,:2] -= [cx,cy]
    else:
        initial = polish(original)
        original = initial.x[:-1].reshape(-1,3)
        initial_radius = initial.x[-1]
    boundary=AreaConstraints(original).fast.boundary(np.r_[original.ravel(),initial_radius])[0].reshape(15,3)
    pin_piece=args.pin_piece
    if pin_piece<0:
        sensitive=np.flatnonzero(np.max(boundary[:,:2],axis=1)<1e-7)
        if len(sensitive)!=1:raise ValueError(f'Expected one double-endpoint boundary piece, found {sensitive.tolist()}')
        pin_piece=int(sensitive[0])
    if not 0<=pin_piece<15:raise ValueError('pin-piece must be in0..14')
    columns=np.array([i for i in range(45)if i!=3*pin_piece+2]);dimension=len(columns)
    started = time.monotonic()
    best_violation = float('inf')
    best_score = float('inf')
    for incumbent in [args.incumbent, args.output if Path(args.output).exists() else None]:
        if incumbent:
            validation = checked(load(incumbent))
            if validation.valid:
                best_score = min(best_score,validation.score)
    trial = 0
    frames=0
    print(json.dumps(dict(event='start',pin_piece=pin_piece,dimension=dimension,continuous_radius=initial_radius)),flush=True)
    def finish():
        print(json.dumps(dict(trials=trial,
            best_score=best_score if np.isfinite(best_score) else None,
            best_violation=best_violation if np.isfinite(best_violation) else None,
            goal=goal, goal_met=best_score<goal,
            frames=frames,pin_piece=pin_piece,dimension=dimension,
            seconds=time.monotonic()-started)),flush=True)
    if best_score < goal:
        finish()
        return
    while time.monotonic()-started < args.seconds:
        angle = rng.uniform(-np.pi,np.pi) if args.frame_angle is None else args.frame_angle
        endpoint_angle=np.arctan2(np.sin(original[pin_piece,2]+angle),np.cos(original[pin_piece,2]+angle))
        angle+=np.round(endpoint_angle,6)-endpoint_angle
        cs,sn = np.cos(angle),np.sin(angle)
        p = original.copy()
        p[:,:2] = p[:,:2] @ np.array([[cs,sn],[-sn,cs]])
        p[:,2] = np.arctan2(np.sin(p[:,2]+angle),np.cos(p[:,2]+angle))
        p[pin_piece,2]=np.round(p[pin_piece,2],6)
        c = AreaConstraints(p)
        at_opt = np.r_[p.ravel(),initial_radius]
        gaps,jac = c.evaluate(at_opt)
        active = gaps < 1e-7
        J = jac[active,:-1]
        left,singular,_ = np.linalg.svd(J,full_matrices=True)
        weights = left[:,-1]
        weights *= np.sign(weights.sum())
        if min(weights) <= 0:
            continue
        z = at_opt.copy();z[-1]=args.target
        gaps,_ = c.evaluate(z)
        g0 = gaps[active]
        budget = weights @ g0
        anchor = np.round(p,6)
        prior = np.zeros(45)
        if args.near_valid:
            known = load(args.near_valid)
            shift = np.round(np.mean(anchor[:,:2]-known[:,:2],axis=0),6)
            known[:,:2] += shift
            offset = known-anchor
            offset[:,2] = np.arctan2(np.sin(offset[:,2]),np.cos(offset[:,2]))
            prior = offset.ravel()/1e-6
        b0 = -g0/1e-6 - J @ ((anchor-p).ravel()/1e-6)
        # Keep the two opposed-flat relative angles in a locally smooth range.
        pairs=(flat_contact_pairs(p,c,gaps) if args.flat_pairs=='auto' else
               [] if args.flat_pairs=='none' else
               [tuple(map(int,pair.split(':')))for pair in args.flat_pairs.split(',')])
        angle_rows = np.zeros((len(pairs),45))
        for row,(i,j) in enumerate(pairs):
            angle_rows[row,3*i+2] = args.flat_weight
            angle_rows[row,3*j+2] = -args.flat_weight
        center_jac = -np.column_stack([J[:,0::3].sum(axis=1),J[:,1::3].sum(axis=1)])
        weighted_center = weights[:,None]*center_jac
        Q,_ = np.linalg.qr(weighted_center)
        projection = np.eye(len(weights)) if args.fixed_center else np.eye(len(weights))-Q@Q.T
        center_inverse = np.linalg.pinv(weighted_center)
        WJ = np.vstack([projection @ (weights[:,None]*J),angle_rows])
        scale = 10000000
        identity = max(1,round(scale*args.regularization))
        basis = np.column_stack([np.eye(dimension)*identity, np.round(WJ[:,columns].T*scale)]).astype(np.int64)
        B = IntegerMatrix.from_matrix(basis.tolist())
        LLL.reduction(B,delta=.99)
        if args.block_size:
            BKZ.reduction(B,BKZ.Param(block_size=args.block_size,max_loops=2))
        M = GSO.Mat(B,update=True)
        reduced = np.array([list(row) for row in B],dtype=float)
        frames+=1
        for attempt in range(args.samples_per_frame):
            if time.monotonic()-started > args.seconds:
                finish()
                return
            concentration = args.concentration
            if concentration <= 0:
                concentration = rng.choice([.2,1.,10.])
            alpha = np.full(len(weights),concentration)
            alpha[np.flatnonzero(active) >= c.fast.n_pairs] *= args.boundary_bias
            allocation = rng.dirichlet(alpha)
            desired = budget*allocation/weights/1e-6
            target_contact = np.r_[projection @ (weights*(desired+b0)),
                -angle_rows @ ((anchor-p).ravel()/1e-6)]
            wanted = prior[columns] + rng.normal(0,args.movement,dimension)
            target = np.r_[wanted*identity,target_contact*scale]
            integer_target = np.rint(target).astype(np.int64).tolist()
            if args.exact_every and attempt % args.exact_every == 0:
                nearest = np.array(CVP.closest_vector(B,integer_target,method='fast'),dtype=float)
            else:
                coefficients = M.babai(integer_target)
                nearest = np.array(B.multiply_left(coefficients),dtype=float)
            delta=np.zeros(45)
            delta[columns]=np.rint(nearest[:dimension]/identity)
            delta=delta.reshape(15,3)
            raw_candidate = anchor + delta*1e-6
            candidate = raw_candidate.copy()
            candidate[:,2] = [canonicalize_theta(t) for t in candidate[:,2]]
            actual,_ = c.evaluate(np.r_[candidate.ravel(),args.target])
            fitted_center = np.zeros(2)
            if not args.fixed_center:
                fitted_center = center_inverse @ (weights*(desired*1e-6-actual[active]))
                shifted = candidate.copy();shifted[:,:2] -= fitted_center
                actual,_ = c.evaluate(np.r_[shifted.ravel(),args.target])
            violation = -actual.min()
            trial += 1
            if violation < best_violation:
                best_violation = violation
                print(json.dumps(dict(trials=trial,violation=violation,
                    movement=float(np.abs(delta).max()),seconds=time.monotonic()-started)),flush=True)
                save(Path(args.output).with_name(args.checkpoint_prefix+'_nearest.json'),raw_candidate)
                state = Path(args.output).with_name(args.checkpoint_prefix+'_basis.npz')
                temporary = state.with_suffix('.tmp.npz')
                full_moves=np.zeros((dimension,45))
                full_moves[:,columns]=np.rint(reduced[:,:dimension]/identity)
                np.savez(temporary, placements=raw_candidate, geometry=candidate, origin=p,
                         moves=full_moves.reshape(dimension,15,3),pin_piece=pin_piece,dimension=dimension,
                         center=fitted_center,
                         target=args.target)
                temporary.replace(state)
            if actual[:c.fast.n_pairs].min() > -5e-10:
                _,_,quick_radius = compute_mec_fast(candidate)
                if quick_radius >= min(best_score,args.target):
                    continue
                result = checked(raw_candidate)
                if result.valid and result.score < min(best_score,args.target):
                    best_score = result.score
                    save(args.output,raw_candidate)
                    print(json.dumps(dict(best=result.score,goal=goal,seconds=time.monotonic()-started)),flush=True)
                    if result.score < goal:
                        finish()
                        return
    finish()


if __name__ == '__main__':
    main()
