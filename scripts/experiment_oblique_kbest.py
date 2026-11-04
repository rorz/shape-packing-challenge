"""Bounded k-best conditional lattice decoding for contact inequalities.

Unlike a fresh-target Babai sweep, every target explores a beam of coherent
rounding histories through all 45 Gram--Schmidt levels. Geometry decides among
the completions; Euclidean target distance only controls the bounded beam.
"""
from __future__ import annotations
import os
for name in ('VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
import argparse,json,shlex,sys,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from fpylll import BKZ,GSO,IntegerMatrix,LLL
from scipy.optimize import minimize
from semicircle_packing.polish import AreaConstraints,checked,load,save
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.geometry import canonicalize_theta
from arena import check_solution

LOG=Path('experiments/2026-11-04/lattice-log.md')

def beam_decode(target,mu,diagonal,width=256,branches=3,freeze_low=0,contact_initial=None,contact_directions=None,stress=None,asymmetry=0,metric_scale=1):
    """Retain width lowest-cost conditional rounding histories at each level."""
    dimension=len(target);coeff=np.zeros((1,dimension),dtype=np.int64);cost=np.zeros(1)
    contacts=None if contact_initial is None else np.asarray(contact_initial)[None,:].copy()
    offsets=np.arange(-(branches//2),branches//2+1,dtype=np.int64)
    for j in range(dimension-1,-1,-1):
        centers=target[j]-coeff[:,j+1:]@mu[j+1:,j]
        local_offsets=np.array([0]) if j<freeze_low else offsets
        integers=np.rint(centers).astype(np.int64)[:,None]+local_offsets
        costs=cost[:,None]+(integers-centers[:,None])**2*diagonal[j]
        scores=costs/(metric_scale*metric_scale)
        alternatives=None
        if contacts is not None:
            residual=integers-centers[:,None]
            alternatives=contacts[:,None,:]+residual[:,:,None]*contact_directions[j][None,None,:]
            negative=np.minimum(alternatives*stress[None,None,:]/1e-6,0)
            scores=scores+asymmetry*np.sum(negative*negative,axis=2)
        count=min(width,costs.size)
        keep=np.argpartition(scores.ravel(),count-1)[:count]
        order=np.argsort(scores.ravel()[keep]);keep=keep[order]
        parent,choice=np.divmod(keep,len(local_offsets))
        coeff=coeff[parent].copy();coeff[:,j]=integers[parent,choice];cost=costs[parent,choice]
        if alternatives is not None:contacts=alternatives[parent,choice]
    return coeff,cost


def continuous_seed(incumbent,pin=True):
    p=incumbent.copy();p[:,:2]-=compute_mec_fast(p)[:2]
    c=AreaConstraints(p);z=np.r_[p.ravel(),compute_mec_fast(p)[2]];objective=np.zeros(46);objective[-1]=1
    constraints=[{'type':'ineq','fun':c.fun,'jac':c.jac}]
    if pin:
        D=np.zeros((2,46))
        for row,(i,j) in enumerate([(1,5),(6,10)]):D[row,3*i+2]=1;D[row,3*j+2]=-1
        wanted=D@z
        constraints.append({'type':'eq','fun':lambda v:D@v-wanted,'jac':lambda v:D})
    result=minimize(lambda v:v[-1],z,jac=lambda v:objective,method='SLSQP',constraints=constraints,
                    options={'maxiter':250,'ftol':2e-13})
    return result.x[:-1].reshape(15,3),result.x[-1],dict(success=bool(result.success),iterations=result.nit,violation=float(-min(c.fun(result.x))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='solution.json');parser.add_argument('--output',required=True)
    parser.add_argument('--seconds',type=float,default=120);parser.add_argument('--seed',type=int,default=2401104)
    parser.add_argument('--width',type=int,default=256);parser.add_argument('--branches',type=int,default=3)
    parser.add_argument('--cuts',type=int,default=0)
    parser.add_argument('--asymmetry',type=float,default=0)
    parser.add_argument('--freeze-low',type=int,default=2)
    parser.add_argument('--angular-repair',action='store_true')
    parser.add_argument('--actual',type=int,default=96);parser.add_argument('--block-size',type=int,default=20)
    parser.add_argument('--target',type=float,default=2.94852695);parser.add_argument('--goal',type=float,default=2.9485269238)
    parser.add_argument('--target-noise',type=float,default=.001)
    parser.add_argument('--targets',choices=['simplex','incumbent'],default='incumbent')
    parser.add_argument('--snap-frame',action='store_true')
    parser.add_argument('--movement',type=float,default=50);parser.add_argument('--angle',type=float,default=0)
    parser.add_argument('--regularization',type=float,default=1e-5);parser.add_argument('--flat-weight',type=float,default=.1)
    parser.add_argument('--no-pin-flat',action='store_true');args=parser.parse_args()
    if args.cuts and not args.asymmetry:parser.error("--cuts requires --asymmetry > 0")
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    events=out.with_suffix('.events.jsonl');rng=np.random.default_rng(args.seed);start=time.monotonic()
    def emit(event,**values):
        row=dict(time=datetime.now(timezone.utc).isoformat(),event=event,seconds=time.monotonic()-start,**values)
        text=json.dumps(row,allow_nan=False);print(text,flush=True)
        with events.open('a') as f:f.write(text+'\n')
        if event!='progress':
            with LOG.open('a') as f:f.write('\n- '+text+'\n')
    with LOG.open('a') as f:
        f.write('\n## Bounded k-best run\n\nHypothesis: coherent alternative GSO roundings reveal feasible inequalities discarded by single Babai decoding.\n\n```text\n'+shlex.join(['.venv/bin/python',*sys.argv])+'\n```\n')
    incumbent=load(args.input);validation=checked(incumbent);best_score=validation.score if validation.valid else float('inf')
    cs,sn=np.cos(args.angle),np.sin(args.angle);rotation=np.array([[cs,sn],[-sn,cs]])
    if args.snap_frame:
        framed=incumbent.copy();framed[:,:2]=framed[:,:2]@rotation
        framed[:,2]=[canonicalize_theta(t+args.angle) for t in framed[:,2]]
        p,initial_radius,initial_info=continuous_seed(framed,not args.no_pin_flat)
        known=framed.copy();known[:,:2]-=compute_mec_fast(known)[:2]
    else:
        p,initial_radius,initial_info=continuous_seed(incumbent,not args.no_pin_flat)
        p[:,:2]=p[:,:2]@rotation;p[:,2]=np.arctan2(np.sin(p[:,2]+args.angle),np.cos(p[:,2]+args.angle))
        known=incumbent.copy();known[:,:2]-=compute_mec_fast(known)[:2];known[:,:2]=known[:,:2]@rotation;known[:,2]+=args.angle
    c=AreaConstraints(p);g,j=c.evaluate(np.r_[p.ravel(),initial_radius]);active=g<1e-7;J=j[active,:45]
    left,singular,_=np.linalg.svd(J,full_matrices=True);weights=left[:,-1];weights*=np.sign(weights.sum())
    if weights.min()<=0:raise RuntimeError('Contact stress has a negative component; model is not a packing minimum')
    gaps,_=c.evaluate(np.r_[p.ravel(),args.target]);g0=gaps[active];budget=weights@g0
    anchor=np.round(p,6);offset=(anchor-p).ravel()/1e-6;b0=-g0/1e-6-J@offset
    known_delta=known-p;known_delta[:,2]=np.arctan2(np.sin(known_delta[:,2]),np.cos(known_delta[:,2]))
    known_gaps=g0+J@known_delta.ravel()
    prior=known-anchor;prior[:,2]=np.arctan2(np.sin(prior[:,2]),np.cos(prior[:,2]));prior=prior.ravel()/1e-6
    angular=np.zeros((2,45))
    for row,(i,k) in enumerate([(1,5),(6,10)]):angular[row,3*i+2]=args.flat_weight;angular[row,3*k+2]=-args.flat_weight
    C=-np.column_stack([J[:,0::3].sum(axis=1),J[:,1::3].sum(axis=1)]);Q,_=np.linalg.qr(weights[:,None]*C)
    projection=np.eye(len(weights))-Q@Q.T;inverse=np.linalg.pinv(weights[:,None]*C)
    metric=np.vstack([projection@(weights[:,None]*J),angular]);scale=10000000;identity=max(1,round(scale*args.regularization))
    lattice=IntegerMatrix.from_matrix(np.column_stack([np.eye(45)*identity,np.rint(metric.T*scale)]).astype(np.int64).tolist())
    LLL.reduction(lattice,delta=.99)
    if args.block_size:BKZ.reduction(lattice,BKZ.Param(block_size=args.block_size,max_loops=2))
    M=GSO.Mat(lattice,update=True);reduced=np.array([list(row) for row in lattice],dtype=np.int64);moves=np.rint(reduced[:,:45]/identity).astype(np.int64)
    mu=np.array([[M.get_mu(i,k) if k<i else 0 for k in range(45)] for i in range(45)]);diagonal=np.array([M.get_r(i,i) for i in range(45)])
    star=reduced.astype(float).copy()
    for row in range(45):
        for earlier in range(row):star[row]-=mu[row,earlier]*star[earlier]
    projected_J=(np.eye(len(weights))-C@inverse@np.diag(weights))@J
    contact_directions=star[:,:45]@projected_J.T*1e-6/identity
    emit('model',active=int(active.sum()),minimum_stress=float(weights.min()),initial_radius=float(initial_radius),initial_info=initial_info,seed=args.seed,width=args.width,branches=args.branches)
    trials=actuals=targets=duplicates=0;seen=set();best_violation=1.;last_report=start;cuts=[]
    while time.monotonic()-start<args.seconds:
        # Deterministic center / face-biased targets alternate with a randomized
        # allocation; exploration is primarily in the conditional lattice tree.
        bias=[1.,3.,8.,1.][targets%4];concentration=[1000.,1000.,1000.,1.][targets%4]
        alpha=np.full(len(weights),concentration);alpha[np.flatnonzero(active)>=105]*=bias
        allocation=(alpha/alpha.sum()) if targets%4<3 else rng.dirichlet(alpha)
        desired=budget*allocation/weights/1e-6
        if args.targets=='incumbent':
            # Euclidean projection of the incumbent's weighted slack onto the
            # feasible contact simplex. Unlike a fresh Dirichlet target, this
            # preserves its useful, highly uneven clearance pattern.
            margin=[0.,.015,.03,.06][targets%4]
            lower=np.zeros(len(weights));lower[np.flatnonzero(active)>=105]=weights[np.flatnonzero(active)>=105]*margin
            total=budget/1e-6-lower.sum()
            if total<=0:continue
            values=weights*known_gaps/1e-6-lower
            if targets>=4:values+=rng.normal(0,args.target_noise,len(values))
            ordered=np.sort(values)[::-1];cumsum=np.cumsum(ordered)-total
            rho=np.flatnonzero(ordered-cumsum/np.arange(1,len(values)+1)>0)[-1]
            threshold=cumsum[rho]/(rho+1)
            desired=(np.maximum(values-threshold,0)+lower)/weights
        wanted=prior+(rng.normal(0,args.movement,45) if targets else 0)
        contact=np.r_[projection@(weights*(desired+b0)),-angular@offset]
        ambient=np.rint(np.r_[wanted*identity,contact*scale]).astype(np.int64)
        gso_target=np.array(M.from_canonical(ambient))
        initial_contacts=None
        if args.asymmetry:
            projected_target=np.array(M.to_canonical(gso_target))[:45]/identity
            raw=g0+J@(offset+projected_target)*1e-6
            initial_contacts=raw+C@(inverse@(weights*(desired*1e-6-raw)))
        dynamic_directions=contact_directions;dynamic_stress=weights
        if cuts:
            displacement=(offset+np.array(M.to_canonical(gso_target))[:45]/identity)*1e-6
            center_constant=inverse@(weights*(desired*1e-6-g0))
            extra=np.array([cut['constant']+cut['center']@center_constant+cut['projected']@displacement for cut in cuts])
            initial_contacts=np.r_[initial_contacts,extra]
            dynamic_directions=np.column_stack([contact_directions,*[star[:,:45]@cut['projected']*1e-6/identity for cut in cuts]])
            dynamic_stress=np.r_[weights,[cut['weight'] for cut in cuts]]
        coefficients,cost=beam_decode(gso_target,mu,diagonal,args.width,args.branches,args.freeze_low,
            initial_contacts,dynamic_directions,dynamic_stress,args.asymmetry,scale)
        units=coefficients@moves;pred=g0+(offset[None,:]+units)@J.T*1e-6
        centers=(weights*(desired[None,:]*1e-6-pred))@inverse.T
        pred+=centers@C.T
        if args.angular_repair and args.freeze_low>2:
            projected_J=(np.eye(len(weights))-C@inverse@np.diag(weights))@J
            changes=moves[2:args.freeze_low]@projected_J.T*1e-6
            options=np.arange(-4,5)
            for sweep in range(3):
                for local,direction in enumerate(changes):
                    alternatives=pred[:,None,:]+options[None,:,None]*direction[None,None,:]
                    negative=np.minimum(alternatives,0)
                    choice=np.argmin(np.sum(negative*negative,axis=2),axis=1)
                    chosen=options[choice]
                    units+=chosen[:,None]*moves[local+2]
                    pred+=chosen[:,None]*direction
            raw=g0+(offset[None,:]+units)@J.T*1e-6
            centers=(weights*(desired[None,:]*1e-6-raw))@inverse.T
            pred=raw+centers@C.T
        ranking=pred
        if cuts:
            delta_physical=(offset[None,:]+units)*1e-6
            center_constant=inverse@(weights*(desired*1e-6-g0))
            ranking=np.column_stack([pred,*[cut['constant']+cut['center']@center_constant+delta_physical@cut['projected'] for cut in cuts]])
        losses=np.minimum(ranking,0);order=np.argsort(np.max(-ranking,axis=1)+np.sqrt(np.mean(losses*losses,axis=1)))
        targets+=1;trials+=len(units)
        checked_this_target=0
        for index in order:
            if checked_this_target>=args.actual:break
            relative=units[index].reshape(15,3).copy();relative[:,:2]-=relative[0,:2]
            key=tuple(relative.ravel());
            if key in seen:duplicates+=1;continue
            seen.add(key);checked_this_target+=1;candidate=np.round(anchor+units[index].reshape(15,3)*1e-6,6)
            candidate[:,2]=[canonicalize_theta(t) for t in candidate[:,2]]
            shifted=candidate.copy();shifted[:,:2]-=centers[index]
            actual,actual_jac=c.evaluate(np.r_[shifted.ravel(),args.target]);actuals+=1;violation=float(-actual.min())
            if args.cuts and len(cuts)<args.cuts and violation>1e-7 and -pred[index].min()<1e-7:
                row=int(np.argmin(actual));raw_jac=actual_jac[row,:45].copy()
                if row in np.flatnonzero(active):
                    weight=float(weights[np.flatnonzero(active).tolist().index(row)])
                    center=-raw_jac.reshape(15,3)[:,:2].sum(axis=0)
                    projected=raw_jac-center@inverse@(weights[:,None]*J)
                    constant=float(actual[row]+raw_jac@(p-shifted).ravel())
                    # Keep genuinely different contact branches, not repeated
                    # nearly identical tangent planes at the same witness.
                    similar=any(cut['row']==row and np.linalg.norm(cut['raw']-raw_jac)<1e-3 for cut in cuts)
                    if not similar:
                        cuts.append(dict(row=row,raw=raw_jac,center=center,projected=projected,constant=constant,weight=weight))
                        emit('cut',row=row,violation=violation,linear_violation=float(-pred[index].min()),count=len(cuts))
            if violation<best_violation:
                best_violation=violation;save(out.with_name(out.stem+'-nearest.json'),candidate)
                np.savez(out.with_name(out.stem+'-state.npz'),placements=candidate,origin=p,moves=moves.reshape(45,15,3),center=centers[index],target=args.target)
                emit('nearest',violation=violation,targets=targets,trials=trials,actuals=actuals,linear_violation=float(-pred[index].min()))
            if actual[:105].min()>-5e-10:
                score=compute_mec_fast(candidate)[2]
                if score<best_score:
                    result=checked(candidate)
                    if result.valid and result.score<best_score:
                        best_score=result.score;save(out,candidate);response=check_solution(json.loads(out.read_text()));out.with_suffix('.api.json').write_text(json.dumps(response,indent=2)+'\n')
                        emit('improvement',score=best_score,api=response,targets=targets,trials=trials)
                        if response.get('valid') and response['score']<args.goal:return
        if time.monotonic()-last_report>10:
            emit('progress',targets=targets,trials=trials,actuals=actuals,duplicates=duplicates,best_violation=best_violation,best_score=best_score if np.isfinite(best_score) else None);last_report=time.monotonic()
    emit('done',targets=targets,trials=trials,actuals=actuals,duplicates=duplicates,best_violation=best_violation,best_score=best_score if np.isfinite(best_score) else None)

if __name__=='__main__':main()
