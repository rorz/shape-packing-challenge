"""Batched lattice search with exact integer opposed-flat angle differences.

This is a restricted proposal family, not an infeasibility proof. Raw submitted
angles are retained separately from once-canonicalized scoring geometry.
"""
from __future__ import annotations
import os
for key in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
from fpylll import BKZ,GSO,IntegerMatrix,LLL
from experiment_batch_lattice import NativeBabai
from experiment_tied_flats import wrap_cells
from experiment_lll import flat_contact_pairs
from breakout_geometry import evaluate_batch
from semicircle_packing.polish import AreaConstraints,checked,load,save
from semicircle_packing.fast_mec import compute_mec_fast
from arena import check_solution


def tied_transform(p,pairs):
    dependent={3*i+2 for i,j in pairs}
    columns=np.array([i for i in range(p.size)if i not in dependent])
    transform=np.eye(p.size,dtype=np.int64)[:,columns]
    for i,j in pairs:transform[3*i+2,np.flatnonzero(columns==3*j+2)[0]]=1
    return columns,transform


class Frame:
    def __init__(self,p,radius,pairs,args):
        self.p=p;self.pairs=pairs;self.differences=np.rint([1e6*(p[i,2]-p[j,2])for i,j in pairs]).astype(np.int64)
        self.columns,T=tied_transform(p,pairs)
        self.pin_piece=getattr(args,'pin_piece',None)
        self.pin_raw=getattr(args,'pin_raw',None)
        if self.pin_piece is not None:
            keep=self.columns!=3*self.pin_piece+2;self.columns=self.columns[keep];T=T[:,keep]
        n=len(self.columns);self.dimension=n
        c=AreaConstraints(p);g,j=c.evaluate(np.r_[p.ravel(),radius]);active=g<1e-7;J=j[active,:45]
        # Fixed raw angle differences admit angular reaction forces. Weight
        # residuals in the actual allowed tangent columns, not the full J.
        U,_,_=np.linalg.svd(J@T,full_matrices=True);w=U[:,-1];w*=np.sign(w.sum())
        if min(w)<=0:raise ValueError('Nonpositive stress')
        self.c=c;self.active=active;self.weights=w
        g,_=c.evaluate(np.r_[p.ravel(),args.target]);g0=g[active];self.budget=float(w@g0)
        self.anchor_units=np.rint(p*1e6).astype(np.int64)
        for (i,k),difference in zip(pairs,self.differences):self.anchor_units[i,2]=self.anchor_units[k,2]+difference
        offset=(self.anchor_units*1e-6-p).ravel()/1e-6;self.b0=-g0/1e-6-J@offset
        C=-w[:,None]*np.column_stack([J[:,0::3].sum(1),J[:,1::3].sum(1)]);Q,_=np.linalg.qr(C);self.projection=np.eye(len(w))-Q@Q.T
        metric=self.projection@(w[:,None]*J)@T;scale=10000000;identity=max(1,round(scale*args.regularization));self.scale=scale;self.identity=identity
        B=IntegerMatrix.from_matrix(np.c_[np.eye(n)*identity,np.rint(metric.T*scale)].astype(np.int64).tolist());LLL.reduction(B,delta=.99)
        if args.block_size:BKZ.reduction(B,BKZ.Param(block_size=args.block_size,max_loops=2))
        M=GSO.Mat(B,update=True);self.M=M;self.moves=np.ascontiguousarray(np.rint(np.array([list(row)for row in B],dtype=float)[:,:n]/identity).astype(np.int64)@T.T)
        self.mu=np.ascontiguousarray([[M.get_mu(i,k)if k<i else 0 for k in range(n)]for i in range(n)])
        r=np.array([M.get_r(i,i)for i in range(n)]);star=np.array([M.to_canonical(np.eye(n)[i])for i in range(n)]);self.gso_transform=np.ascontiguousarray((star/r[:,None]).T)
    def sample(self,rng,count,args,decoder):
        alpha=np.full(len(self.weights),args.concentration,dtype=float);alpha[np.flatnonzero(self.active)>=105]*=args.boundary_bias
        desired=self.budget*rng.dirichlet(alpha,size=count)/self.weights[None,:]/1e-6
        contact=(self.weights[None,:]*(desired+self.b0))@self.projection.T
        wanted=rng.normal(0,args.movement,(count,self.dimension))
        ambient=np.rint(np.c_[wanted*self.identity,contact*self.scale]).astype(np.int64)
        actual,units,coeff,ambiguous=decoder.decode(ambient@self.gso_transform,self.mu,self.moves,self.anchor_units)
        for k in np.flatnonzero(ambiguous):coeff[k]=self.M.babai(ambient[k].tolist())
        if np.any(ambiguous):
            # Reconstruct both raw input and scoring geometry from patched coefficients.
            from semicircle_packing.geometry import canonicalize_theta
            for k in np.flatnonzero(ambiguous):
                delta=(coeff[k]@self.moves).reshape(15,3);delta[:,:2]-=delta[0,:2].copy();u=self.anchor_units+delta
                u[:,2]=np.rint([canonicalize_theta(t*1e-6)*1e6 for t in u[:,2]]).astype(np.int64);units[k]=u.ravel();actual[k]=u*1e-6
        if self.pin_piece is not None:
            units[:,3*self.pin_piece+2]=self.anchor_units[self.pin_piece,2]
            actual[:,self.pin_piece,2]=self.anchor_units[self.pin_piece,2]*1e-6
        return actual,units,coeff,ambient,int(ambiguous.sum())
    def raw(self,coeff):
        delta=(coeff@self.moves).reshape(15,3);delta[:,:2]-=delta[0,:2].copy()
        raw=(self.anchor_units+delta)*1e-6
        if self.pin_raw is not None:raw[self.pin_piece,2]=self.pin_raw
        return raw


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--inputs',nargs='+',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seconds',type=float,default=300);ap.add_argument('--seed',type=int,default=110474);ap.add_argument('--target',type=float,default=2.94852695);ap.add_argument('--goal',type=float,default=2.9485269238);ap.add_argument('--batch',type=int,default=2048);ap.add_argument('--block-size',type=int,default=20);ap.add_argument('--movement',type=float,default=50);ap.add_argument('--concentration',type=float,default=10);ap.add_argument('--boundary-bias',type=float,default=5);ap.add_argument('--regularization',type=float,default=1e-5);ap.add_argument('--verify',type=int,default=128);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);start=time.monotonic();rng=np.random.default_rng(a.seed);decoder=NativeBabai();decoder.reset();seeds=[]
    for path in a.inputs:
        p=load(path);cx,cy,r=compute_mec_fast(p);p[:,:2]-=[cx,cy];c=AreaConstraints(p);g,_=c.evaluate(np.r_[p.ravel(),r]);pairs=flat_contact_pairs(p,c,g);ds=np.array([p[i,2]-p[k,2]for i,k in pairs])
        if len(pairs)!=2 or np.max(abs(ds-np.round(ds,6)))>2e-11:raise ValueError(f'{path}: needs two exact raw integer flat differences')
        cells=wrap_cells(p,pairs);seeds.append((p,r,pairs,cells))
    counts=dict(frames=0,proposals=0,unique=0,branch_reject=0,ambiguous=0,checks=0,elite_valid=0);best=checked(load('solution.json')).score;nearest=float('inf');last=start;elite_best=best+1e-7
    def emit(event,**data):
        row=dict(event=event,seconds=time.monotonic()-start,**data);line=json.dumps(row);print(line,flush=True)
        with (a.output/'events.jsonl').open('a')as f:f.write(line+'\n')
    emit('start',inputs=a.inputs,dimension=43,parameters=vars(a)|{'output':str(a.output)})
    while time.monotonic()-start<a.seconds:
        seed=counts['frames']%len(seeds);p,r,pairs,cells=seeds[seed];widths=np.array([hi-lo for lo,hi in cells]);lo,hi=cells[int(rng.choice(len(cells),p=widths/widths.sum()))];phi=rng.uniform(lo,hi);cs,sn=np.cos(phi),np.sin(phi);posed=p.copy();posed[:,:2]=p[:,:2]@np.array([[cs,sn],[-sn,cs]]);posed[:,2]=np.arctan2(np.sin(p[:,2]+phi),np.cos(p[:,2]+phi))
        try:frame=Frame(posed,r,pairs,a)
        except ValueError:continue
        actual,units,coeff,ambient,patches=frame.sample(rng,a.batch,a,decoder);counts['ambiguous']+=patches;counts['proposals']+=len(actual);counts['frames']+=1
        if counts['frames']<=len(seeds)and a.verify:
            count=min(a.verify,len(coeff));expected=np.array([frame.M.babai(v.tolist())for v in ambient[:count]]);mismatches=int(np.any(expected!=coeff[:count],axis=1).sum());emit('decoder_audit',seed=seed,tested=count,mismatches=mismatches)
            if mismatches:raise AssertionError('Decoder mismatch')
        mask=np.ones(len(actual),dtype=bool)
        for (i,k),d in zip(pairs,frame.differences):mask&=(units[:,3*i+2]-units[:,3*k+2])==d
        counts['branch_reject']+=int((~mask).sum());keep,_=decoder.unique(units);keep&=mask;actual=actual[keep];coeff=coeff[keep]
        if not len(actual):continue
        areas,circles,ok=evaluate_batch(actual,centers=[0.,0.]);counts['unique']+=len(actual)
        violation=np.maximum(np.max((areas-1e-6)/frame.c.scale,axis=1),circles[:,2]-a.target);index=int(np.argmin(violation))
        if violation[index]<nearest:
            nearest=float(violation[index]);raw=frame.raw(coeff[index]);save(a.output/'nearest-raw.json',raw);np.savez(a.output/'nearest-state.npz',placements=raw,geometry=actual[index],origin=posed,moves=frame.moves.reshape(43,15,3),center=circles[index,:2],target=a.target);emit('nearest',violation=nearest,radius=float(circles[index,2]),seed=seed,counts=counts.copy())
        # Empirical screen only: preserve a guard wider than measured estimator
        # error, then let exact GEOS decide every accepted candidate.
        eligible=np.flatnonzero((areas.max(1)<=1e-6+1e-11)&(circles[:,2]<elite_best-2e-12))
        eligible=eligible[np.argsort(circles[eligible,2])]
        for index in eligible:
            raw=frame.raw(coeff[index]);v=checked(raw);counts['checks']+=1
            if v.valid and v.score<elite_best:
                elite_best=v.score;counts['elite_valid']+=1;name=f"elite-{counts['elite_valid']:03d}";save(a.output/(name+'.json'),raw);np.savez(a.output/(name+'.npz'),placements=raw,geometry=actual[index],origin=posed,moves=frame.moves.reshape(43,15,3),center=circles[index,:2],target=a.target);emit('elite',score=v.score,path=str(a.output/(name+'.json')),seed=seed)
            if v.valid and v.score<best:
                save(a.output/'candidate.json',raw);response=check_solution(json.loads((a.output/'candidate.json').read_text()));(a.output/'candidate-api.json').write_text(json.dumps(response,indent=2)+'\n');emit('api',**response)
                if response.get('valid')and response['score']<best:best=response['score'];save(a.output/'best.json',raw)
                if best<a.goal:break
        if best<a.goal:break
        if time.monotonic()-last>10:emit('progress',best=best,nearest=nearest,counts=counts.copy());last=time.monotonic()
    result=dict(best=best,nearest=nearest,counts=counts,goal_met=best<a.goal);(a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');emit('done',**result)


if __name__=='__main__':main()
