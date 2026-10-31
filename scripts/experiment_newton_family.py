"""Fast contact-simplex projection across the soft packing family."""
import json,time
from pathlib import Path
import numpy as np
from semicircle_packing.polish import AreaConstraints
from semicircle_packing.geometry import Semicircle
from semicircle_packing.scoring import validate_and_score
from semicircle_packing.exact_area import intersection_areas
from semicircle_packing.fast_mec import compute_mec_fast
from rounding_search import write_solution

m=np.load('experiments/soft_modes.npz');base=m['p'];radius=float(m['radius']);R=2.9485269237
c=AreaConstraints(base);g,j=c.evaluate(np.r_[base.ravel(),radius]);active=g<1e-8
U,S,V=np.linalg.svd(j[active,:45],full_matrices=True);w=U[:,-1];w*=np.sign(w.sum());pinv=(V[:40].T/S[:40])@U[:,:40].T;B=V[40:]
D=np.zeros((2,45));D[0,3*1+2]=1;D[0,3*5+2]=-1;D[1,3*6+2]=1;D[1,3*10+2]=-1
_,_,T=np.linalg.svd(D@B.T,full_matrices=True);B=T[2:]@B
rng=np.random.default_rng(1675355);start=time.perf_counter();samples=eligible=valid=0;best=float('inf');bestgap=-float('inf')
for family in range(10000):
 alpha=rng.normal(size=len(B))*rng.choice([.0002,.0005,.001,.002,.005]);p=base+(alpha@B).reshape(15,3)
 distribution=np.ones(41) if family%2 else rng.lognormal(0,.8,41)
 for iteration in range(12):
  gaps,jac=c.evaluate(np.r_[p.ravel(),R]);budget=w@gaps[active]
  if budget<1e-9:
   p=base+(p-base)*.7;continue
  target=distribution*budget/(w@distribution)
  delta=pinv@(target-gaps[active]);p+=delta.reshape(15,3)
 gaps,jac=c.evaluate(np.r_[p.ravel(),R])
 if gaps.min()<-1e-9:continue
 count=8192;ang=rng.uniform(-np.pi,np.pi,count);cs=np.cos(ang)[:,None];sn=np.sin(ang)[:,None];shift=rng.uniform(-.5e-6,.5e-6,(count,1,2))
 posed=np.empty((count,15,3));posed[:,:,0]=p[:,0]*cs-p[:,1]*sn;posed[:,:,1]=p[:,0]*sn+p[:,1]*cs;posed[:,:,:2]+=shift;posed[:,:,2]=p[:,2]+ang[:,None]
 rounded=np.round(posed,6);rounded[:,:,2]=np.round(np.arctan2(np.sin(rounded[:,:,2]),np.cos(rounded[:,:,2])),6)
 err=rounded-posed;ex=err[:,:,0].copy();ey=err[:,:,1].copy();err[:,:,0]=ex*cs+ey*sn;err[:,:,1]=-ex*sn+ey*cs;err[:,:,2]=np.arctan2(np.sin(err[:,:,2]),np.cos(err[:,:,2]))
 pred=gaps[:105]+err.reshape(count,45)@jac[:105,:45].T
 chosen=np.flatnonzero(pred.min(axis=1)>-1e-8);samples+=count;eligible+=len(chosen)
 for ix in chosen:
  q=rounded[ix];areas=intersection_areas(q)
  if np.max(areas-c.allowance)>3e-11:continue
  score=compute_mec_fast(q)[2]
  if score>=2.9485269238856113:continue
  result=validate_and_score([Semicircle(*a) for a in q])
  print('CHECK',family,result.valid,result.score,flush=True)
  if result.valid:
   valid+=1
   if result.score<best:
    best=result.score;write_solution(Path('experiments/newton_family_winner.json'),q);raise SystemExit('WINNER '+str(best))
 if family%10==0:print(json.dumps({'family':family,'samples':samples,'eligible':eligible,'valid':valid,'elapsed':time.perf_counter()-start,'budget':budget,'min_gap':gaps.min()}),flush=True)
