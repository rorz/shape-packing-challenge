"""Fast contact-simplex projection across the soft packing family."""
import json,time
from pathlib import Path
import numpy as np
from semicircle_packing.polish import AreaConstraints
from semicircle_packing.geometry import Semicircle
from semicircle_packing.scoring import validate_and_score
from semicircle_packing.exact_area import polygon_intersection_areas as intersection_areas
from semicircle_packing.fast_mec import compute_mec_fast
from rounding_search import write_solution

from semicircle_packing.polish import load
base=load('experiments/branch-3.141616-3.141592.json');radius=compute_mec_fast(base)[2];R=2.9485269237
c=AreaConstraints(base);g,j=c.evaluate(np.r_[base.ravel(),radius]);active=g<1e-7
D=np.zeros((2,45));D[0,3*1+2]=1;D[0,3*5+2]=-1;D[1,3*6+2]=1;D[1,3*10+2]=-1
U,S,V=np.linalg.svd(np.vstack([j[active,:45],D]),full_matrices=True)
wfull=U[:,-1];wfull*=np.sign(wfull[:41].sum());w=wfull[:41];wd=wfull[41:]
pinv=(V[:42].T/S[:42])@U[:,:42].T;B=V[42:]
print('MODEL',radius,w.min(),S[-2:],flush=True)
rng=np.random.default_rng(2675355);start=time.perf_counter();samples=eligible=valid=0;best=float('inf');bestgap=-float('inf')
for family in range(10000):
 alpha=rng.normal(size=len(B))*rng.choice([.0002,.0005,.001,.002,.005]);p=base+(alpha@B).reshape(15,3)
 distribution=np.ones(41) if family%2 else rng.lognormal(0,.8,41)
 for iteration in range(12):
  gaps,jac=c.evaluate(np.r_[p.ravel(),R]);dflat=D@(p-base).ravel();budget=w@gaps[active]+wd@dflat
  if budget<1e-9:
   p=base+(p-base)*.7;continue
  target=distribution*budget/(w@distribution)
  delta=pinv@np.r_[target-gaps[active],-dflat];p+=delta.reshape(15,3)
 gaps,jac=c.evaluate(np.r_[p.ravel(),R])
 if gaps.min()<-1e-9:continue
 count=8192;ang=rng.uniform(-np.pi,np.pi,count);cs=np.cos(ang)[:,None];sn=np.sin(ang)[:,None];shift=rng.uniform(-.5e-6,.5e-6,(count,1,2))
 posed=np.empty((count,15,3));posed[:,:,0]=p[:,0]*cs-p[:,1]*sn;posed[:,:,1]=p[:,0]*sn+p[:,1]*cs;posed[:,:,:2]+=shift;posed[:,:,2]=p[:,2]+ang[:,None]
 rounded=np.round(posed,6);rounded[:,:,2]=np.round(np.arctan2(np.sin(rounded[:,:,2]),np.cos(rounded[:,:,2])),6)
 err=rounded-posed;ex=err[:,:,0].copy();ey=err[:,:,1].copy();err[:,:,0]=ex*cs+ey*sn;err[:,:,1]=-ex*sn+ey*cs;err[:,:,2]=np.arctan2(np.sin(err[:,:,2]),np.cos(err[:,:,2]))
 allpred=gaps+err.reshape(count,45)@jac[:,:45].T
 pred=allpred[:,:105]
 ix=int(np.argmax(allpred.min(axis=1)));approx=float(allpred[ix].min())
 if approx>bestgap:
  q=rounded[ix];areas=intersection_areas(q);score=compute_mec_fast(q)[2]
  actual=min(float(((c.allowance-areas)/c.scale).min()),R-score)
  if actual>bestgap:
   bestgap=actual;write_solution(Path('experiments/newton_negative_nearest.json'),q)
   Path('experiments/newton_negative_seed.json').write_text(json.dumps([dict(zip(('x','y','theta'),map(float,row))) for row in p],indent=2)+'\n')
   print('NEAREST',family,bestgap,score,flush=True)
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
    best=result.score;write_solution(Path('experiments/newton_negative_winner.json'),q);raise SystemExit('WINNER '+str(best))
 if family%10==0:print(json.dumps({'family':family,'samples':samples,'eligible':eligible,'valid':valid,'elapsed':time.perf_counter()-start,'budget':budget,'min_gap':gaps.min()}),flush=True)
