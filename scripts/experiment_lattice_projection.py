"""Douglas--Rachford search between exact packing feasibility and the 6dp lattice."""
import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from semicircle_packing.polish import AreaConstraints,load
from semicircle_packing.geometry import Semicircle
from semicircle_packing.scoring import compute_mec,validate_and_score
from rounding_search import write_solution

rng=np.random.default_rng(64252)
original=load('experiments/continuous.json')
target=2.9485269237
start=time.perf_counter();best=2.9485270638716834
for frame in range(50):
 angle=0 if frame==0 else rng.uniform(-np.pi,np.pi);c,s=np.cos(angle),np.sin(angle)
 base=original.copy();base[:,:2]=base[:,:2]@np.array([[c,s],[-s,c]])
 base[:,2]+=angle;base=np.round(base,6)
 center=np.array(compute_mec([Semicircle(*p) for p in base])[:2]);rel=base.copy();rel[:,:2]-=center
 cons=AreaConstraints(rel,polygon_correction=True)
 last=None;cached=None
 def eval(u):
  global last,cached
  if last is not None and np.array_equal(u,last):return cached
  p=base+u[:45].reshape(15,3)*1e-6
  p[:,:2]-=center+u[45:]*1e-6
  g,j=cons.evaluate(np.r_[p.ravel(),target]);j=j[:,:45]
  cj=-j.reshape(len(j),15,3)[:,:,:2].sum(axis=1)
  cached=g*1e6,np.column_stack([j,cj]);last=u.copy();return cached
 v=rng.uniform(-.5,.5,47);v[45:]=0;xold=np.zeros(47);smallest=1e9
 for it in range(500):
  point=v.copy()
  def obj(u):
   d=u-point;d[45:]*=1e-3
   return .5*d.dot(d)
  def jac(u):
   d=u-point;d[45:]*=1e-6
   return d
  opt=minimize(obj,xold,jac=jac,constraints={'type':'ineq','fun':lambda u:eval(u)[0],'jac':lambda u:eval(u)[1]},method='SLSQP',options={'maxiter':30,'ftol':1e-8})
  x=opt.x;xold=x
  y=2*x-v;y[:45]=np.round(y[:45]);v+=1.0*(y-x)
  err=np.linalg.norm(x[:45]-y[:45]);smallest=min(smallest,err)
  if it%50==0:print(json.dumps({'frame':frame,'iter':it,'elapsed':time.perf_counter()-start,'error':err,'best_error':smallest,'violation':-min(eval(x)[0]),'success':opt.success}),flush=True)
  check=y.copy();check[45:]=x[45:]
  lattice_g=eval(check)[0]
  if lattice_g[:105].min()>-1e-4:
   candidate=np.round(base+y[:45].reshape(15,3)*1e-6,6)
   result=validate_and_score([Semicircle(*p) for p in candidate])
   print('CHECK',result.valid,result.score,err,flush=True)
   if result.valid and result.score<best:
    best=result.score;write_solution(Path('experiments/projected_lattice_best.json'),candidate)
    if best<target:raise SystemExit('WINNER '+str(best))
