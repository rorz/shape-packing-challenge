"""Explore soft continuous modes, then solve local lattice feasibility."""
import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from semicircle_packing.polish import load,AreaConstraints
from lattice_polish import solve
from rounding_search import area_model,write_solution
from semicircle_packing.exact_area import intersection_areas_and_jacobian
rng=np.random.default_rng(2837545)
modes=np.load('experiments/soft_modes.npz');base=modes['p'];basis=modes['basis'];constraint=AreaConstraints(base)
t0=time.perf_counter()
for trial in range(100):
 wanted=(rng.normal(size=len(basis))@basis)*rng.choice([500,1500,5000,15000])
 last=None;cached=None
 def evaluate(u):
  global last,cached
  if last is not None and np.array_equal(last,u):return cached
  p=base+u.reshape(15,3)*1e-6
  g,j=constraint.evaluate(np.r_[p.ravel(),2.94852670 if trial%2 else 2.94852680]);cached=g*1e6,j[:,:45];last=u.copy();return cached
 opt=minimize(lambda u:.5*np.sum((u-wanted)**2),np.zeros(45),jac=lambda u:u-wanted,constraints={'type':'ineq','fun':lambda u:evaluate(u)[0],'jac':lambda u:evaluate(u)[1]},method='SLSQP',options={'maxiter':100,'ftol':1e-7})
 wanted=opt.x.copy()
 opt=minimize(lambda u:.5*np.sum((u-wanted)**2),wanted,jac=lambda u:u-wanted,constraints={'type':'ineq','fun':lambda u:evaluate(u)[0],'jac':lambda u:evaluate(u)[1]},method='SLSQP',options={'maxiter':100,'ftol':1e-10})
 p=base+opt.x.reshape(15,3)*1e-6
 print('SEED',trial,opt.success,-min(evaluate(opt.x)[0]),np.linalg.norm(opt.x),flush=True)
 angle=rng.uniform(-np.pi,np.pi);c,s=np.cos(angle),np.sin(angle);p[:,:2]=p[:,:2]@np.array([[c,s],[-s,c]]);p[:,2]+=angle;p[:,2]=np.round(np.arctan2(np.sin(np.round(p[:,2],6)),np.cos(np.round(p[:,2],6))),6);p=np.round(p,6)
 steps=[5,15,40][trial%3]
 cuts=[]
 pairlist,baseareas,basejac=area_model(p)
 exactbase,_=intersection_areas_and_jacobian(p)
 lexpairs=[(i,j) for i in range(15) for j in range(i+1,15)]
 for ai,bi in [(1,5),(6,10)]:
  areaindex=lexpairs.index((ai,bi));polyindex=pairlist.index((bi,ai));correction=exactbase[areaindex]-baseareas[polyindex]
  normal=np.array([np.cos(p[ai,2]),np.sin(p[ai,2])])
  for offset in np.linspace(-2*steps,2*steps,7):
   for turn in np.linspace(-2*steps,2*steps,7):
    anchor=p.copy();anchor[ai,:2]+=normal*offset*1e-6;anchor[ai,2]+=turn*1e-6
    ar,jj=intersection_areas_and_jacobian(anchor)
    cuts.append((anchor,np.array([ar[areaindex]-correction-5e-10*(steps/15)**2]),jj[areaindex:areaindex+1]))
 for it in range(6):
  q,info=solve(p,steps=steps,seconds=5,margin=1e-12,cuts=cuts,target=2.9485269237)
  print(json.dumps({'trial':trial,'iter':it,'elapsed':time.perf_counter()-t0,**info}),flush=True)
  if q is None:break
  if info['valid']:
   write_solution(Path('experiments/soft_lattice_winner.json'),q)
   raise SystemExit('WINNER '+str(info['score']))
  _,areas,jac=area_model(q);active=areas>1e-6-1e-12
  cuts.append((q.copy(),areas[active],jac[active]))
