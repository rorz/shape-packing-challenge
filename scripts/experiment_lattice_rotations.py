"""Independent multi-frame integer contact search; every improvement is checked."""
import json,time
from pathlib import Path
import numpy as np
from lattice_polish import solve
from rounding_search import area_model,write_solution

rng=np.random.default_rng(7621)
base=np.array([[p[k] for k in ('x','y','theta')] for p in json.load(open('experiments/continuous.json'))])
best=2.9485270638716834
t0=time.perf_counter()
for trial in range(30):
 angle=0 if trial==0 else rng.uniform(-np.pi,np.pi)
 c,s=np.cos(angle),np.sin(angle)
 p=base.copy();p[:,:2]=p[:,:2]@np.array([[c,s],[-s,c]])
 p[:,:2]+=rng.uniform(-.5e-6,.5e-6,2);p[:,2]+=angle;p=np.round(p,6)
 cuts=[]
 for iteration in range(5):
  q,info=solve(p,steps=3 if trial%2 else 7,seconds=8,margin=1e-12,cuts=cuts,target=2.9485269238)
  print(json.dumps({'trial':trial,'round':iteration,'angle':angle,'elapsed':time.perf_counter()-t0,**info}),flush=True)
  if q is None:break
  if info['valid'] and info['score']<best:
   best=info['score'];write_solution(Path('experiments/rotated_lattice_best.json'),q)
   print('IMPROVED',best,flush=True)
   if best<2.9485269238856113:raise SystemExit
  _,areas,jac=area_model(q)
  active=areas>1e-6-1e-12
  if active.any():cuts.append((q.copy(),areas[active],jac[active]))
  else:p=q;cuts=[]
