import numpy as np,json,time
from pathlib import Path
from semicircle_packing.polish import load
from semicircle_packing.geometry import canonicalize_theta
from lattice_polish import solve
from rounding_search import area_model,write_solution
p=load('experiments/original.json');t=time.perf_counter()
for reflected in (False,True):
 for k in range(4):
  q=p.copy()
  if reflected:q[:,1]*=-1;q[:,2]*=-1
  for _ in range(k):q[:,:2]=q[:,:2][:,[1,0]]*np.array([-1,1])
  q[:,2]=[canonicalize_theta(v+k*np.pi/2) for v in q[:,2]];cuts=[]
  for it in range(6):
   v,info=solve(q,steps=15,seconds=5,cuts=cuts,target=2.9485269237,margin=1e-12)
   print(json.dumps({'reflect':reflected,'quadrant':k,'round':it,'elapsed':time.perf_counter()-t,**info}),flush=True)
   if v is None:break
   if info['valid'] and info['score']<2.9485269238856113:write_solution(Path('experiments/quadrant_winner.json'),v);raise SystemExit('WINNER')
   _,a,j=area_model(v);active=a>1e-6-1e-12;cuts.append((v.copy(),a[active],j[active]))
