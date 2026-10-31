"""Coordinate and short-cluster repair around a tight-radius lattice witness."""
import json,itertools
from pathlib import Path
import numpy as np
from semicircle_packing.polish import load,AreaConstraints,checked,save
from semicircle_packing.fast_mec import compute_mec_fast
from arena import check_solution

p=load('experiments/sweep_repair_614_c10/solution.json');c=AreaConstraints(p);cx,cy,_=compute_mec_fast(p);R=2.9485269238
pairs=np.array([(i,j) for i in range(15) for j in range(i+1,15)])

def evaluate(q):
 q=q.copy();q[:,:2]-=[cx,cy];return c.evaluate(np.r_[q.ravel(),R])[0]
for iteration in range(10):
 before=evaluate(p);bad=np.flatnonzero(before[:105]<-1e-10)
 print('ITER',iteration,'bad',[(pairs[k].tolist(),float(before[k])) for k in bad],flush=True)
 if not len(bad):
  result=checked(p);print('CHECK',result.valid,result.score,flush=True)
  if result.valid and result.score<2.9485269238856113:
   save('experiments/local_repair_winner.json',p);print('API',check_solution(json.loads(Path('experiments/local_repair_winner.json').read_text())),flush=True);raise SystemExit
 order=[1,9,6,11,7,0,2,5,10]
 changed=False
 for i in order:
  related=np.flatnonzero((pairs==i).any(axis=1));mask=np.r_[related,np.arange(105+3*i,105+3*i+3)]
  current=evaluate(p);best=float(np.minimum(current[mask],0).dot(np.minimum(current[mask],0)));bestq=None
  if best<1e-25:continue
  for dx,dy,dt in itertools.product(range(-2,3),range(-2,3),range(-10,11)):
   q=p.copy();q[i]+=np.array([dx,dy,dt])*1e-6;q=np.round(q,6);g=evaluate(q)
   loss=float(np.minimum(g[mask],0).dot(np.minimum(g[mask],0)))
   if loss<best-1e-25:best,bestq=loss,q
  if bestq is not None:p=bestq;changed=True;print('MOVE',i,best,flush=True)
 if not changed:break
save('experiments/local_repair_nearest.json',p)
