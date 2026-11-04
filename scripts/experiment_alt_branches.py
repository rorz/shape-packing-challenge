"""Audit and continue both signs of the two opposed-flat angle branches."""
from __future__ import annotations
import os
for key in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from semicircle_packing.polish import AreaConstraints,load
from semicircle_packing.fast_mec import compute_mec_fast

PAIRS=[(1,5),(6,10)]
def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def write_full(path,p):dump(path,[dict(zip(('x','y','theta'),map(float,row)))for row in p])
def wrap(a):return np.arctan2(np.sin(a),np.cos(a))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',default='experiments/soft_modes.npz');ap.add_argument('--output',default='experiments/2026-11-04/alternative/branches');ap.add_argument('--seconds',type=float,default=180);ap.add_argument('--first',default='-60,-40,-25,-23,-15,0,15,23,25,40,60');ap.add_argument('--second',default='-5,-1,0,1,5');ap.add_argument('--maxiter',type=int,default=160);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
 base=np.load(a.input)['p'].copy() if a.input.endswith('.npz') else load(a.input);cx,cy,r=compute_mec_fast(base);base[:,:2]-=[cx,cy]
 D=np.zeros((2,46))
 for row,(i,j)in enumerate(PAIRS):D[row,3*i+2]=1;D[row,3*j+2]=-1
 center=D@np.r_[base.ravel(),r];center-=wrap(center-np.pi)
 objective=np.zeros(46);objective[-1]=1;c=AreaConstraints(base);summary=[]
 first=list(map(float,a.first.split(',')));second=list(map(float,a.second.split(',')))
 # Follow each sign from the original continuous packing, avoiding a prior branch's accumulated deformation.
 for i in first:
  for j in second:
   if time.monotonic()-start>a.seconds:break
   target=center+np.array([i,j])*1e-6;z=np.r_[base.ravel(),r+1e-6]
   result=minimize(lambda z:z[-1],z,jac=lambda z:objective,method='SLSQP',constraints=[dict(type='ineq',fun=c.fun,jac=c.jac),dict(type='eq',fun=lambda z:D@z-target,jac=lambda z:D)],options=dict(maxiter=a.maxiter,ftol=2e-13))
   p=result.x[:-1].reshape(15,3);g=c.fun(result.x);cx,cy,R=compute_mec_fast(p);name=f'f{i:+07.2f}_s{j:+06.2f}'.replace('.','p');write_full(out/(name+'.json'),p)
   row=dict(first_micro=i,second_micro=j,success=bool(result.success),message=str(result.message),iterations=int(result.nit),radius=float(result.x[-1]),actual_radius=R,violation=float(max(0,-g.min())),equality_error=float(np.max(np.abs(D@result.x-target))),path=str(out/(name+'.json')),seconds=time.monotonic()-start)
   summary.append(row);print(json.dumps(row),flush=True);dump(out/'summary.json',summary)
  if time.monotonic()-start>a.seconds:break
 print(json.dumps(dict(event='done',cases=len(summary),seconds=time.monotonic()-start)),flush=True)
if __name__=='__main__':main()
