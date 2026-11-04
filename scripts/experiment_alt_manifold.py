"""Follow exact contact equations and measure the two nonrigid soft modes."""
from __future__ import annotations
import os
for key in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
from semicircle_packing.polish import AreaConstraints,load
from semicircle_packing.fast_mec import compute_mec_fast

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',default='experiments/2026-11-04/alternative/current-branch/f+023p35_s-00p65.json');ap.add_argument('--output',default='experiments/2026-11-04/alternative/manifold');ap.add_argument('--extent',type=float,default=.002);ap.add_argument('--seconds',type=float,default=30);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
 p=load(a.input);cx,cy,R=compute_mec_fast(p);p[:,:2]-=[cx,cy];z0=np.r_[p.ravel(),R];c=AreaConstraints(p);g,j=c.evaluate(z0);active=g<1e-8
 D=np.zeros((2,46));D[0,5]=1;D[0,17]=-1;D[1,20]=1;D[1,32]=-1;A=np.vstack([j[active],D]);u,s,v=np.linalg.svd(A,full_matrices=True);rank=int((s>1e-9).sum());null=v[rank:];rotation=np.zeros(46);rotation[0:45:3]=-p[:,1];rotation[1:45:3]=p[:,0];rotation[2:45:3]=1;rotation/=np.linalg.norm(rotation);_,_,mix=np.linalg.svd((null@rotation)[None,:],full_matrices=True);modes=mix[1:]@null
 if len(modes)!=2:raise ValueError(f'Expected two nonrigid modes, found{len(modes)}; contacts={active.sum()},rank={rank}')
 def solve(alpha):
  z=z0+modes.T@alpha
  for iteration in range(30):
   gaps,jac=c.evaluate(z);delta=z-z0;F=np.r_[gaps[active],D@delta,modes@delta-alpha,rotation@delta];JJ=np.vstack([jac[active],D,modes,rotation])
   if np.max(np.abs(F))<2e-12:break
   step=np.linalg.solve(JJ,-F);norm=np.linalg.norm(F)
   for power in range(10):
    trial=z+step*.5**power;gg,_=c.evaluate(trial);dd=trial-z0;FF=np.r_[gg[active],D@dd,modes@dd-alpha,rotation@dd]
    if np.linalg.norm(FF)<norm:z=trial;break
   else:break
  gaps,_=c.evaluate(z);return z,dict(alpha=alpha.tolist(),radius=float(z[-1]),violation=float(max(0,-gaps.min())),iterations=iteration+1,residual=float(np.max(np.abs(F))))
 rows=[];values={};grid=np.linspace(-a.extent,a.extent,5)
 for i,x in enumerate(grid):
  for k,y in enumerate(grid):
   if time.monotonic()-started>a.seconds:break
   z,row=solve(np.array([x,y]));name=f'm{i}-{k}.json';(out/name).write_text(json.dumps([dict(zip(('x','y','theta'),map(float,q)))for q in z[:-1].reshape(15,3)],indent=2)+'\n');row['path']=str(out/name);rows.append(row);values[i,k]=z[-1]
 h=a.extent/2;zero=values[2,2];H=np.array([[(values[3,2]+values[1,2]-2*zero)/h**2,(values[3,3]-values[3,1]-values[1,3]+values[1,1])/(4*h**2)],[(values[3,3]-values[3,1]-values[1,3]+values[1,1])/(4*h**2),(values[2,3]+values[2,1]-2*zero)/h**2]])
 eigenvalues,eigenvectors=np.linalg.eigh(H);budget=2.9485269238856113-zero;summary=dict(radius=zero,contacts=int(active.sum()),rank=rank,mode_count=len(modes),hessian=H.tolist(),eigenvalues=eigenvalues.tolist(),soft_extent_at_leader=(np.sqrt(2*budget/eigenvalues)).tolist(),max_violation=max(x['violation']for x in rows),max_radius_model_error=max(abs(x['radius']-(zero+.5*np.array(x['alpha'])@H@np.array(x['alpha'])))for x in rows),seconds=time.monotonic()-started,points=rows)
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');np.savez(out/'modes.npz',origin=z0,modes=modes,active=active,rotation=rotation);print(json.dumps({k:v for k,v in summary.items()if k!='points'},indent=2))
if __name__=='__main__':main()
