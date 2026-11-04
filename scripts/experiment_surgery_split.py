"""Optimize13 semicircles plus a disk, then enumerate cuts and split the disk."""
from __future__ import annotations
import os
for key in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse,itertools,json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from experiment_surgery import Geometry,optimize,relax,write_full,fingerprint,audit
from semicircle_packing.polish import load,polish as area_polish,checked,save,AreaConstraints
from semicircle_packing.fast_mec import compute_mec_fast
from arena import check_solution
LEADER=2.9485269238856113

def disk_optimize(p,R):
 g=Geometry(np.ones(len(p)),np.r_[np.zeros(len(p)-1,dtype=int),1]);z=np.r_[p.ravel(),R];direction=np.zeros(len(z));direction[-1]=1;bounds=[(None,None)]*len(z);bounds[-2]=(0,0);bounds[-1]=(1,4);p[-1,2]=0;z[-2]=0
 sol=minimize(lambda z:z[-1],z,jac=lambda z:direction,method='SLSQP',constraints=dict(type='ineq',fun=g.fun,jac=g.jac),bounds=bounds,options=dict(ftol=2e-11,maxiter=180));return sol.x[:-1].reshape(-1,3),float(sol.x[-1]),float(g.fun(sol.x).min()),int(sol.nit),g

def cut_angles(p,R):
 center=p[-1,:2];bearings=[]
 for row in p[:-1]:
  n=np.array([np.cos(row[2]),np.sin(row[2])]);t=np.array([-n[1],n[0]]);q=center-row[:2]
  if q@n>=0:point=row[:2]+q/np.linalg.norm(q)
  else:point=row[:2]+t*np.clip(q@t,-1,1)
  gap=np.linalg.norm(point-center)-1
  if gap<3e-5:bearings.append(np.arctan2(*(point-center)[::-1]))
 if R-np.linalg.norm(center)-1<3e-5:bearings.append(np.arctan2(center[1],center[0]))
 events=np.unique(np.round((np.array(bearings)-np.pi/2)%np.pi,8));cuts=[]
 if len(events):
  for i,angle in enumerate(events):other=events[(i+1)%len(events)]+(np.pi if i==len(events)-1 else 0);cuts.append(float(((angle+other)*.5)%np.pi))
 cuts.extend(float(v)for v in np.linspace(0,np.pi,8,endpoint=False));return sorted(set(round(v,7)for v in cuts)),len(events)
def disk_audit(out):
 rng=np.random.default_rng(811041);g=Geometry([1,.6,1],[0,0,1]);p=np.array([[-1.2,.1,.4],[.2,-.4,-.3],[1.,.5,0]]);z=np.r_[p.ravel(),3.2];v,j=g.evaluate(z);errors=[];h=1e-7
 for k in range(len(z)):
  d=np.zeros(len(z));d[k]=h;fd=(g.fun(z+d)-g.fun(z-d))/(2*h);errors.append(float(np.max(np.abs(fd-j[:,k]))))
 # A complementary pair at one center must have the same union-separation gap as a disk.
 comparisons=[]
 for _ in range(50):
  theta=rng.uniform(-np.pi,np.pi);other=np.r_[rng.normal(size=2)*2,rng.uniform(-np.pi,np.pi)];center=rng.normal(size=2);unit=Geometry([1,1,1]);disk=Geometry([1,1],[1,0]);gp=unit.fun(np.r_[center,theta,center,theta+np.pi,other,4.]);gd=disk.fun(np.r_[center,0,other,4.]);comparisons.append(max(0,float(gd[0]-min(gp[1],gp[2]))))
 result=dict(seed=811041,max_derivative_error=max(errors),disk_separation_upper_error=max(comparisons));(out/'disk-kernel-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(event='disk_audit',**result)),flush=True)
 if max(errors)>1e-6:raise ValueError('Disk kernel derivative audit failed')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seconds',type=float,default=180);ap.add_argument('--seed',type=int,default=8110402);ap.add_argument('--input',default='experiments/2026-11-04/round2/topology/strict-baseline.json');ap.add_argument('--output',default='experiments/2026-11-04/round2/topology/split');ap.add_argument('--container',type=float,default=3.12);ap.add_argument('--audit-only',action='store_true');a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);disk_audit(out)
 if a.audit_only:return
 rng=np.random.default_rng(a.seed);started=time.monotonic();origin=load(a.input);unit=Geometry(np.ones(15));origin,baseline,_,_=optimize(unit,origin,3.0);best=baseline;best_area=float('inf');trials=0;fusions=0;pool={};pairs=list(itertools.combinations(range(15),2));rng.shuffle(pairs);pairs=[(1,5),(6,10)]+[v for v in pairs if v not in[(1,5),(6,10)]]
 with(out/'trials.jsonl').open('w')as stream:
  for pair in itertools.cycle(pairs):
   if time.monotonic()-started>a.seconds:break
   keep=[i for i in range(15)if i not in pair];disk=np.r_[origin[list(pair),:2].mean(axis=0),0];macro=np.vstack([origin[keep],disk]);macro,R,gap,nit,geometry=disk_optimize(macro,3.15);fusions+=1;finger=fingerprint(geometry,macro,R);key=(finger,round(R,6),round(np.linalg.norm(macro[-1,:2]),5));row=dict(event='fusion',fusion=fusions,removed=list(pair),radius=R,gap=gap,iterations=nit,fingerprint=finger,seconds=time.monotonic()-started)
   if key in pool:row['duplicate']=True;print(json.dumps(row),flush=True);stream.write(json.dumps(row)+'\n');stream.flush();continue
   pool[key]=(macro.copy(),R);write_full(out/f'fusion-{fusions:03d}.json',macro);cuts,contacts=cut_angles(macro,R);row.update(cuts=cuts,contact_events=contacts);print(json.dumps(row),flush=True);stream.write(json.dumps(row)+'\n');stream.flush()
   for cut in cuts:
    for stagger in [0.,.3,-.3,.85,-.85]:
     if time.monotonic()-started>a.seconds:break
     t=np.array([-np.sin(cut),np.cos(cut)]);center=macro[-1,:2];p=np.vstack([macro[:-1],np.r_[center+t*stagger*.5,cut],np.r_[center-t*stagger*.5,cut+np.pi]])
     if stagger:p,loss,relaxit=relax(unit,p,max(a.container,R+.03),iterations=160)
     else:loss=0.;relaxit=0
     p,score,mingap,nit=optimize(unit,p,max(R,a.container),iterations=180);trials+=1;signature=fingerprint(unit,p,score);record=dict(event='split',trial=trials,fusion=fusions,removed=list(pair),cut=cut,stagger=stagger,relax_loss=loss,relax_iterations=relaxit,radius=score,gap=mingap,iterations=nit,fingerprint=signature,seconds=time.monotonic()-started)
     if mingap>-2e-7 and score<best:best=score;write_full(out/'strict-best.json',p);record['strict_best']=True
     if mingap>-2e-7 and score<baseline+2e-6:
      result=area_polish(p,maxiter=180);q=result.x[:-1].reshape(15,3);pr=float(result.x[-1]);violation=float(max(0,-AreaConstraints(q).fun(result.x).min()));record.update(area_radius=pr,area_violation=violation)
      if violation<2e-8 and pr<best_area:best_area=pr;write_full(out/'continuous-best.json',q)
      v=checked(q);record.update(rounded_valid=v.valid,rounded_score=v.score)
      if v.valid and v.score<2.948526966725367:
       save(out/'valid-best.json',q);response=check_solution(json.loads((out/'valid-best.json').read_text()));(out/'api.json').write_text(json.dumps(response,indent=2)+'\n');record['api']=response
       if response.get('valid')and response.get('score',99)<LEADER:stream.write(json.dumps(record)+'\n');stream.flush();print(json.dumps(record),flush=True);return
     stream.write(json.dumps(record)+'\n');stream.flush();print(json.dumps(record),flush=True)
    if time.monotonic()-started>a.seconds:break
 print(json.dumps(dict(event='done',trials=trials,fusions=fusions,unique_fusions=len(pool),strict_best=best,continuous_best=best_area,seconds=time.monotonic()-started)),flush=True)
if __name__=='__main__':main()
