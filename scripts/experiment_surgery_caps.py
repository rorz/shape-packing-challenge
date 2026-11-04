"""Split a disk into unequal circular segments and transfer area between them."""
from __future__ import annotations
import os
for key in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse,itertools,json,time
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon
from experiment_surgery import Geometry,optimize,relax,write_full,fingerprint
from experiment_surgery_split import cut_angles
from semicircle_packing.polish import load,polish as area_polish,checked,save,AreaConstraints
from arena import check_solution

def cap_polygon(p,cut,points=2049):
 beta=np.arccos(cut);angle=np.linspace(p[2]-beta,p[2]+beta,points);return Polygon(np.c_[p[0]+np.cos(angle),p[1]+np.sin(angle)])
def audit(out):
 rng=np.random.default_rng(811042);g=Geometry([1,.6,1],cuts=[.65,-.4,.2]);p=np.array([[-1.2,.1,.4],[.2,-.4,-.3],[1.,.5,2.1]]);z=np.r_[p.ravel(),3.2];v,j=g.evaluate(z);errors=[];h=1e-7
 for k in range(len(z)):
  d=np.zeros(len(z));d[k]=h;fd=(g.fun(z+d)-g.fun(z-d))/(2*h);errors.append(float(np.max(np.abs(fd-j[:,k]))))
 mismatches=0;containment_undershoot=0.;union_errors=[]
 for _ in range(200):
  cuts=rng.uniform(-.85,.85,2);p=rng.normal(size=(2,3));p[:,:2]*=1.5;g=Geometry([1,1],cuts=cuts);gaps=g.fun(np.r_[p.ravel(),4]);polys=[cap_polygon(p[i],cuts[i])for i in range(2)];overlap=polys[0].intersection(polys[1]).area
  if abs(gaps[0])>1e-5 and((gaps[0]<0)!=(overlap>1e-9)):mismatches+=1
  for i in range(2):
   beta=np.arccos(cuts[i]);angle=np.linspace(p[i,2]-beta,p[i,2]+beta,2049);sample=np.max(np.hypot(p[i,0]+np.cos(angle),p[i,1]+np.sin(angle)));radius=4-min(gaps[1+3*i:1+3*(i+1)]);containment_undershoot=max(containment_undershoot,float(sample-radius))
 for cut in [-.8,-.4,.4,.8]:
  a=cap_polygon([0,0,0],cut,8193);b=cap_polygon([0,0,np.pi],-cut,8193);union_errors.append(abs(a.union(b).area-np.pi))
 result=dict(seed=811042,derivative_error=max(errors),polygon_overlap_mismatches=mismatches,containment_undershoot=containment_undershoot,complement_union_area_error=max(union_errors));(out/'cap-kernel-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(event='cap_audit',**result)),flush=True)
 if max(errors)>1e-6 or mismatches or containment_undershoot>1e-9:raise ValueError('Cap audit failed')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seconds',type=float,default=180);ap.add_argument('--seed',type=int,default=8110403);ap.add_argument('--source',default='experiments/2026-11-04/round2/topology/split');ap.add_argument('--output',default='experiments/2026-11-04/round2/topology/caps');ap.add_argument('--audit-only',action='store_true');a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);audit(out)
 if a.audit_only:return
 rng=np.random.default_rng(a.seed);started=time.monotonic();macros=[]
 for path in sorted(Path(a.source).glob('fusion-*.json')):
  p=load(path);n=len(p);g=Geometry(np.ones(n),np.r_[np.zeros(n-1,dtype=int),1]);gg=g.fun(np.r_[p.ravel(),0]);R=-gg[-3*n:].min()
  if R<3.08:macros.append((str(path),p,float(R)))
 cases=[]
 for k,(_,p,R)in enumerate(macros):
  cuts,_=cut_angles(p,R)
  for cut in cuts:
   for asymmetry in [.8,-.8,.4,-.4]:
    for stagger in [.3,-.3,.85,-.85]:cases.append((k,cut,asymmetry,stagger))
 rng.shuffle(cases);unit=Geometry(np.ones(15));baseline=2.948680774628151;best=baseline;bestarea=float('inf');trials=0
 with(out/'trials.jsonl').open('w')as stream:
  for macro_id,cut,asymmetry,stagger in itertools.cycle(cases):
   if time.monotonic()-started>a.seconds:break
   source,macro,R=macros[macro_id];center=macro[-1,:2];t=np.array([-np.sin(cut),np.cos(cut)]);p=np.vstack([macro[:-1],np.r_[center+t*stagger*.5,cut],np.r_[center-t*stagger*.5,cut+np.pi]]);stages=[];currentR=R
   for coefficient in [1.,.75,.5,.25,0.]:
    offset=asymmetry*coefficient;g=Geometry(np.ones(15),cuts=np.r_[np.zeros(13),offset,-offset]);p,loss,lit=relax(g,p,max(3.08,currentR+.025),iterations=120);p,currentR,gap,nit=optimize(g,p,max(3.08,currentR+.025),iterations=120);stages.append(dict(offset=offset,radius=currentR,gap=gap,loss=loss,relax_iterations=lit,iterations=nit))
   trials+=1;finger=fingerprint(unit,p,currentR);row=dict(trial=trials,seed=a.seed,source=source,cut=cut,initial_offset=asymmetry,stagger=stagger,stages=stages,radius=currentR,gap=gap,fingerprint=finger,seconds=time.monotonic()-started)
   if gap>-2e-7 and currentR<best:best=currentR;row['strict_best']=True;write_full(out/'strict-best.json',p)
   if gap>-2e-7 and currentR<baseline+2e-6:
    result=area_polish(p,maxiter=180);q=result.x[:-1].reshape(15,3);pr=float(result.x[-1]);violation=float(max(0,-AreaConstraints(q).fun(result.x).min()));row.update(area_radius=pr,area_violation=violation)
    if violation<2e-8 and pr<bestarea:bestarea=pr;write_full(out/'continuous-best.json',q)
    v=checked(q);row.update(rounded_valid=v.valid,rounded_score=v.score)
    if v.valid and v.score<2.948526966725367:
     save(out/'valid-best.json',q);response=check_solution(json.loads((out/'valid-best.json').read_text()));(out/'api.json').write_text(json.dumps(response,indent=2)+'\n');row['api']=response
     if response.get('valid')and response.get('score',99)<2.9485269238856113:stream.write(json.dumps(row)+'\n');stream.flush();print(json.dumps(row),flush=True);return
   stream.write(json.dumps(row)+'\n');stream.flush();print(json.dumps(row),flush=True)
 print(json.dumps(dict(event='done',trials=trials,macro_seeds=len(macros),strict_best=best,continuous_best=bestarea,seconds=time.monotonic()-started)),flush=True)
if __name__=='__main__':main()
