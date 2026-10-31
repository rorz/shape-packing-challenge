"""Parallel-tempered nonlinear search in saved reduced lattice bases."""
import os
for key in ('VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import argparse,json,math,time
from pathlib import Path
import numpy as np
from semicircle_packing.polish import AreaConstraints,checked,save
from semicircle_packing.fast_mec import compute_mec_fast
from arena import check_solution
LEADER=2.9485269238856113

def canon(u):
 u=u.copy(); t=u[:,2]*1e-6;u[:,2]=np.rint(np.arctan2(np.sin(t),np.cos(t))*1e6).astype(np.int64);return u

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seconds',type=float,default=120);ap.add_argument('--seed',type=int,default=71239);ap.add_argument('--checkpoint',default='experiments/lll_hillclimb_projected_state.npz');ap.add_argument('--output',default='experiments/tempered_best.json');ap.add_argument('--target',type=float,default=LEADER-2e-12);ap.add_argument('--min-temperature',type=float,default=.02);ap.add_argument('--max-temperature',type=float,default=50);ap.add_argument('--replicas',type=int,default=10);ap.add_argument('--quench-every',type=int,default=0);ap.add_argument('--weight-power',type=float,default=.5);ap.add_argument('--radius-weight',type=float,default=3);a=ap.parse_args()
 rng=np.random.default_rng(a.seed);start=time.monotonic();state=np.load(a.checkpoint);p=state['placements'].copy();origin=state['origin'].copy();moves=np.rint(state['moves']).astype(np.int64);c=AreaConstraints(origin);N=c.fast.n_pairs;count=0;best=math.inf;bestv=math.inf;verified=math.inf;lastreport=start
 temperatures=np.geomspace(a.min_temperature,a.max_temperature,a.replicas)
 def evaluate(u,cutoff=math.inf,center=None):
  nonlocal count
  u=canon(u);p=u*1e-6;g,_=c.evaluate(np.r_[p.ravel(),a.target]);neg=np.minimum(g[:N],0)*1e8;areas=float(neg@neg);count+=1
  if areas>cutoff: return areas,math.inf,u,center,None
  cx,cy,R=compute_mec_fast(p,initial_center=center);excess=max(0,(R-a.target)*1e8);E=areas+a.radius_weight*excess**2
  violation=max(0,-g[:N].min(),R-a.target)
  return E,violation,u,np.array([cx,cy]),R
 def record(x):
  nonlocal best,bestv,verified
  E,v,u,center,R=x
  if E<best:
   best=E;bestv=v;save(Path(a.output).with_name(Path(a.output).stem+'_nearest.json'),u*1e-6)
   print(json.dumps(dict(event='best',energy=E,violation=v,radius=R,evaluations=count,seconds=time.monotonic()-start)),flush=True)
  if R is not None and R<min(LEADER,verified):
   p=u*1e-6;g,_=c.evaluate(np.r_[p.ravel(),a.target]);
   if g[:N].min()>=-2e-10:
    v=checked(p)
    if v.valid and v.score<min(LEADER,verified):
     verified=v.score;save(a.output,p);r=check_solution([dict(x=float(x),y=float(y),theta=float(t))for x,y,t in p]);Path(a.output).with_name(Path(a.output).stem+'_api.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(event='verified',**r)),flush=True)
     if r['valid'] and r['score']<LEADER:return True
  return False
 base=evaluate(np.rint(p*1e6).astype(np.int64));replicas=[base for _ in temperatures];beststate=base;record(base);iteration=accepted=swaps=0
 # Estimate initial energy along each lattice direction and bias proposals toward softer moves.
 ranks=[]
 for m in moves:
  vals=[evaluate(base[2]+s*m)[0] for s in (-1,1)];ranks.append(min(vals)-base[0])
 weights=1/(np.maximum(0,ranks)+1)**a.weight_power;weights[np.abs(np.array(ranks))<1e-3]=0;weights=weights/weights.sum()
 while time.monotonic()-start<a.seconds:
  iteration+=1
  for i,T in enumerate(temperatures):
   x=replicas[i];number=int(rng.choice([1,2,3,4],p=[.55,.30,.12,.03]));ids=rng.choice(len(moves),number,replace=False,p=weights if rng.random()<.8 else None);signs=rng.choice([-1,1],number);direction=np.sum(moves[ids]*signs[:,None,None],axis=0)
   if rng.random()<.03:direction*=int(rng.integers(2,5))
   threshold=x[0]-T*math.log(rng.random());trial=evaluate(x[2]+direction,cutoff=threshold,center=x[3])
   if trial[0]<=threshold:
    replicas[i]=trial;accepted+=1
    if trial[0]<beststate[0]:beststate=trial
    if record(trial):return
  if a.quench_every and iteration%a.quench_every==0:
   i=int(rng.integers(len(replicas)));x=replicas[i]
   for sweep in range(6):
    better=x
    for k in rng.permutation(len(moves)):
     for sign in rng.permutation([-1,1]):
      trial=evaluate(x[2]+sign*moves[k],cutoff=better[0],center=x[3])
      if trial[0]<better[0]-1e-7:better=trial
    if better is x:break
    x=better
    if x[0]<beststate[0]:beststate=x
    if record(x):return
   replicas[i]=x
  if iteration%4==0:
   for i in range(iteration//4%2,len(replicas)-1,2):
    x,y=replicas[i:i+2];delta=(1/temperatures[i]-1/temperatures[i+1])*(x[0]-y[0])
    if delta>=0 or math.log(rng.random())<delta:replicas[i],replicas[i+1]=y,x;swaps+=1
  if iteration%500==0:
   # Cold restart of some replicas, retaining hot chains for barrier crossings.
   for i in range(len(replicas)//3): replicas[i]=beststate
  if time.monotonic()-lastreport>10:
   print(json.dumps(dict(event='progress',evaluations=count,iterations=iteration,accepted=accepted,swaps=swaps,best=best,best_violation=bestv,energies=[x[0]for x in replicas],seconds=time.monotonic()-start)),flush=True);lastreport=time.monotonic()
 print(json.dumps(dict(event='done',evaluations=count,iterations=iteration,accepted=accepted,swaps=swaps,best=best,best_violation=bestv,seconds=time.monotonic()-start)),flush=True)
if __name__=='__main__':main()
