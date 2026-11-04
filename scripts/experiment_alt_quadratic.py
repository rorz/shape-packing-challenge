"""Curvature-aware integer search with a branch-locked quadratic contact model."""
from __future__ import annotations
import os
for key in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
from pyscipopt import Model,quicksum,SCIP_PARAMEMPHASIS
from semicircle_packing.polish import AreaConstraints,load,checked,save
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.geometry import canonicalize_theta
from arena import check_solution
LEADER=2.9485269238856113
BEST=2.948526966725367

def flat_clearance(p,i,j):
 na=np.array([np.cos(p[i,2]),np.sin(p[i,2])]);ta=np.array([-na[1],na[0]]);nb=np.array([np.cos(p[j,2]),np.sin(p[j,2])]);tb=np.array([-nb[1],nb[0]]);den=na@nb;slope=-(ta@nb)/den;d=p[i,:2]-p[j,:2];intercept=-(d@nb)/den;offset=-d@ta;width=abs(ta@tb);lo=max(-1.,offset-width);hi=min(1.,offset+width);length=hi-lo;dlo=float(offset-width>-1);dhi=float(offset+width<1);q=abs(slope);tol=1e-6
 if length<=0:raise ValueError('Separated flat segments outside inversion domain')
 if tol<=.5*q*length**2:
  height=np.sqrt(2*q*tol);dh=0.
 else:
  height=tol/length+.5*q*length;dh=-tol/length**2+.5*q
 tip=hi if slope>=0 else lo;dtip=dhi if slope>=0 else dlo;limit=height-slope*tip;dl=dh*(dhi-dlo)-slope*dtip;derivative=np.zeros(45);derivative[3*i:3*i+2]=-dl*ta+nb/den;derivative[3*j:3*j+2]=dl*ta-nb/den;derivative[3*i+2]=dl*(d@na)+(d@tb)/den
 return (limit-intercept)/np.sqrt(2),derivative/np.sqrt(2)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',default='solution.json');ap.add_argument('--output',default='experiments/2026-11-04/alternative/quadratic');ap.add_argument('--seconds',type=float,default=120);ap.add_argument('--steps',type=int,default=50);ap.add_argument('--seed',type=int,default=110401);ap.add_argument('--difference',type=float,nargs=2);ap.add_argument('--hessian-step',type=float,default=.1);ap.add_argument('--rounds',type=int,default=1);ap.add_argument('--clearance-power',action='store_true');ap.add_argument('--flat-inversion',action='store_true');ap.add_argument('--reduced',action='store_true');ap.add_argument('--coefficients',type=int,default=4);ap.add_argument('--basis-regularization',type=float,default=1e-4);ap.add_argument('--stress-aggregate',action='store_true');ap.add_argument('--feasibility-emphasis',action='store_true');a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.monotonic();rng=np.random.default_rng(a.seed)
 base=np.round(load(a.input),6)
 if a.difference is not None:
  base[1,2]=round(base[5,2]+a.difference[0],6);base[6,2]=round(base[10,2]+a.difference[1],6)
 cx,cy,initial_radius=compute_mec_fast(base);center=np.array([cx,cy]);constraint=AreaConstraints(base);target=LEADER-2e-12
 independent=[i for i in range(45)if i not in (0,1,5,20)];n=len(independent);T=np.zeros((45,n+2))
 for column,index in enumerate(independent):T[index,column]=1
 T[5,independent.index(17)]=1;T[20,independent.index(32)]=1;T[0::3,-2]=-1;T[1::3,-1]=-1
 def native(y):
  local=base.copy();local[:,:2]-=center;local+=(T@y).reshape(15,3)*1e-6;g,j=constraint.evaluate(np.r_[local.ravel(),target]);g=g.copy();j=j.copy()
  if a.clearance_power:
   N=constraint.fast.n_pairs;area=np.maximum(0,constraint.allowance-g[:N]*constraint.scale);power=np.full(N,2/3);pairs=[(i,k)for i in range(15)for k in range(i+1,15)]
   for pair in [(1,5),(6,10)]:power[pairs.index(pair)]=.5
   ratio=(np.maximum(area,1e-30)/constraint.allowance)**(power-1);g[:N]=(constraint.allowance**power-area**power)/(power*constraint.allowance**(power-1)*constraint.scale);j[:N]*=ratio[:,None]
  if a.flat_inversion:
   pairs=[(i,k)for i in range(15)for k in range(i+1,15)]
   for ii,jj in [(1,5),(6,10)]:
    gap,derivative=flat_clearance(local,ii,jj);row=pairs.index((ii,jj));g[row]=gap;j[row,:45]=derivative
  return g*1e9,(j[:,:45]@T)*1000
 records=[]
 for iteration in range(a.rounds):
  remaining=a.seconds-(time.monotonic()-start)
  if remaining<=0:break
  y0=np.zeros(n+2);g0,L=native(y0);active=(g0<max(1e5,a.steps*1e4))&(np.linalg.norm(L,axis=1)>1e-7);ids=np.flatnonzero(active);H=np.empty((len(g0),n+2,n+2));h=a.hessian_step
  for column in range(n+2):
   direction=np.zeros(n+2);direction[column]=h;H[:,:,column]=(native(direction)[1]-native(-direction)[1])/(2*h)
  H=(H+H.transpose(0,2,1))*.5;H[np.abs(H)<1e-7]=0
  # Measure model error on randomized bounded directions before asking SCIP to search it.
  model_errors=[];false_feasible=[];false_infeasible=[];violation_errors=[]
  for _ in range(30):
   y=rng.integers(-min(a.steps,20),min(a.steps,20)+1,n+2).astype(float);actual=native(y)[0];pred=g0+L@y+.5*np.einsum('i,kij,j->k',y,H,y);model_errors.append(float(np.max(np.abs(actual[ids]-pred[ids]))));aa=actual[ids];pp=pred[ids];false_feasible.append(float(max(0,-aa[pp>=0].min()))if np.any(pp>=0)else 0);false_infeasible.append(float(max(0,-pp[aa>=0].min()))if np.any(aa>=0)else 0);mask=(aa<0)|(pp<0);violation_errors.append(float(np.max(np.abs(aa[mask]-pp[mask])))if mask.any()else 0)
  model=Model('packing_quadratic')
  if a.feasibility_emphasis:model.setEmphasis(SCIP_PARAMEMPHASIS.FEASIBILITY)
  model.setParam('limits/time',max(.1,remaining));model.setParam('parallel/maxnthreads',1);model.setParam('randomization/randomseedshift',a.seed+iteration);model.setParam('display/verblevel',0);model.setParam('numerics/feastol',1e-7)
  variables=[model.addVar(f'u{k}',vtype='I'if k<n and not a.reduced else'C',lb=-a.steps*(1 if k<n else 2),ub=a.steps*(1 if k<n else 2))for k in range(n+2)];coefficients=[];basis=None
  if a.reduced:
   from fpylll import IntegerMatrix,LLL
   uu,ss,vv=np.linalg.svd(L[ids],full_matrices=False);weights=np.abs(uu[:,-1]);weighted_center=weights[:,None]*L[ids,n:];Q,_=np.linalg.qr(weighted_center);projection=np.eye(len(ids))-Q@Q.T;WJ=projection@(weights[:,None]*L[ids,:n]/1000);scale=10000000;identity=max(1,round(scale*a.basis_regularization));embedded=np.column_stack([np.eye(n)*identity,np.round(WJ.T*scale)]).astype(np.int64);B=IntegerMatrix.from_matrix(embedded.tolist());LLL.reduction(B,delta=.99);basis=np.rint(np.array([list(row)for row in B],dtype=float)[:,:n]/identity).astype(np.int64);np.save(out/f'basis-{iteration}.npy',basis)
   coefficients=[model.addVar(f'k{k}',vtype='I',lb=-a.coefficients,ub=a.coefficients)for k in range(n)]
   for k in range(n):model.addCons(variables[k]==quicksum(int(basis[j,k])*coefficients[j]for j in range(n)if basis[j,k]),name=f'lattice_{k}')
  slack=model.addVar('violation_nm',vtype='C',lb=-1000,ub=max(10000,float(-g0.min()+1000)),obj=1)
  terms=0;expressions=[]
  for row in ids:
   linear=quicksum(float(L[row,k])*variables[k]for k in np.flatnonzero(L[row]))
   quadratic=quicksum(float(H[row,i,j]*(.5 if i==j else 1))*variables[i]*variables[j]for i,j in zip(*np.triu_indices(n+2))if H[row,i,j]!=0)
   terms+=int(np.count_nonzero(np.triu(H[row])));expression=float(g0[row])+linear+quadratic;expressions.append(expression);model.addCons(expression+slack>=0,name=f'contact_{row}')
  if a.stress_aggregate:
   uu,ss,vv=np.linalg.svd(L[ids],full_matrices=False);stress=np.abs(uu[:,-1]);stress/=stress.sum();model.addCons(quicksum(float(w)*expression for w,expression in zip(stress,expressions))+slack>=0,name='positive_stress_aggregate')
  warm=model.createSol()
  for v in variables+coefficients:model.setSolVal(warm,v,0)
  model.setSolVal(warm,slack,float(max(0,-g0[ids].min())+1e-3));warm_added=model.addSol(warm)
  setup=dict(event='setup',iteration=iteration,integer_variables=n,continuous_variables=2,contacts=len(ids),quadratic_terms=terms,max_hessian=float(np.abs(H[ids]).max()),model_error_nm_max=max(model_errors),model_error_nm_median=float(np.median(model_errors)),false_feasible_nm_max=max(false_feasible),false_infeasible_nm_max=max(false_infeasible),violation_error_nm_max=max(violation_errors),warm_added=bool(warm_added),initial_radius=initial_radius,seconds=time.monotonic()-start)
  print(json.dumps(setup),flush=True);(out/f'setup-{iteration}.json').write_text(json.dumps(setup,indent=2)+'\n');model.optimize();sol=model.getBestSol()
  if sol is None:
   record=dict(event='done',status=str(model.getStatus()),nodes=int(model.getNNodes()),seconds=time.monotonic()-start);records.append(record);print(json.dumps(record),flush=True);break
  y=np.array([model.getSolVal(sol,v)for v in variables]);y[:n]=np.rint(y[:n])if basis is None else np.rint([model.getSolVal(sol,k)for k in coefficients])@basis;physical=np.zeros(45)
  for column,index in enumerate(independent):physical[index]=y[column]
  physical[5]=physical[17];physical[20]=physical[32];candidate=base+physical.reshape(15,3)*1e-6;candidate[:,2]=[canonicalize_theta(t)for t in candidate[:,2]];save(out/f'candidate-{iteration}.json',candidate)
  actual,j=native(y);cx,cy,R=compute_mec_fast(candidate);v=checked(candidate);record=dict(event='candidate',iteration=iteration,status=str(model.getStatus()),nodes=int(model.getNNodes()),dual_bound=float(model.getDualbound()),linearized_violation_nm=float(model.getSolVal(sol,slack)),actual_violation_nm=float(max(0,-actual.min())),radius=R,valid=v.valid,score=v.score,errors=v.errors,seconds=time.monotonic()-start);records.append(record);print(json.dumps(record),flush=True)
  if v.valid and v.score<BEST:
   response=check_solution([dict(x=float(x),y=float(y),theta=float(t))for x,y,t in candidate]);(out/f'api-{iteration}.json').write_text(json.dumps(response,indent=2)+'\n');print(json.dumps(dict(event='api',**response)),flush=True)
   if response.get('valid') and response.get('score',99)<LEADER:break
  base=candidate;center=np.array([cx,cy]);initial_radius=R
 (out/'summary.json').write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__':main()
