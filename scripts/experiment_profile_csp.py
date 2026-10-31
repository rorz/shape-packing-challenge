"""Sample shallow continuous shape modes, then solve each local grid CSP."""
from __future__ import annotations
import os
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,itertools,time
import numpy as np
from scipy.optimize import minimize
from scripts.experiment_contact_csp import load,save,checked,pair_kernel,domains,compatible,find_assignment
from semicircle_packing.polish import AreaConstraints,polygon_areas
from semicircle_packing.exact_area import intersection_areas


PRESERVE_BRANCHES=False

def constraints_for(c,base):
    result=[dict(type="ineq",fun=c.fun,jac=c.jac)]
    if PRESERVE_BRANCHES:
        matrix=np.zeros((2,46));matrix[0,5]=1;matrix[0,17]=-1;matrix[1,20]=1;matrix[1,32]=-1
        target=matrix@np.r_[base.ravel(),0.]
        result.append(dict(type="eq",fun=lambda z:matrix@z-target,jac=lambda _:matrix))
    return result


def profile(base,delta):
    c=AreaConstraints(base,1e-6,True);z=np.r_[base.ravel(),2.94853];d=np.zeros(46);d[-1]=1
    bounds=[(None,None)]*46;bounds[2]=(base[0,2],base[0,2])
    values=np.atleast_1d(delta);indices=[9] if len(values)==1 else [1,2,6,9]
    for index,value in zip(indices,values):bounds[3*index+2]=(base[index,2]+value,base[index,2]+value)
    r=minimize(lambda v:v[-1],z,jac=lambda _:d,bounds=bounds,constraints=constraints_for(c,base),method='SLSQP',options=dict(maxiter=120,ftol=2e-13))
    return r.x[:-1].reshape(15,3),r.x[-1],np.min(c.fun(r.x))


def regularized_profile(base,rng):
    c=AreaConstraints(base,1e-6,True);z=np.r_[base.ravel(),2.94852655];seed=base+rng.normal(0,.0008,(15,3));weight=.02
    def objective(v):
        diff=v[:-1]-seed.ravel()
        return v[-1]+.5*weight*np.dot(diff,diff),np.r_[weight*diff,1.]
    r=minimize(objective,z,jac=True,constraints=constraints_for(c,base),method="SLSQP",options=dict(maxiter=200,ftol=2e-13))
    return r.x[:-1].reshape(15,3),r.x[-1],np.min(c.fun(r.x))


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--input',default='experiments/discrete_continuous.json');pa.add_argument('--output',default='experiments/discrete_profile_csp.json');pa.add_argument('--seconds',type=float,default=180);pa.add_argument('--seed',type=int,default=511);pa.add_argument('--steps',type=int,default=3);pa.add_argument('--frames',type=int,default=10);pa.add_argument('--target',type=float,default=2.94852692388);pa.add_argument('--joint',action='store_true');pa.add_argument('--regularized',action='store_true');pa.add_argument('--preserve-flat-branches',action='store_true');a=pa.parse_args();rng=np.random.default_rng(a.seed)
    global PRESERVE_BRANCHES
    PRESERVE_BRANCHES=a.preserve_flat_branches
    origin=load(a.input);kernel=pair_kernel();start=time.monotonic();run=0;frame=0
    while time.monotonic()-start<a.seconds:
        delta=rng.uniform(-.0003,.0003,4) if a.joint else rng.uniform(-.0006,.0006)
        base,radius,gap=regularized_profile(origin,rng) if a.regularized else profile(origin,delta)
        if gap < -1e-8 or radius>a.target:continue
        area=intersection_areas(base);poly=polygon_areas(base);allpairs=list(itertools.combinations(range(15),2));ids=np.flatnonzero(area>1e-8);pairs=[allpairs[k]for k in ids];limits=1e-6+area[ids]-poly[ids]-1e-11
        for _ in range(a.frames):
            angle=rng.uniform(-np.pi,np.pi);c,s=np.cos(angle),np.sin(angle);p=base.copy();p[:,0]=c*base[:,0]-s*base[:,1];p[:,1]=s*base[:,0]+c*base[:,1];p[:,2]=(p[:,2]+angle+np.pi)%(2*np.pi)-np.pi
            shift=rng.uniform(-.5e-6,.5e-6,2);p[:,:2]+=shift;choices=domains(p,a.target,a.steps,shift)
            if not all(len(q)for q in choices):continue
            flat_angles={(i,j):np.round(p[i,2]-p[j,2],6)for i,j in [(1,5),(6,10)]} if PRESERVE_BRANCHES else None
            adj,maps=compatible(choices,pairs,limits,kernel,p,flat_angles);result,nodes=find_assignment(choices,adj,maps,rng,1000)
            if result is not None:
                candidate=np.array([choices[i][k]for i,k in enumerate(result)]);val=checked(candidate)
                print('CANDIDATE',run,frame,'delta',delta,'valid',val.valid,'score',val.score,'errors',val.errors,'elapsed',time.monotonic()-start,flush=True)
                if val.valid and val.score<2.9485269238856113:
                    save(a.output,candidate);print('WIN',val.score,flush=True);return
            frame+=1
        if run%10==0:print('profile',run,'frame',frame,'radius',radius,'delta',delta,'elapsed',time.monotonic()-start,flush=True)
        run+=1
    print('done',run,frame,'elapsed',time.monotonic()-start,flush=True)
if __name__=='__main__':main()
