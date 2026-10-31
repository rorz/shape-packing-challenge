"""Independent basin exploration: exact SAT + centroid-preserving shape moves."""
from __future__ import annotations
import argparse
import os
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import math
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from semicircle_packing.fast_geometry import FastGeometry

CENTROID = 4/(3*math.pi)


def read(path):
    return np.array([[v['x'],v['y'],v['theta']] for v in json.loads(Path(path).read_text())])


def write(path, x):
    Path(path).write_text(json.dumps([dict(x=float(p[0]),y=float(p[1]),theta=float(p[2])) for p in x],indent=2)+'\n')


def centroid_move(x, indices, angle):
    old=x[indices,2].copy()
    x[indices,2] += angle
    x[indices,0] += CENTROID*(np.cos(old)-np.cos(x[indices,2]))
    x[indices,1] += CENTROID*(np.sin(old)-np.sin(x[indices,2]))


def mutate(x,rng,kind):
    x=x.copy()
    if kind==0:
        indices=rng.choice(15,size=rng.integers(1,5),replace=False)
        centroid_move(x,indices,rng.normal(0,.8,len(indices)))
        x[indices,:2] += rng.normal(0,.18,(len(indices),2))
    elif kind==1:
        indices=rng.choice(15,size=2,replace=False)
        centroid_move(x,indices,np.pi*rng.choice([-1,1],2))
    elif kind==2:
        indices=rng.choice(15,size=2,replace=False)
        x[indices,:2]=x[indices[::-1],:2]
        centroid_move(x,indices,rng.uniform(-np.pi,np.pi,2))
    elif kind==3:
        x[:,:2] += rng.normal(0,.14,(15,2))
        centroid_move(x,np.arange(15),rng.normal(0,.35,15))
    elif kind==4:
        indices=rng.choice(15,size=rng.integers(3,7),replace=False)
        centroid_move(x,indices,rng.normal(0,1.4,len(indices)))
        x[indices,:2] += rng.normal(0,.35,(len(indices),2))
    elif kind==5:
        # Cluster twist preserves the internal arrangement, breaks rim contacts.
        anchor=rng.integers(15)
        cent=x[:,:2]+CENTROID*np.c_[np.cos(x[:,2]),np.sin(x[:,2])]
        indices=np.argsort(np.linalg.norm(cent-cent[anchor],axis=1))[:rng.integers(3,7)]
        pivot=np.mean(cent[indices],axis=0)
        a=rng.normal(0,.35)
        rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        x[indices,:2]=(x[indices,:2]-pivot)@rot.T+pivot
        x[indices,2]+=a
    return x


def relax(geometry,x,radius,maxiter=300):
    def f(flat):
        value, grad, _=geometry.penalty(flat.reshape(15,3),radius)
        return value,grad.ravel()
    sol=minimize(f,x.ravel(),method='L-BFGS-B',jac=True,options=dict(maxiter=maxiter,ftol=1e-14,gtol=1e-7,maxls=30))
    return sol.x.reshape(15,3), sol.fun,sol.nit


def polish(geometry,x,radius,maxiter=300):
    z=np.r_[x.ravel(),radius]
    direction=np.zeros(len(z)); direction[-1]=1
    sol=minimize(lambda v:v[-1],z,jac=lambda _:direction,method='SLSQP',constraints=[dict(type='ineq',fun=geometry.constraints,jac=geometry.jacobian)],options=dict(maxiter=maxiter,ftol=2e-11))
    return sol.x[:-1].reshape(15,3),sol.x[-1],float(np.min(geometry.constraints(sol.x))),sol.nit


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=11)
    p.add_argument('--seconds',type=float,default=180)
    p.add_argument('--iterations',type=int,default=1000000)
    p.add_argument('--input',default='solution.json')
    p.add_argument('--prefix',default='experiments/strategy_basins')
    a=p.parse_args()
    rng=np.random.default_rng(a.seed)
    geometry=FastGeometry()
    x=read(a.input)
    start=time.monotonic()
    x,r,err,it=polish(geometry,x,2.95)
    print('initial',r,err,it,flush=True)
    best=x.copy();best_r=r
    pool=[(x.copy(),r)]
    best_loss=1
    run=0
    while run<a.iterations and time.monotonic()-start<a.seconds:
        # Mix current basin, archive of local minima, and larger topology moves.
        base,base_r=pool[rng.integers(len(pool))]
        kind=run%6
        trial=mutate(base,rng,kind)
        target=min(best_r-.002,base_r-.005)
        trial,loss,nit=relax(geometry,trial,target)
        if loss<2e-4:
            candidate,score,gap,nit2=polish(geometry,trial,max(target,base_r))
            if gap > -1e-7:
                if all(abs(score-pr)>1e-7 for _,pr in pool):
                    pool.append((candidate.copy(),score));pool.sort(key=lambda item:item[1]);pool=pool[:12]
                if score<best_r-1e-9:
                    best,best_r=candidate,score
                    write(a.prefix+'.json',best)
                    print('BEST',run,kind,best_r,gap,'pool',len(pool),'elapsed',time.monotonic()-start,flush=True)
        if loss<best_loss:
            best_loss=loss
        if run%100==0:
            print('run',run,'kind',kind,'loss',loss,'best_loss',best_loss,'best',best_r,'pool',len(pool),'elapsed',time.monotonic()-start,flush=True)
        run+=1
    print('done',run,best_r,'elapsed',time.monotonic()-start,flush=True)

if __name__=='__main__': main()
