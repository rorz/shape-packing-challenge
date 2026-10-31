"""Radius-pressure continuation, with exact-contact polishing."""
from __future__ import annotations
import os
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse,time
import numpy as np
from scipy.optimize import minimize
from scripts.experiment_basins import FastGeometry,read,write,mutate,polish


def pressure(g,x,r):
    z=np.r_[x.ravel(),r]
    for weight in (20.,200.,2000.):
        def objective(v):
            gap,jac=g.evaluate(v)
            neg=np.minimum(gap,0)
            der=weight*(neg@jac)
            der[-1]+=1
            return v[-1]+.5*weight*(neg@neg),der
        result=minimize(objective,z,jac=True,method='L-BFGS-B',bounds=[(None,None)]*45+[(2.8,3.5)],options=dict(maxiter=350,ftol=1e-13,gtol=1e-6,maxls=30))
        z=result.x
    return z[:-1].reshape(15,3),z[-1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=180);p.add_argument('--seed',type=int,default=98);p.add_argument('--iterations',type=int,default=1000000)
    p.add_argument('--input',default='solution.json');p.add_argument('--prefix',default='experiments/strategy_pressure')
    a=p.parse_args();rng=np.random.default_rng(a.seed);g=FastGeometry();x=read(a.input)
    best,r,_,_=polish(g,x,2.95);best_r=r; pool=[(best.copy(),r)];start=time.monotonic();run=0
    print('start',r,flush=True)
    while run<a.iterations and time.monotonic()-start<a.seconds:
        base,br=pool[rng.integers(len(pool))]
        trial=mutate(base,rng,run%6)
        if run%3==0:
            trial,pr=pressure(g,trial,br+.02)
        else:pr=br+.04
        x,r,err,it=polish(g,trial,pr,maxiter=200)
        if err>-1e-7 and r<3.06:
            if all(abs(r-pr)>1e-6 for _,pr in pool):
                pool.append((x.copy(),r));pool.sort(key=lambda item:item[1]);pool=pool[:15]
            if r<best_r-1e-8:
                best,best_r=x,r;write(a.prefix+'.json',best)
                print('BEST',run,r,err,'pool',len(pool),'elapsed',time.monotonic()-start,flush=True)
        if run%100==0: print('run',run,'candidate',r,'err',err,'best',best_r,'pool',len(pool),'elapsed',time.monotonic()-start,flush=True)
        run+=1
    print('done',run,best_r,flush=True)

if __name__=='__main__':main()
