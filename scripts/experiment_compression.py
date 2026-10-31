"""Inflate the container to unlock collective motion, then slowly compress."""
from __future__ import annotations
import os
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,time
import numpy as np
from scripts.experiment_basins import FastGeometry,read,write,mutate,relax,polish


def main():
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=120);p.add_argument('--iterations',type=int,default=1000000);p.add_argument('--seed',type=int,default=62);p.add_argument('--input',default='solution.json');p.add_argument('--prefix',default='experiments/strategy_compression');a=p.parse_args()
    rng=np.random.default_rng(a.seed);g=FastGeometry();best=read(a.input);best,best_r,_,_=polish(g,best,2.95)
    pool=[(best.copy(),best_r)];start=time.monotonic();run=0
    while run<a.iterations and time.monotonic()-start<a.seconds:
        base,br=pool[rng.integers(len(pool))]
        trial=mutate(base,rng,run%6);trial[:,:2]*=1.07
        if run%4==0:
            az=rng.uniform(-np.pi,np.pi,15);r=2.3*np.sqrt(rng.uniform(0,1,15));trial=np.c_[r*np.cos(az),r*np.sin(az),rng.uniform(-np.pi,np.pi,15)]
        last_r=3.3
        for radius in np.arange(3.3,2.92,-.02):
            trial,loss,nit=relax(g,trial,radius,maxiter=400)
            last_r=radius
            if loss>3e-5:break
        x,r,e,it=polish(g,trial,last_r,maxiter=250)
        if e>-1e-7 and r<3.1:
            if all(abs(r-pr)>1e-6 for _,pr in pool):
                pool.append((x.copy(),r));pool.sort(key=lambda v:v[1]);pool=pool[:20]
            if r<best_r-1e-8:
                best,best_r=x,r;write(a.prefix+'.json',best);print('BEST',run,best_r,e,'elapsed',time.monotonic()-start,flush=True)
        if run%50==0: print('run',run,'r',r,'best',best_r,'pool',len(pool),'elapsed',time.monotonic()-start,flush=True)
        run+=1
    print('done',run,best_r,'elapsed',time.monotonic()-start,flush=True)
if __name__=='__main__':main()
