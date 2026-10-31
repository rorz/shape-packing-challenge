"""Search symmetric contact topologies, then release symmetry to polish."""
import os
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import time, argparse
import numpy as np
from scipy.optimize import minimize
from scripts.experiment_basins import FastGeometry,write,polish

class Symmetry:
    def __init__(self,fold):
        self.fold=fold;self.m=15//fold
        self.a=np.arange(fold)*2*np.pi/fold
        self.co=np.cos(self.a);self.si=np.sin(self.a)
    def unpack(self,z):
        p=z[:-1].reshape(self.m,3)
        x=np.empty((self.m,self.fold,3))
        x[:,:,0]=p[:,0,None]*self.co-p[:,1,None]*self.si
        x[:,:,1]=p[:,0,None]*self.si+p[:,1,None]*self.co
        x[:,:,2]=p[:,2,None]+self.a
        return np.r_[x.ravel(),z[-1]]
    def jac(self,j):
        jj=j[:,:-1].reshape(-1,self.m,self.fold,3)
        grad=np.empty((len(j),self.m,3))
        grad[:,:,0]=np.sum(jj[:,:,:,0]*self.co+jj[:,:,:,1]*self.si,axis=2)
        grad[:,:,1]=np.sum(-jj[:,:,:,0]*self.si+jj[:,:,:,1]*self.co,axis=2)
        grad[:,:,2]=np.sum(jj[:,:,:,2],axis=2)
        return np.c_[grad.reshape(len(j),-1),j[:,-1]]

def main():
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=120);p.add_argument('--seed',type=int,default=3);p.add_argument('--iterations',type=int,default=1000000)
    a=p.parse_args();rng=np.random.default_rng(a.seed);start=time.monotonic();g=FastGeometry();best=10;run=0
    while run<a.iterations and time.monotonic()-start<a.seconds:
        fold=3 if run%2 else 5;sym=Symmetry(fold)
        az=rng.uniform(-np.pi,np.pi,sym.m);r=np.linspace(.65,2.15,sym.m)+rng.normal(0,.15,sym.m)
        z=np.r_[np.c_[r*np.cos(az),r*np.sin(az),az+rng.uniform(-np.pi,np.pi,sym.m)].ravel(),3.1]
        for weight in (20.,200.,2000.):
            def obj(v):
                gap,jac=g.evaluate(sym.unpack(v));neg=np.minimum(gap,0);grad=weight*(neg@sym.jac(jac));grad[-1]+=1
                return v[-1]+.5*weight*(neg@neg),grad
            result=minimize(obj,z,jac=True,method='L-BFGS-B',options=dict(maxiter=250,ftol=1e-13,gtol=1e-6,maxls=30));z=result.x
        expanded=sym.unpack(z); x,r,e,it=polish(g,expanded[:-1].reshape(15,3),expanded[-1],maxiter=200)
        if e>-1e-7 and r<best:
            best=r;write('experiments/strategy_symmetry.json',x)
            print('BEST',run,'fold',fold,best,e,'elapsed',time.monotonic()-start,flush=True)
        if run%100==0:print('run',run,'fold',fold,'r',r,'err',e,'best',best,'elapsed',time.monotonic()-start,flush=True)
        run+=1
    print('done',run,best,flush=True)
if __name__=='__main__': main()
