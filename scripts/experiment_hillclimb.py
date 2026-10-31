"""Discrete contact-neighborhood search: exact areas, dual MEC screening."""
from __future__ import annotations
import os
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,itertools,time
from pathlib import Path
import numpy as np
from semicircle_packing.polish import load,save,polygon_areas,checked
from semicircle_packing.exact_area import intersection_areas,intersection_areas_and_jacobian
from semicircle_packing.fast_geometry import FastGeometry
from semicircle_packing.geometry import Semicircle
from semicircle_packing.scoring import compute_mec


def dual_weights(base,center,radius):
    g=FastGeometry();p=base.copy();p[:,:2]-=center
    gap,j=g.boundary(np.r_[p.ravel(),radius]);active=np.flatnonzero(gap<1e-4)
    normal=np.array([[-j[k,3*(k//3)],-j[k,3*(k//3)+1]]for k in active])
    weights=[]
    for ids in itertools.combinations(range(len(active)),3):
        mat=np.vstack([normal[list(ids)].T,np.ones(3)])
        if abs(np.linalg.det(mat))<1e-9:continue
        ww=np.linalg.solve(mat,np.array([0.,0.,1.]))
        if min(ww)>=-1e-9:
            v=np.zeros(len(active));v[list(ids)]=ww;weights.append(v)
    return active,np.array(weights).T


def boundary_radii(p,center,active):
    xx=p[:,:,0]-center[0];yy=p[:,:,1]-center[1];tt=p[:,:,2]
    sn=np.sin(tt);co=np.cos(tt)
    ep=np.sqrt((xx+sn)**2+(yy-co)**2)
    em=np.sqrt((xx-sn)**2+(yy+co)**2)
    rr=np.hypot(xx,yy)+1
    all_r=np.stack([ep,em,rr],axis=2).reshape(len(p),45)
    return all_r[:,active]


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--input',default='experiments/original.json');pa.add_argument('--output',default='experiments/discrete_hill.json');pa.add_argument('--seconds',type=float,default=120);pa.add_argument('--seed',type=int,default=81);pa.add_argument('--triples',type=int,default=100000);a=pa.parse_args();rng=np.random.default_rng(a.seed)
    base=np.round(load(a.input),6);r=compute_mec([Semicircle(*q)for q in base])[2];start=time.monotonic();run=0
    pair=[]
    for i in range(45):
        for s in (-2,-1,1,2):
            v=np.zeros(45);v[i]=s;pair.append(v)
        for j in range(i):
            for s,t in itertools.product((-1,1),repeat=2):
                v=np.zeros(45);v[i]=s;v[j]=t;pair.append(v)
    pair=np.array(pair)
    while time.monotonic()-start<a.seconds:
        cx,cy,r=compute_mec([Semicircle(*q)for q in base]);center=np.array([cx,cy]);active,dual=dual_weights(base,center,r)
        areas,jac=intersection_areas_and_jacobian(base);actual=polygon_areas(base);correction=areas-actual
        if run==0 or run%4==0:delta=pair.copy()
        else:
            ids=np.argsort(rng.random((a.triples,45)),axis=1)[:,:3]
            delta=np.zeros((a.triples,45));delta[np.arange(a.triples)[:,None],ids]=rng.choice([-1,1],size=(a.triples,3))
        proposed=base[None,:,:]+delta.reshape(-1,15,3)*1e-6
        predicted=np.max(boundary_radii(proposed,center,active)@dual,axis=1)
        indices=np.flatnonzero(predicted<r-1e-11)
        # Linearization is a cheap screen; actual area rejects corner crossings.
        model=actual[None,:]+(delta[indices]*1e-6)@jac.T
        indices=indices[np.max(model,axis=1)<=1.0002e-6]
        indices=indices[np.argsort(predicted[indices])]
        count=0;won=False
        for idx in indices:
            p=proposed[idx]
            newareas=intersection_areas(p)-correction
            if max(newareas)>1.00005e-6:continue
            count+=1
            validation=checked(p)
            if validation.valid and validation.score<r-1e-11:
                base=np.round(p,6);r=validation.score;save(a.output,base);won=True
                print('BEST',run,r,'move',delta[idx].nonzero()[0].tolist(),'tested',count,'eligible',len(indices),'elapsed',time.monotonic()-start,flush=True)
                break
        if not won:print('pass',run,'score',r,'eligible',len(indices),'tested',count,'elapsed',time.monotonic()-start,flush=True)
        run+=1
    print('done',run,r,flush=True)
if __name__=='__main__':main()
