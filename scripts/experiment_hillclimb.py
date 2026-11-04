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
from semicircle_packing.geometry import Semicircle, canonicalize_theta, semicircle_polygon
from semicircle_packing.scoring import compute_mec
from semicircle_packing.fast_mec import compute_mec_fast


def dual_weights(base,center,radius):
    g=FastGeometry(len(base));p=base.copy();p[:,:2]-=center
    gap,j=g.boundary(np.r_[p.ravel(),radius]);active=np.flatnonzero(gap<1e-4)
    normal=np.array([[-j[k,3*(k//3)],-j[k,3*(k//3)+1]]for k in active])
    valid=np.linalg.norm(normal,axis=1)>1e-12;active=active[valid];normal=normal[valid]
    normal/=np.linalg.norm(normal,axis=1)[:,None]
    weights=[]
    for i,k in itertools.combinations(range(len(active)),2):
        if np.linalg.norm(normal[i]+normal[k])<1e-8:
            v=np.zeros(len(active));v[[i,k]]=.5;weights.append(v)
    for ids in itertools.combinations(range(len(active)),3):
        mat=np.vstack([normal[list(ids)].T,np.ones(3)])
        if abs(np.linalg.det(mat))<1e-9:continue
        ww=np.linalg.solve(mat,np.array([0.,0.,1.]))
        if min(ww)>=-1e-12:
            ww=np.maximum(ww,0);ww/=ww.sum()
            v=np.zeros(len(active));v[list(ids)]=ww;weights.append(v)
    return active,np.array(weights).T if weights else np.empty((len(active),0))


def boundary_radii(p,center,active):
    """Feature distances at a fixed center; these are NOT dual lower bounds."""
    xx=p[:,:,0]-center[0];yy=p[:,:,1]-center[1];tt=p[:,:,2]
    sn=np.sin(tt);co=np.cos(tt)
    ep=np.sqrt((xx+sn)**2+(yy-co)**2)
    em=np.sqrt((xx-sn)**2+(yy+co)**2)
    rr=np.hypot(xx,yy)+1
    all_r=np.stack([ep,em,rr],axis=2).reshape(len(p),3*p.shape[1])
    return all_r[:,active]


def support_lower_bounds(proposed,base,center,active,dual):
    """Sound fixed-direction support bounds, allowing free enclosing centers.

    With nonnegative weights summing to one and sum(w*u)=0, R >= sum(w*h(u)).
    A residual correction covers numerical imbalance in sum(w*u). MEC centers
    lie in the convex hull of the shapes, bounded by max(|piece-center|)+1.
    """
    if not dual.shape[1]:return np.ones(len(proposed))
    relative=base.copy();relative[:,:2]-=center
    _,jac=FastGeometry(len(base)).boundary(np.r_[relative.ravel(),0.])
    directions=np.array([[-jac[k,3*(k//3)],-jac[k,3*(k//3)+1]]for k in active])
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    pieces=active//3;theta=proposed[:,pieces,2]
    co,sn=np.cos(theta),np.sin(theta);ux,uy=directions.T
    curved=co*ux+sn*uy>=0
    unit_support=np.where(curved,1.,np.abs(-sn*ux+co*uy))
    position=proposed[:,pieces,:2]-center
    support=np.einsum('bki,ki->bk',position,directions)+unit_support
    imbalance=np.linalg.norm(directions.T@dual,axis=0)
    center_bound=np.linalg.norm(proposed[:,:,:2]-center,axis=2).max(axis=1)+1
    bounds=support@dual-center_bound[:,None]*imbalance[None,:]
    # Conservative arithmetic guard; the uncorrected fixed-center distances
    # could instead exceed the true radius by O(displacement**2).
    return bounds.max(axis=1)-1e-12


def exact_overlap_free(candidate,base,polygons,cache):
    """Cache polygon variants and check only pairs incident to changed pieces."""
    changed=np.any(candidate!=base,axis=1);current=list(polygons)
    for i in np.flatnonzero(changed):
        key=(int(i),*map(float,candidate[i]))
        if key not in cache:cache[key]=semicircle_polygon(Semicircle(*candidate[i]))
        current[i]=cache[key]
    for i in range(len(base)):
        for j in range(i):
            if not(changed[i] or changed[j]):continue
            if np.linalg.norm(candidate[i,:2]-candidate[j,:2])>2:continue
            if current[i].intersection(current[j]).area>1e-6:return False
    return True


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--input',default='experiments/original.json');pa.add_argument('--output',default='experiments/discrete_hill.json');pa.add_argument('--seconds',type=float,default=120);pa.add_argument('--seed',type=int,default=81);pa.add_argument('--triples',type=int,default=100000);pa.add_argument('--heuristic-areas',action='store_true',help='Opt into historical tangent/static-deficit screens; these are not exclusion proofs');a=pa.parse_args();rng=np.random.default_rng(a.seed)
    base=np.round(load(a.input),6);base[:,2]=[canonicalize_theta(t)for t in base[:,2]]
    initial=checked(base)
    if not initial.valid:pa.error('The exact cached-neighborhood search requires a valid input packing')
    r=initial.score;start=time.monotonic();run=0
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
        if a.heuristic_areas:
            areas,jac=intersection_areas_and_jacobian(base);actual=polygon_areas(base);correction=areas-actual
        else:polygons=[semicircle_polygon(Semicircle(*q))for q in base];cache={}
        if run==0 or run%4==0:delta=pair.copy()
        else:
            ids=np.argsort(rng.random((a.triples,45)),axis=1)[:,:3]
            delta=np.zeros((a.triples,45));delta[np.arange(a.triples)[:,None],ids]=rng.choice([-1,1],size=(a.triples,3))
        proposed=base[None,:,:]+delta.reshape(-1,15,3)*1e-6
        theta=np.round(proposed[:,:,2],6);proposed[:,:,2]=np.round(np.arctan2(np.sin(theta),np.cos(theta)),6)
        predicted=support_lower_bounds(proposed,base,center,active,dual)
        indices=np.flatnonzero(predicted<r-1e-11)
        if a.heuristic_areas:
            model=actual[None,:]+(delta[indices]*1e-6)@jac.T
            indices=indices[np.max(model,axis=1)<=1.0002e-6]
        indices=indices[np.argsort(predicted[indices])]
        count=0;area_checks=0;won=False
        for idx in indices:
            if time.monotonic()-start>a.seconds:break
            p=proposed[idx]
            area_checks+=1
            if a.heuristic_areas:
                newareas=intersection_areas(p)-correction
                if max(newareas)>1.00005e-6:continue
            elif not exact_overlap_free(p,base,polygons,cache):continue
            count+=1
            if compute_mec_fast(p)[2]>=r-1e-11:continue
            validation=checked(p)
            if validation.valid and validation.score<r-1e-11:
                base=np.round(p,6);r=validation.score;save(a.output,base);won=True
                print('BEST',run,r,'move',delta[idx].nonzero()[0].tolist(),'area_checked',area_checks,'area_valid',count,'eligible',len(indices),'elapsed',time.monotonic()-start,flush=True)
                break
        if not won:print('pass',run,'score',r,'eligible',len(indices),'area_checked',area_checks,'area_valid',count,'elapsed',time.monotonic()-start,flush=True)
        run+=1
    print('done',run,r,flush=True)
if __name__=='__main__':main()
