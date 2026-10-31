"""Crossover near-feasible LLL states through an exact contact CSP."""
from __future__ import annotations
import os
for key in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ.setdefault(key,'1')
import argparse,heapq,itertools,time
import numpy as np
from scripts.experiment_contact_csp import load,save,checked,pair_kernel,compatible,find_assignment
from semicircle_packing.polish import AreaConstraints,polygon_areas
from semicircle_packing.exact_area import intersection_areas
from semicircle_packing.geometry import Semicircle
from semicircle_packing.scoring import compute_mec


def canonical(units):
    p=units.copy();t=p[:,2]*1e-6;p[:,2]=np.rint(np.arctan2(np.sin(t),np.cos(t))*1e6).astype(np.int64);return p


def initial_at(choices,q,target):
    masks=[]
    for d in choices:
        x=d[:,0]-q[0];y=d[:,1]-q[1];sn=np.sin(d[:,2]);co=np.cos(d[:,2]);r=np.maximum(np.hypot(x+sn,y-co),np.hypot(x-sn,y+co));r=np.maximum(r,np.where(x*co+y*sn>=0,np.hypot(x,y)+1,0));mask=r<=target;masks.append(int.from_bytes(np.packbits(mask,bitorder='little').tobytes(),'little'))
    return masks


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--checkpoint',default='experiments/lll_hillclimb_best_state.npz');pa.add_argument('--output',default='experiments/discrete_lll_crossover.json');pa.add_argument('--seconds',type=float,default=180);pa.add_argument('--seed',type=int,default=6711);pa.add_argument('--samples',type=int,default=10000);pa.add_argument('--parents',type=int,default=200);pa.add_argument('--circle-headroom',type=float,default=0.0);pa.add_argument('--micro-neighbors',action='store_true');a=pa.parse_args();rng=np.random.default_rng(a.seed);start=time.monotonic();run=0;kernel=pair_kernel();bad_pairs=set();target=2.94852692388
    while time.monotonic()-start<a.seconds:
        state=np.load(a.checkpoint);base=state['origin'];current=canonical(np.rint(state['placements']*1e6).astype(np.int64));moves=np.rint(state['moves']).astype(np.int64);constraint=AreaConstraints(base);cx0,cy0,_=compute_mec([Semicircle(*q)for q in current*1e-6]);fitness_center=np.array([cx0,cy0]);heap=[];seen=set();forced=[current]+[canonical(current+direction*sign)for direction in moves for sign in (-1,1)]
        for k in range(a.samples):
            if k==0:units=current.copy()
            elif k<2*len(moves)+1:units=current+moves[(k-1)//2]*(1 if k%2 else -1)
            else:
                count=int(rng.integers(1,5));ids=rng.choice(len(moves),count,replace=False);signs=rng.choice([-1,1],count);units=current+np.sum(moves[ids]*signs[:,None,None],axis=0)
            units=canonical(units)
            key=units.tobytes()
            if key in seen:continue
            seen.add(key)
            physical=units*1e-6;physical[:,:2]-=fitness_center
            g,_=constraint.evaluate(np.r_[physical.ravel(),target]);neg=np.minimum(g,0);loss=float(neg@neg)
            item=(-loss,k,units)
            if len(heap)<a.parents:heapq.heappush(heap,item)
            elif item[0]>heap[0][0]:heapq.heapreplace(heap,item)
        micro=np.array(list(itertools.product((-1,0,1),repeat=3)),dtype=np.int64) if a.micro_neighbors else np.empty((0,3),dtype=np.int64)
        choices=[np.ascontiguousarray(np.unique(np.array([item[2][i]for item in heap]+[u[i]for u in forced]+[current[i]+dd for dd in micro]),axis=0)*1e-6)for i in range(15)]
        areas=intersection_areas(base);polygons=polygon_areas(base);allpairs=list(itertools.combinations(range(15),2));ids=np.flatnonzero(areas>1e-8);pairs=[allpairs[k]for k in ids];limits=np.where(np.linalg.norm(constraint._areas_and_jac(np.r_[base.ravel(),target])[1],axis=1)[ids]>.1,1.000001e-6,1.0018e-6)
        adj,maps=compatible(choices,pairs,limits,kernel)
        lookup=[{tuple(np.rint(q*1e6).astype(np.int64)):k for k,q in enumerate(domain)}for domain in choices]
        for i,j,ii,jj in bad_pairs:
            vi=lookup[i].get(ii);vj=lookup[j].get(jj)
            if vi is not None and vj is not None:
                maps[i,j][vi]&=~(1<<vj);maps[j,i][vj]&=~(1<<vi)
        bestp=current*1e-6;cx,cy,rr=compute_mec([Semicircle(*q)for q in bestp]);q=np.array([cx,cy]);found=False
        for offset in rng.permutation(np.array(list(itertools.product([-1e-7,0,1e-7],repeat=2)))):
            masks=initial_at(choices,q+offset,target+a.circle_headroom)
            for attempt in range(5):
                if time.monotonic()-start>=a.seconds:return
                answer,nodes=find_assignment(choices,adj,maps,rng,10000,masks)
                if answer is None:break
                candidate=np.array([choices[i][k]for i,k in enumerate(answer)]);validation=checked(candidate)
                print('candidate',run,validation.valid,validation.score,validation.errors,'elapsed',time.monotonic()-start,flush=True)
                if validation.valid and validation.score<2.9485269238856113:
                    save(a.output,candidate);print('WIN',validation.score,flush=True);return
                actual=polygon_areas(candidate)
                for k,(i,j)in enumerate(allpairs):
                    if actual[k]>1e-6 and(i,j)in maps:
                        vi,vj=answer[i],answer[j];maps[i,j][vi]&=~(1<<vj);maps[j,i][vj]&=~(1<<vi)
                        bad_pairs.add((i,j,tuple(np.rint(candidate[i]*1e6).astype(np.int64)),tuple(np.rint(candidate[j]*1e6).astype(np.int64))))
        if run%10==0:print('round',run,'poolloss',-max(heap)[0],'domains',[len(v)for v in choices],'elapsed',time.monotonic()-start,flush=True)
        run+=1
    print('done',run,'elapsed',time.monotonic()-start,flush=True)
if __name__=='__main__':main()
