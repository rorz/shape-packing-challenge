"""Discover all subset reflections at each tested diameter using conflict components.

Reflecting both members preserves a pair's overlap. Reflecting exactly one has
the same cross-overlap in either direction. Thus every incompatible cross pair
forces equal reflection bits. Every feasible subset is a union of components.
Native areas are a permissive discovery screen; GEOS certifies saved seeds.
"""
from __future__ import annotations
import os
for key in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse,itertools,json,time
from pathlib import Path
import numpy as np
from semicircle_packing.polish import load,polygon_areas,polish
from semicircle_packing.exact_area import polygon_intersection_areas
from semicircle_packing.fast_mec import compute_mec_fast
from experiment_surgery_inventory import signature,write


def components(cross,limit):
    n=len(cross);parent=list(range(n))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i in range(n):
        for j in range(i):
            if max(cross[i,j],cross[j,i])>limit:parent[root(i)]=root(j)
    groups={}
    for i in range(n):groups.setdefault(root(i),[]).append(i)
    return list(groups.values())


def reflected(p,axis):
    normal=np.array([-np.sin(axis),np.cos(axis)]);q=p.copy();q[:,:2]-=2*np.outer(p[:,:2]@normal,normal);q[:,2]=2*axis-p[:,2];return q


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',default='experiments/2026-11-04/round2/topology/inventory/baseline.json');ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seconds',type=float,default=90);ap.add_argument('--axes',type=int,default=200000);ap.add_argument('--relaxation',type=float,default=3e-11);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);start=time.monotonic();p=load(a.input);cx,cy,R=compute_mec_fast(p);p[:,:2]-=[cx,cy];n=len(p);N=2*n;u=np.arange(n)[:,None];v=np.arange(n)[None,:]+n;cross_indices=u*(2*N-u-1)//2+v-u-1
    # Axes exchanging any two boundary features are deliberately included;
    # isolated exact symmetries are easily missed by uniform random sampling.
    features=list(p[:,:2])
    for x,y,t in p:
        for s in (-1,1):features.append(np.array([x-s*np.sin(t),y+s*np.cos(t)]))
    angles=np.arctan2(np.array(features)[:,1],np.array(features)[:,0]);axes=list(angles%np.pi)
    axes.extend((x+y)*.5%np.pi for x,y in itertools.combinations(angles,2))
    axes.extend(np.pi*(np.arange(a.axes)+.5)/a.axes);axes=list(dict.fromkeys(round(float(x),14)for x in axes))
    known=[signature(p)]
    mirror=Path('experiments/2026-11-04/round2/topology/continuous-best.json')
    if mirror.exists():known.append(signature(load(mirror)))
    counts=dict(axes=0,disconnected=0,subsets=0,known=0,geos_checks=0,repairs=0,new=0);records=[];seen=set();last=start
    def emit(event,**data):
        row=dict(event=event,seconds=time.monotonic()-start,**data);print(json.dumps(row),flush=True)
        with (a.output/'events.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
    emit('start',radius=R,finite_axes=len(axes),relaxation=a.relaxation,all_subset_sizes=True)
    for axis in axes:
        if time.monotonic()-start>a.seconds:break
        q=reflected(p,axis);allareas=polygon_intersection_areas(np.r_[p,q]);cross=allareas[cross_indices];groups=components(cross,1e-6+a.relaxation);counts['axes']+=1
        if len(groups)==1:continue
        counts['disconnected']+=1
        # Complementary bit patterns differ only by one global reflection.
        for bits in range(1,2**(len(groups)-1)):
            subset=[i for k,g in enumerate(groups[1:])if bits&(1<<k)for i in g];candidate=p.copy();candidate[subset]=q[subset];counts['subsets']+=1;sig=signature(candidate)
            if any(np.max(abs(sig-old))<1e-5 for old in known):counts['known']+=1;continue
            tag=tuple(np.rint(sig*1e5).astype(int))
            if tag in seen:continue
            seen.add(tag);counts['geos_checks']+=1;areas=polygon_areas(candidate);radius=compute_mec_fast(candidate)[2];record=dict(axis=axis,subset=subset,components=groups,max_area=float(areas.max()),radius=radius)
            if areas.max()>1e-6:
                counts['repairs']+=1;result=polish(candidate,maxiter=300);candidate=result.x[:-1].reshape(n,3);areas=polygon_areas(candidate);radius=compute_mec_fast(candidate)[2];record.update(repaired_radius=radius,repaired_max_area=float(areas.max()))
            if areas.max()>1e-6+2e-12 or radius>R+5e-8:record['status']='repair_failed'
            elif any(np.max(abs(signature(candidate)-old))<1e-5 for old in known):record['status']='repaired_known'
            else:
                counts['new']+=1;path=a.output/f"seed-{counts['new']:03d}.json";write(path,candidate);known.append(signature(candidate));record.update(status='new_continuous_seed',path=str(path))
            records.append(record);emit('candidate',**record)
        if time.monotonic()-last>10:emit('progress',counts=counts.copy());last=time.monotonic()
    result=dict(radius=R,counts=counts,complete=counts['axes']==len(axes),finite_axes=len(axes),records=records);(a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');emit('done',**{k:v for k,v in result.items()if k!='records'})


if __name__=='__main__':main()
