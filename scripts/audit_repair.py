"""Bounded exact-contact CSP repair of the new Douglas--Rachford phases.

The center candidates include MEC centers, nearby scans, and intersections of
container-center disks induced by pairs of the current support features.
Native circle intersections generate candidates; exact GEOS pair checks learn
additional incompatibilities before the reference scorer can promote anything.
"""
from __future__ import annotations

import os
for name in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

from arena import check_solution
from experiment_contact_csp import compatible, find_assignment, pair_kernel
from semicircle_packing.exact_area import intersection_areas
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.geometry import Semicircle, canonicalize_theta, farthest_boundary_point_from, semicircle_polygon
from semicircle_packing.polish import checked, load, polygon_areas, save


def center_options(base, choices, radius, steps, rng, maximum):
    cx,cy,actual_radius=compute_mec_fast(base);center=np.array([cx,cy])
    supports=[]
    for i,q in enumerate(base):
        far=np.array(farthest_boundary_point_from(Semicircle(*q),cx,cy))
        if actual_radius-np.linalg.norm(far-center)>2e-9:continue
        radial=np.dot(q[:2]-center,[np.cos(q[2]),np.sin(q[2])])>=0
        if radial:points=np.unique(choices[i][:,:2],axis=0);intrinsic=1.
        else:
            tangent=np.c_[-np.sin(choices[i][:,2]),np.cos(choices[i][:,2])]
            sign=1 if np.dot(far-q[:2],[-np.sin(q[2]),np.cos(q[2])])>=0 else -1
            points=np.unique(choices[i][:,:2]+sign*tangent,axis=0);intrinsic=0.
        supports.append((i,points,radius-intrinsic))
    queue=[center]
    queue += [center+np.array([x,y])*1e-6 for x,y in itertools.product((-.5,-.1,0,.1,.5),repeat=2)]
    intersections=[]
    for (_,aa,ra),(_,bb,rb) in itertools.combinations(supports,2):
        delta=bb[None,:,:]-aa[:,None,:];distance=np.linalg.norm(delta,axis=2)
        direction=delta/np.maximum(distance[:,:,None],1e-20)
        along=(ra*ra-rb*rb+distance*distance)/(2*np.maximum(distance,1e-20))
        h2=ra*ra-along*along;middle=aa[:,None,:]+along[:,:,None]*direction
        perpendicular=np.stack([-direction[:,:,1],direction[:,:,0]],axis=2)
        for sign in (-1,1):
            points=middle+sign*np.sqrt(np.maximum(h2,0))[:,:,None]*perpendicular
            mask=(h2>=0)&(np.linalg.norm(points-center,axis=2)<(steps*4+2)*1e-6)
            intersections.extend(points[mask])
    if intersections:
        array=np.array(intersections);distance=np.linalg.norm(array-center,axis=1)
        nearest=np.argsort(distance)[:maximum//2]
        sampled=rng.choice(len(array),min(maximum//2,len(array)),replace=False)
        queue.extend(array[np.r_[nearest,sampled]])
    seen=set();result=[]
    for q in queue:
        key=tuple(np.rint(q/1e-11).astype(np.int64))
        if key not in seen:seen.add(key);result.append(q)
    return result


def containment_domains(choices,center,radius):
    result=[]
    for p in choices:
        x,y=(p[:,:2]-center).T;t=p[:,2];sn,co=np.sin(t),np.cos(t)
        extent=np.maximum(np.hypot(x+sn,y-co),np.hypot(x-sn,y+co))
        radial=np.where(x*co+y*sn>=0,np.hypot(x,y)+1,0)
        valid=np.maximum(extent,radial)<=radius+2e-12
        result.append(int.from_bytes(np.packbits(valid,bitorder="little").tobytes(),"little"))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,default=Path("experiments/2026-11-04/difference-map"))
    parser.add_argument("--output",type=Path,default=Path("experiments/2026-11-04/repair"))
    parser.add_argument("--seconds",type=float,default=180)
    parser.add_argument("--seed",type=int,default=110414)
    parser.add_argument("--centers",type=int,default=96)
    parser.add_argument("--nodes",type=int,default=10000)
    parser.add_argument("--steps",type=int,nargs="+",default=[1,2,3])
    parser.add_argument("--targets",type=float,nargs="+",default=[2.94852694,2.9485269238])
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed);started=time.monotonic();deadline=started+args.seconds
    events=args.output/"events.jsonl";models=[];best=2.948526966725367;kernel=pair_kernel()
    paths=sorted(args.directory.glob("*_nearest.json"))
    snapshots=[]
    for path in paths:
        raw=path.read_bytes();data=json.loads(raw);p=np.array([[v[k]for k in("x","y","theta")]for v in data]);p=np.round(p,6);p[:,2]=[canonicalize_theta(t) for t in p[:,2]]
        copy=args.output/(path.stem+"-input.json");save(copy,p)
        snapshots.append((path,p,hashlib.sha256(raw).hexdigest()))
    def emit(kind,**values):
        row=dict(event=kind,seconds=time.monotonic()-started,**values)
        with events.open("a") as out:out.write(json.dumps(row)+"\n")
        print(json.dumps(row),flush=True)
    for steps in args.steps:
        for path,base,digest in snapshots:
            if time.monotonic()>deadline:break
            offsets=np.array(list(itertools.product(range(-steps,steps+1),repeat=3)))*1e-6
            choices=[]
            for q in base:
                domain=np.round(q+offsets,6)
                domain=domain[np.abs(domain[:,2])<=np.pi]
                choices.append(np.ascontiguousarray(domain))
            allpairs=list(itertools.combinations(range(15),2));areas=intersection_areas(base);poly=polygon_areas(base)
            ids=[k for k,(i,j) in enumerate(allpairs) if np.linalg.norm(base[i,:2]-base[j,:2])<=2+4*steps*1e-6]
            pairs=[allpairs[k] for k in ids]
            limits=1e-6+areas[ids]-poly[ids]+2e-11
            adjacency,maps=compatible(choices,pairs,limits,kernel,base=None)
            cache={};counts=dict(centers=0,empty=0,no_assignment=0,node_limited=0,candidates=0,learned=0,valid=0)
            emit("model",input=str(path),sha256=digest,steps=steps,choices=[len(q)for q in choices],pairs=len(pairs))
            for target in args.targets:
                centers=center_options(base,choices,target,steps,rng,args.centers)
                for center in centers:
                    if time.monotonic()>deadline:break
                    counts["centers"]+=1;initial=containment_domains(choices,center,target)
                    if not all(initial):counts["empty"]+=1;continue
                    for attempt in range(4):
                        answer,nodes=find_assignment(choices,adjacency,maps,rng,args.nodes,initial)
                        if answer is None:
                            counts["no_assignment"]+=1
                            counts["node_limited"]+=int(nodes>args.nodes)
                            break
                        counts["candidates"]+=1;candidate=np.array([choices[i][k] for i,k in enumerate(answer)])
                        invalid=[]
                        for i,j in pairs:
                            for item in (i,j):
                                key=(item,answer[item])
                                if key not in cache:cache[key]=semicircle_polygon(Semicircle(*candidate[item]))
                            if cache[i,answer[i]].intersection(cache[j,answer[j]]).area>1e-6:invalid.append((i,j))
                        if invalid:
                            for i,j in invalid:
                                maps[i,j][answer[i]]&=~(1<<answer[j]);maps[j,i][answer[j]]&=~(1<<answer[i])
                            counts["learned"]+=len(invalid);continue
                        validation=checked(candidate)
                        if validation.valid:
                            counts["valid"]+=1
                            if validation.score<best-1e-12:
                                output=args.output/"checked-candidate.json";save(output,candidate)
                                response=check_solution(json.loads(output.read_text()))
                                (args.output/"api-best.json").write_text(json.dumps(response,indent=2)+"\n")
                                if response.get("valid") and response["score"]<best-1e-12:
                                    best=response["score"];save(args.output/"best.json",candidate)
                                    emit("improvement",input=str(path),steps=steps,target=target,score=best,api=response)
                                else:emit("server_rejection",input=str(path),steps=steps,target=target,api=response)
                        else:emit("reference_rejection",input=str(path),steps=steps,errors=validation.errors)
                        break
                if time.monotonic()>deadline:break
            row=dict(input=str(path),steps=steps,**counts);models.append(row);emit("model_done",**row)
        if time.monotonic()>deadline:break
    summary=dict(seed=args.seed,targets=args.targets,models=models,best=best,seconds=time.monotonic()-started)
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");emit("done",best=best,models=len(models))


if __name__=="__main__":main()
