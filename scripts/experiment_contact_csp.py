"""Exact nonlinear contact CSP over nearby six-decimal placements."""
from __future__ import annotations
import os
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,ctypes,hashlib,itertools,json,subprocess,tempfile,time
from pathlib import Path
import numpy as np
from semicircle_packing.polish import load,save,polygon_areas,checked
from semicircle_packing.exact_area import intersection_areas,intersection_areas_and_jacobian


def pair_kernel():
    src=Path(__file__).with_name('_discrete_compatibility.cpp'); other=src.parent.parent/'src/semicircle_packing/area_kernel.cpp';digest=hashlib.sha256(src.read_bytes()+other.read_bytes()).hexdigest()[:16]
    output=Path(tempfile.gettempdir())/('packing-csp-'+digest+'.so')
    if not output.exists():subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(src),'-o',str(output)],check=True)
    lib=ctypes.CDLL(str(output));f=lib.pair_area_matrix;array=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');f.argtypes=[ctypes.c_int,ctypes.c_int,array,array,array];return f


def domains(p,radius,steps,center=(0.,0.)):
    offsets=np.array(list(itertools.product(range(-steps,steps+1),repeat=3)))*1e-6
    choices=[]
    for q in p:
        d=np.round(q,6)[None,:]+offsets
        x,y,t=d.T;x=x-center[0];y=y-center[1];sn=np.sin(t);co=np.cos(t)
        rad=np.maximum(np.hypot(x+sn,y-co),np.hypot(x-sn,y+co))
        radial=np.where(x*co+y*sn>=0,np.hypot(x,y)+1,0)
        valid=(np.maximum(rad,radial)<=radius)&(np.abs(d[:,2])<=np.pi)
        choices.append(np.ascontiguousarray(np.round(d[valid],6)))
    return choices


def flat_areas(a,b):
    na=np.c_[np.cos(a[:,2]),np.sin(a[:,2])];ta=np.c_[-na[:,1],na[:,0]]
    nb=np.c_[np.cos(b[:,2]),np.sin(b[:,2])];tb=np.c_[-nb[:,1],nb[:,0]]
    den=na@nb.T;slope=-(ta@nb.T)/den
    intercept=-(a[:,:2]@nb.T-np.sum(b[:,:2]*nb,axis=1)[None,:])/den
    offset=ta@b[:,:2].T-np.sum(ta*a[:,:2],axis=1)[:,None]
    width=np.abs(ta@tb.T);lo=np.maximum(-1.,offset-width);hi=np.minimum(1.,offset+width)
    hl=intercept+slope*lo;hu=intercept+slope*hi
    positive=(hl>=0)&(hu>=0);crossing=(hl>0)^(hu>0)
    out=np.where(positive,.5*(hl+hu)*(hi-lo),0.)
    np.divide(.5*np.maximum(hl,hu)**2,np.abs(slope),out=out,where=crossing)
    return np.where(hi>lo,out,0.)


def compatible(choices,pairs,limits,kernel,base=None,flat_angles=None):
    if base is not None:
        aa,jj=intersection_areas_and_jacobian(base)
        lookup={(i,j):k for k,(i,j)in enumerate(itertools.combinations(range(15),2))}
    adjacency=[[]for _ in choices];maps={}
    for (i,j),limit in zip(pairs,limits):
        a,b=choices[i],choices[j]
        if base is not None and np.linalg.norm(jj[lookup[i,j]])<.1:
            k=lookup[i,j];da=(a-base[i])@jj[k,3*i:3*i+3];db=(b-base[j])@jj[k,3*j:3*j+3]
            out=aa[k]+da[:,None]+db[None,:]
            mask=out<=limit+1e-10
        elif base is not None:
            out=flat_areas(a,b);mask=out<=limit-2e-12
        else:
            out=np.empty((len(a),len(b)));kernel(len(a),len(b),a,b,out);mask=out<=limit
        if flat_angles is not None and(i,j)in flat_angles:
            mask&=np.abs(a[:,2,None]-b[None,:,2]-flat_angles[i,j])<1e-9
        forward=[];backward=[]
        for row in mask:forward.append(int.from_bytes(np.packbits(row,bitorder='little').tobytes(),'little'))
        for row in mask.T:backward.append(int.from_bytes(np.packbits(row,bitorder='little').tobytes(),'little'))
        maps[i,j]=forward;maps[j,i]=backward;adjacency[i].append(j);adjacency[j].append(i)
    return adjacency,maps


def find_assignment(choices,adj,maps,rng,limit,initial=None):
    domains=[(1<<len(p))-1 for p in choices] if initial is None else list(initial);nodes=0
    def propagate(dd,queue):
        while queue:
            i,j=queue.pop();domain=dd[i];other=dd[j];revised=0;mm=maps[i,j]
            while domain:
                bit=domain&-domain; k=bit.bit_length()-1; domain-=bit
                if mm[k]&other:revised|=bit
            if revised!=dd[i]:
                if not revised:return False
                dd[i]=revised
                queue.extend((k,i)for k in adj[i]if k!=j)
        return True
    def visit(dd):
        nonlocal nodes
        nodes+=1
        if nodes>limit:return None
        pending=[i for i,d in enumerate(dd)if d.bit_count()>1]
        if not pending:return [d.bit_length()-1 for d in dd]
        i=min(pending,key=lambda i:(dd[i].bit_count(),-len(adj[i])))
        values=[];rest=dd[i]
        while rest:
            bit=rest&-rest;rest-=bit;values.append(bit)
        rng.shuffle(values)
        for bit in values:
            child=dd.copy();child[i]=bit
            if propagate(child,[(j,i)for j in adj[i]]):
                answer=visit(child)
                if answer is not None:return answer
        return None
    if not all(domains):return None,nodes
    if not propagate(domains,list(maps)):return None,nodes
    return visit(domains),nodes


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--input',default='experiments/continuous.json');pa.add_argument('--output',default='experiments/discrete_csp.json');pa.add_argument('--seconds',type=float,default=120);pa.add_argument('--seed',type=int,default=34);pa.add_argument('--steps',type=int,default=2);pa.add_argument('--target',type=float,default=2.94852692);pa.add_argument('--nodes',type=int,default=10000);pa.add_argument('--exact-contacts',action='store_true');pa.add_argument('--lock-flat',action='store_true');a=pa.parse_args();rng=np.random.default_rng(a.seed)
    base=load(a.input);areas=intersection_areas(base);polygons=polygon_areas(base);allpairs=list(itertools.combinations(range(15),2));ids=np.flatnonzero(areas>1e-8);pairs=[allpairs[k]for k in ids];limits=1e-6+areas[ids]-polygons[ids]-1e-11
    kernel=pair_kernel();start=time.monotonic();run=0
    while time.monotonic()-start<a.seconds:
        angle=rng.uniform(-np.pi,np.pi);c,s=np.cos(angle),np.sin(angle);p=base.copy();p[:,0]=c*base[:,0]-s*base[:,1];p[:,1]=s*base[:,0]+c*base[:,1];p[:,2]=(p[:,2]+angle+np.pi)%(2*np.pi)-np.pi
        shift=rng.uniform(-.5e-6,.5e-6,2);p[:,:2]+=shift
        # Frame translation is also the enclosing center, so subtract it for containment.
        choices=domains(p,a.target,a.steps,shift)
        if not all(len(v)for v in choices):run+=1;continue
        flat_angles={(i,j):np.round(p[i,2]-p[j,2],6)for i,j in [(1,5),(6,10)]} if a.lock_flat else None
        adjacency,maps=compatible(choices,pairs,limits,kernel,None if a.exact_contacts else p,flat_angles)
        result,nodes=find_assignment(choices,adjacency,maps,rng,a.nodes)
        if result is not None:
            candidate=np.array([choices[i][k]for i,k in enumerate(result)]);val=checked(candidate)
            print('CANDIDATE',run,val.valid,val.score,val.errors,'nodes',nodes,'elapsed',time.monotonic()-start,flush=True)
            if val.valid:
                save(a.output,candidate)
                if val.score<2.9485269238856113:print('WIN',val.score,flush=True);break
        if run%100==0:print('frame',run,'domains',[len(v)for v in choices],'nodes',nodes,'elapsed',time.monotonic()-start,flush=True)
        run+=1
    print('done',run,'elapsed',time.monotonic()-start,flush=True)
if __name__=='__main__':main()
