"""Directed contact opening at a bounded radius, followed by recompression.

Unlike removal/regrowth, all15 pieces remain full sized. An exact separating
gap is maximized for one initially active contact while the polygon overlap
constraints stay enforced. The resulting boundary state is then compressed.
Continuous seeds are explicitly distinct from rounded, API-valid submissions.
"""
from __future__ import annotations
import os
for key in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse,json,time,itertools
from pathlib import Path
import numpy as np
import networkx as nx
from scipy.optimize import minimize
from semicircle_packing.polish import AreaConstraints,load,polish,polygon_areas,checked,save
from semicircle_packing.fast_geometry import FastGeometry
from semicircle_packing.fast_mec import compute_mec_fast
from experiment_basins import congruent_packings
from experiment_surgery import write_full
from arena import check_solution


def contact_graph(p,radius):
    c=AreaConstraints(p);g=c.fun(np.r_[p.ravel(),radius]);graph=nx.Graph()
    for i in range(15):graph.add_node(i,boundary=int(sum(g[105+3*i:108+3*i]<1e-7)))
    for k,(i,j)in enumerate(itertools.combinations(range(15),2)):
        if g[k]<1e-7:graph.add_edge(i,j)
    return graph


def open_contact(p,cap,row,maxiter):
    constraints=AreaConstraints(p);geometry=FastGeometry()
    def objective(v):
        z=np.r_[v,cap]
        if row<105:g,j=geometry.evaluate(z);k=row
        else:g,j=geometry.boundary(z);k=row-105
        return -g[k],-j[k,:45]
    bounds=[(None,None)]*45;bounds[2]=(p[0,2],p[0,2]) # Remove rigid rotation only.
    result=minimize(objective,p.ravel(),jac=True,method='SLSQP',bounds=bounds,
        constraints={'type':'ineq','fun':lambda v:constraints.fun(np.r_[v,cap]),
                     'jac':lambda v:constraints.jac(np.r_[v,cap])[:,:45]},
        options=dict(ftol=2e-12,maxiter=maxiter))
    return result.x.reshape(15,3),dict(opened_gap=float(-result.fun),iterations=result.nit,
        success=bool(result.success),violation=float(max(0,-constraints.fun(np.r_[result.x,cap]).min())))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs',nargs='+',default=['experiments/branch-3.141616-3.141592.json','experiments/2026-11-04/round2/topology/continuous-best.json'])
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seconds',type=float,default=180)
    ap.add_argument('--seed',type=int,default=110483);ap.add_argument('--barriers',type=float,nargs='+',default=[.0001,.001,.01,.05])
    ap.add_argument('--maxiter',type=int,default=120);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();rng=np.random.default_rng(a.seed);bases=[];graphs=[];archive=[];best=checked(load('solution.json')).score
    def emit(event,**data):
        row=dict(event=event,seconds=time.monotonic()-started,**data);line=json.dumps(row);print(line,flush=True)
        with(a.output/'events.jsonl').open('a')as f:f.write(line+'\n')
    schedule=[]
    for index,path in enumerate(a.inputs):
        p=load(path);cx,cy,r=compute_mec_fast(p);p[:,:2]-=[cx,cy];c=AreaConstraints(p);active=np.flatnonzero(c.fun(np.r_[p.ravel(),r])<1e-7)
        bases.append((p,r));graphs.append(contact_graph(p,r));archive.append(p);write_full(a.output/f'input-{index}.json',p)
        for barrier in a.barriers:
            order=active.copy();rng.shuffle(order);schedule.extend((index,int(row),barrier)for row in order)
        emit('baseline',input=path,radius=r,active=len(active))
    # Interleave geometries and barrier scales rather than spend the whole run
    # on the first seed and smallest permitted release.
    rng.shuffle(schedule);trials=novel_graphs=novel_geometry=competitive=0;best_continuous=min(r for _,r in bases)
    for index,row,barrier in schedule:
        if time.monotonic()-started>a.seconds:break
        base,radius=bases[index];opened,opening=open_contact(base,radius+barrier,row,a.maxiter)
        if opening['violation']>1e-7:
            # A failed nonlinear solve is not a path through the permitted
            # radius barrier. Preserve the failure instead of calling its
            # unconstrained jump a successful structural move.
            emit('opening_rejected',trial=trials,input=index,row=row,barrier=barrier,opening=opening)
            trials+=1;continue
        solved=polish(opened,maxiter=180);q=solved.x[:-1].reshape(15,3);cx,cy,r=compute_mec_fast(q);q[:,:2]-=[cx,cy]
        c=AreaConstraints(q);violation=float(max(0,-c.fun(np.r_[q.ravel(),r]).min()));record=dict(trial=trials,input=index,row=row,barrier=barrier,opening=opening,radius=r,violation=violation,polish_iterations=solved.nit)
        if violation<2e-8:
            G=contact_graph(q,r);new_graph=all(not nx.is_isomorphic(G,old,node_match=nx.algorithms.isomorphism.categorical_node_match('boundary',0))for old in graphs)
            new_geometry=all(not congruent_packings(q,old,tolerance=1e-5)for old in archive)
            record.update(new_graph=new_graph,new_geometry=new_geometry)
            if new_graph:graphs.append(G);novel_graphs+=1
            if new_geometry:archive.append(q.copy());novel_geometry+=1
            if new_graph or(new_geometry and r<radius+2e-7):
                path=a.output/f'seed-{trials:04d}.json';write_full(path,q);record['path']=str(path)
            if r<radius+2e-7:
                exact=polygon_areas(q);record['exact_polygon_maximum_area']=float(exact.max())
                if exact.max()<=1e-6+2e-12:
                    competitive+=1;best_continuous=min(best_continuous,r)
                    v=checked(q);record.update(rounded_valid=v.valid,rounded_score=v.score)
                    if v.valid and v.score<best:
                        path=a.output/f'candidate-{trials:04d}.json';save(path,q);response=check_solution(json.loads(path.read_text()));path.with_suffix('.api.json').write_text(json.dumps(response,indent=2)+'\n');record['api']=response
                        if response.get('valid')and response['score']<best:best=response['score']
        emit('trial',**record);trials+=1
        if best<2.9485269238856113:break
    result=dict(trials=trials,scheduled=len(schedule),novel_graphs=novel_graphs,novel_geometry=novel_geometry,
                competitive_continuous=competitive,best_continuous=best_continuous,best_valid=best,goal_met=best<2.9485269238856113)
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');emit('done',**result)


if __name__=='__main__':main()
