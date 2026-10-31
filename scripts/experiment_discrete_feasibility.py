"""Six-decimal feasibility search for a fixed target radius, with contact cuts."""
from __future__ import annotations
import os
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp
from scripts.rounding_search import area_model,exact_valid,write_solution
from scripts.lattice_polish import containment_model
from semicircle_packing.polish import load
from semicircle_packing.geometry import Semicircle
from semicircle_packing.scoring import compute_mec


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='experiments/original.json');p.add_argument('--target',type=float,default=2.94852685);p.add_argument('--steps',type=int,default=30);p.add_argument('--rounds',type=int,default=100);p.add_argument('--seconds',type=float,default=2);p.add_argument('--seed',type=int,default=15);p.add_argument('--output',default='experiments/discrete_feasible.json');a=p.parse_args()
    base=np.round(load(a.input),6);rng=np.random.default_rng(a.seed)
    cx,cy,radius=compute_mec([Semicircle(*q) for q in base]);center=np.array([cx,cy]);gaps,contain=containment_model(base,center,radius)
    pairs,areas,jac=area_model(base); cuts=[(base.copy(),areas,jac)];started=time.monotonic()
    for run in range(a.rounds):
        rows=[];limits=[]
        for anchor,aa,jj in cuts:
            rows.append(jj);limits.append(1e-6-2e-12-aa-jj@(base-anchor).ravel())
        ar=np.zeros((sum(len(q)for q in rows),48));ar[:,:45]=np.vstack(rows)*1e-6/1e-8
        mat=np.vstack([ar,contain]);lo=np.r_[np.full(len(ar),-np.inf),-gaps/1e-6];hi=np.r_[np.concatenate(limits)/1e-8,np.full(len(gaps),np.inf)]
        bound=Bounds(np.r_[np.full(45,-a.steps),-3*a.steps,-3*a.steps,(a.target-radius)/1e-6],np.r_[np.full(45,a.steps),3*a.steps,3*a.steps,(a.target-radius)/1e-6]);bound.lb[:2]=0;bound.ub[:2]=0
        # A tiny random objective asks HiGHS for different feasible lattice points.
        obj=np.zeros(48)
        if run%5==4:obj[:45]=rng.normal(0,1,45)*1e-4
        result=milp(obj,integrality=np.r_[np.ones(45),np.zeros(3)],bounds=bound,constraints=LinearConstraint(mat,lo,hi),options=dict(time_limit=a.seconds,mip_rel_gap=0.0))
        if result.x is None:
            print('no candidate',run,result.message,'cuts',len(cuts),'elapsed',time.monotonic()-started,flush=True);continue
        candidate=np.round(base+np.rint(result.x[:45]).reshape(15,3)*1e-6,6)
        pp,aa,jj=area_model(candidate);max_area=float(max(aa));score=compute_mec([Semicircle(*q)for q in candidate])[2]
        print('candidate',run,'score',score,'maxarea',max_area,'cuts',len(cuts),'elapsed',time.monotonic()-started,flush=True)
        write_solution(Path(a.output).with_name('discrete_last.json'),candidate)
        if max_area<=1e-6 and score<a.target+2e-10:
            write_solution(Path(a.output),candidate);print('WIN',score,flush=True);break
        # Convex cuts at near-parallel flat contacts; include any violated contact.
        active=aa>1e-6-2e-12
        if np.any(active):cuts.append((candidate.copy(),aa[active],jj[active]))
        else:
            # Tighten the enclosing-circle linearization when the area is feasible.
            cx,cy,radius=compute_mec([Semicircle(*q)for q in candidate]); base=candidate;center=np.array([cx,cy]);gaps,contain=containment_model(base,center,radius)
            pairs,areas,jac=area_model(base);cuts=[(base.copy(),areas,jac)]
if __name__=='__main__':main()
