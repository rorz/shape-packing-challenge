"""Measure actual exact-contact sections across nearby integer flat branches."""
import argparse,csv,itertools,json,time
from pathlib import Path
import numpy as np
from experiment_mirror_family import Manifold,cells,full,wrap
from semicircle_packing.polish import load
from experiment_tied_batch import tied_transform

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seconds',type=float,default=120);ap.add_argument('--output',default='experiments/2026-11-04/round3/mirror/branch-map');a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);model=Manifold(load('experiments/2026-11-04/round2/topology/continuous-best.json'));baseline=float(model.z[-1]);start=time.monotonic();rows=[];_,T=tied_transform(model.p,[(1,14),(4,8)])
 for first_sign,second_sign,k,l in itertools.product([1,-1],[1,-1],range(-8,9),range(-4,5)):
  if time.monotonic()-start>a.seconds:break
  first=(3.141570 if first_sign>0 else-3.141616)+k*1e-6;second=(3.141593 if second_sign>0 else-3.141592)+l*1e-6;p,m=model.section((first,second),(0,0));valid=m['violation']<2e-11 and m['residual']<2e-11 and m['min_stress']>0;_,J=model.c.evaluate(np.r_[p.ravel(),m['radius']]);U,S,_=np.linalg.svd(J[model.active,:45]@T,full_matrices=True);w=U[:,-1];w*=np.sign(w.sum());tied_valid=m['violation']<2e-11 and m['residual']<2e-11 and w.min()>0;row=dict(first_raw=first,second_raw=second,first_sign=first_sign,second_sign=second_sign,first_offset_k=k,second_offset_l=l,radius=m['radius'],cost_nm=(m['radius']-baseline)*1e9,violation=m['violation'],residual=m['residual'],minimum_stress=m['min_stress'],iterations=m['iterations'],feasible_positive_stress=valid,path=None,wrap_cell_width=None,tied_stress_minimum=float(w.min()),tied_stress_singular_value=float(S[-1]),feasible_positive_tied_stress=bool(tied_valid))
  if tied_valid:
   intervals=cells(p,[first_sign,second_sign]);lo,hi=max(intervals,key=lambda q:q[1]-q[0]);phi=(lo+hi)*.5;cs,sn=np.cos(phi),np.sin(phi);q=p.copy();q[:,:2]=q[:,:2]@np.array([[cs,sn],[-sn,cs]]);q[:,2]=wrap(q[:,2]+phi);path=out/f'branch-{round(first*1e6):+d}-{round(second*1e6):+d}.json';full(path,q);row.update(path=str(path),wrap_cell_width=sum(hi-lo for lo,hi in intervals))
  rows.append(row)
 (out/'branches.json').write_text(json.dumps(rows,indent=2)+'\n')
 with(out/'branches.csv').open('w',newline='')as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 good=[r for r in rows if r['feasible_positive_stress']];summary=dict(baseline_radius=baseline,expected_sections=612,tested_sections=len(rows),feasible_positive_stress=len(good),cost_counts_nm={str(t):sum(r['cost_nm']<t for r in good)for t in [5,20,50]},minimum_radius=min(r['radius']for r in good),maximum_cost_nm=max(r['cost_nm']for r in good),minimum_stress=min(r['minimum_stress']for r in good),max_violation=max(r['violation']for r in good),seconds=time.monotonic()-start,by_sign_cell={f'{s1:+d},{s2:+d}':dict(feasible=sum(r['first_sign']==s1 and r['second_sign']==s2 for r in good),**{f'below_{t}nm':sum(r['first_sign']==s1 and r['second_sign']==s2 and r['cost_nm']<t for r in good)for t in [5,20,50]})for s1,s2 in itertools.product([1,-1],repeat=2)})
 tied=[r for r in rows if r['feasible_positive_tied_stress']];summary.update(feasible_positive_tied_stress=len(tied),cost_counts_tied_nm={str(t):sum(r['cost_nm']<t for r in tied)for t in [5,20,50]},minimum_tied_stress=min(r['tied_stress_minimum']for r in tied))
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
