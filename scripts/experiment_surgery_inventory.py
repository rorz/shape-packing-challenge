"""Finite connected-patch reflection inventory with GEOS acceptance."""
from __future__ import annotations
import os
for key in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse,itertools,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from semicircle_packing.polish import AreaConstraints,polygon_areas
from semicircle_packing.exact_area import polygon_intersection_areas
from semicircle_packing.geometry import Semicircle,semicircle_polygon
from semicircle_packing.fast_mec import compute_mec_fast
ROOT=Path('experiments/2026-11-04/round2/topology')
PAIRS=list(itertools.combinations(range(15),2))
def write(path,p):path.write_text(json.dumps([dict(zip(('x','y','theta'),map(float,row)))for row in p],indent=2)+'\n')
def signature(p):
 c=p[:,:2]+4/(3*np.pi)*np.c_[np.cos(p[:,2]),np.sin(p[:,2])];return np.r_[np.sort(pdist(p[:,:2])),np.sort(pdist(c))]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seconds',type=float,default=60);ap.add_argument('--output',default=str(ROOT/'inventory'));a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);started=time.monotonic();base=np.load('experiments/soft_modes.npz')['p'].copy();cx,cy,R=compute_mec_fast(base);base[:,:2]-=[cx,cy];Rlimit=R+1e-9
 def repair(p):
  constraint=AreaConstraints(p,allowance=1e-6-2e-13,polygon_correction=True,fast_polygon=False)
  def fun(u):return constraint.fun(np.r_[(p+u.reshape(15,3)*1e-6).ravel(),Rlimit])*1e6
  def jac(u):return constraint.jac(np.r_[(p+u.reshape(15,3)*1e-6).ravel(),Rlimit])[:,:45]
  sol=minimize(lambda u:.5*(u@u),np.zeros(45),jac=lambda u:u,method='SLSQP',bounds=[(-.02,.02)]*45,constraints=dict(type='ineq',fun=fun,jac=jac),options=dict(maxiter=50,ftol=1e-13));q=p+sol.x.reshape(15,3)*1e-6;areas=polygon_areas(q);radius=compute_mec_fast(q)[2];return q,dict(success=bool(areas.max()<=1e-6 and radius<=Rlimit+2e-12),max_area=float(areas.max()),radius=radius,max_displacement_nm=float(np.abs(sol.x).max()*1000),iterations=int(sol.nit))
 baseline_areas=polygon_areas(base);baseline_repair=None
 if baseline_areas.max()>1e-6:base,baseline_repair=repair(base)
 baseline_areas=polygon_areas(base);write(out/'baseline.json',base)
 if baseline_areas.max()>1e-6:raise ValueError(f'Baseline GEOS invalid after tiny repair: {baseline_areas.max()}')
 constraint=AreaConstraints(base);g,j=constraint.evaluate(np.r_[base.ravel(),Rlimit]);neighbors=[set()for _ in range(15)]
 for row,(i,k)in enumerate(PAIRS):
  if g[row]<1e-7:neighbors[i].add(k);neighbors[k].add(i)
 subsets=[];by_size={}
 for count in range(1,5):
  for subset in itertools.combinations(range(15),count):
   seen={subset[0]};pending=[subset[0]];allowed=set(subset)
   while pending:
    node=pending.pop();new=(neighbors[node]&allowed)-seen;seen|=new;pending.extend(new)
   if len(seen)==count:subsets.append(subset)
  by_size[count]=sum(len(s)==count for s in subsets)
 norms=np.linalg.norm(base[:,:2],axis=1);boundary=g[105:].reshape(15,3);rim=[i for i in range(15)if boundary[i,2]<1e-7 and base[i,:2]@np.array([np.cos(base[i,2]),np.sin(base[i,2])])>=0];axes=[]
 def add_axis(angle,label):
  angle=float(angle%np.pi)
  if not any(abs(np.arctan2(np.sin(2*(angle-old[0])),np.cos(2*(angle-old[0]))))<1e-10 for old in axes):axes.append((angle,label))
 for i in rim:add_axis(np.arctan2(base[i,1],base[i,0]),dict(kind='radial',pieces=[i]))
 for i,k in itertools.combinations(rim,2):
  if abs(norms[i]-norms[k])<1e-6:add_axis((np.arctan2(base[i,1],base[i,0])+np.arctan2(base[k,1],base[k,0]))*.5,dict(kind='rim_pair',pieces=[i,k]))
 control_angle=float((np.arctan2(base[3,1],base[3,0])+np.arctan2(base[4,1],base[4,0]))*.5%np.pi);add_axis(control_angle,dict(kind='positive_control',pieces=[3,4]));base_signature=signature(base);accepted_signatures=[];records=[];tested=screened=geos_calls=feasible=congruent=0;control=None;baseline_polys=[semicircle_polygon(Semicircle(*v))for v in base]
 def evaluate(subset,angle,label,is_control=False):
  nonlocal tested,screened,geos_calls,feasible,congruent,control
  tested+=1;normal=np.array([-np.sin(angle),np.cos(angle)]);q=base.copy();indices=np.array(subset);q[indices,:2]-=2*np.outer(q[indices,:2]@normal,normal);q[indices,2]=2*angle-q[indices,2];cross=[row for row,(i,k)in enumerate(PAIRS)if(i in subset)!=(k in subset)];native=polygon_intersection_areas(q)
  row=dict(subset=list(subset),axis=angle,axis_source=label,positive_control=is_control,native_cross_excess=float(native[cross].max()-1e-6))
  if native[cross].max()>1e-6+3e-11:
   screened+=1;row['status']='native_rejected'
   if is_control:control=row
   return
  polys=baseline_polys.copy()
  for i in subset:polys[i]=semicircle_polygon(Semicircle(*q[i]))
  actual=np.array([polys[i].intersection(polys[k]).area for i,k in(PAIRS[r]for r in cross)]);geos_calls+=1;row['geos_cross_excess']=float(actual.max()-1e-6);repair_info=None
  if actual.max()>1e-6:
   q,repair_info=repair(q);row['repair']=repair_info
   if not repair_info['success']:
    row['status']='geos_repair_failed'
    if is_control:control=row
    records.append(row);return
  # Full GEOS check is authoritative, including internal pairs after a numerical repair.
  all_areas=polygon_areas(q);geos_calls+=1;row['max_geos_area']=float(all_areas.max());row['radius']=compute_mec_fast(q)[2]
  if all_areas.max()>1e-6 or row['radius']>Rlimit+2e-12:
   row['status']='full_geos_rejected'
   if is_control:control=row
   records.append(row);return
  feasible+=1;sig=signature(q);difference=float(np.max(np.abs(sig-base_signature)));row['invariant_difference_from_baseline']=difference
  if difference<1e-5:congruent+=1;row['status']='baseline_invariants_match'
  elif any(np.max(np.abs(sig-old))<1e-5 for old in accepted_signatures):row['status']='duplicate_invariants'
  else:
   accepted_signatures.append(sig);path=out/f'seed-{len(accepted_signatures):03d}.json';write(path,q);row.update(status='saved_noncongruent',path=str(path))
  if is_control:control=row
  records.append(row)
 evaluate((1,5,14),control_angle,dict(kind='positive_control',pieces=[3,4]),True)
 for subset in subsets:
  for angle,label in axes:
   if time.monotonic()-started>a.seconds:break
   if subset==(1,5,14)and abs(np.arctan2(np.sin(2*(angle-control_angle)),np.cos(2*(angle-control_angle))))<1e-10:continue
   evaluate(subset,angle,label)
  if time.monotonic()-started>a.seconds:break
 report=dict(complete=time.monotonic()-started<=a.seconds,seconds=time.monotonic()-started,baseline_radius=R,global_radius_limit=Rlimit,baseline_repair=baseline_repair,connected_subsets_by_size=by_size,connected_subsets=len(subsets),rim_pieces=rim,axis_candidates=len(axes),finite_candidates=len(subsets)*len(axes),tested=tested,native_rejected=screened,geos_calls=geos_calls,feasible=feasible,baseline_invariant_matches=congruent,distinct_noncongruent_seeds=len(accepted_signatures),invariant_cluster_tolerance=1e-5,positive_control=control,records=records);(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='records'},indent=2))
if __name__=='__main__':main()
