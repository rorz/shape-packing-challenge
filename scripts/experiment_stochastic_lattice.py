"""Heuristic stochastic nearest-plane proposals across fresh packing frames.

Alternative decisions are sampled using deterministic Babai residuals, then
recomputed after propagation. This is not an exact Klein/lattice distribution.
"""
from __future__ import annotations
import os
for name in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[name]='1'
import argparse,ctypes,json,shlex,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from experiment_batch_lattice import NativeBabai,Frame,EliteArchive
from breakout_geometry import evaluate_batch
from semicircle_packing.polish import checked,load,save,polygon_areas
from semicircle_packing.fast_mec import compute_mec_fast
from arena import check_solution


class NativeStochastic(NativeBabai):
 def __init__(self):
  super().__init__();double=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');integer=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS');flag=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
  self.lib.packing_batch_stochastic.argtypes=[ctypes.c_int,ctypes.c_int,double,double,double,integer,integer,ctypes.c_uint64,ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_double,integer,integer,double,flag,flag,double];self.lib.packing_batch_stochastic.restype=None
  self.lib.packing_batch_unique_centers.argtypes=[ctypes.c_int,integer,ctypes.c_int64,flag,integer];self.lib.packing_batch_unique_centers.restype=None
 def decode_random(self,targets,mu,norms,moves,anchor,*,seed,max_flips=3,temperature=.15,norm_power=.25,fraction=1.):
  targets=np.ascontiguousarray(targets,dtype=float);mu=np.ascontiguousarray(mu,dtype=float);norms=np.ascontiguousarray(norms,dtype=float);moves=np.ascontiguousarray(moves,dtype=np.int64);anchor=np.ascontiguousarray(anchor,dtype=np.int64).reshape(45);count,n=targets.shape
  coeff=np.empty((count,n),dtype=np.int64);units=np.empty((count,45),dtype=np.int64);placements=np.empty((count,15,3));ambiguous=np.empty(count,dtype=np.int32);changes=np.empty(count,dtype=np.int32);distances=np.empty((count,2))
  self.lib.packing_batch_stochastic(count,n,targets,mu,norms,moves,anchor,seed,max_flips,temperature,norm_power,fraction,coeff,units,placements,ambiguous,changes,distances)
  return (placements,units,coeff,ambiguous),changes,distances
 def unique_centers(self,units,limit=1000000):
  """Diversity diagnostic only; angle variants still receive full scoring."""
  units=np.ascontiguousarray(units,dtype=np.int64).reshape(-1,45);keep=np.empty(len(units),dtype=np.int32);info=np.empty(3,dtype=np.int64);self.lib.packing_batch_unique_centers(len(units),units,limit,keep,info);return keep.astype(bool),info


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',default='experiments/branch-3.141616-3.141592.json');ap.add_argument('--output',type=Path,required=True);ap.add_argument('--mode',choices=['baseline','stochastic'],default='stochastic');ap.add_argument('--seconds',type=float,default=60);ap.add_argument('--seed',type=int,default=3110511);ap.add_argument('--decoder-seed',type=int,default=371911);ap.add_argument('--target',type=float,default=2.94852698);ap.add_argument('--goal',type=float,default=2.9485269238);ap.add_argument('--batch',type=int,default=2048);ap.add_argument('--movement',type=float,default=50);ap.add_argument('--concentration',type=float,default=10);ap.add_argument('--boundary-bias',type=float,default=5);ap.add_argument('--regularization',type=float,default=1e-5);ap.add_argument('--flat-weight',type=float,default=.1);ap.add_argument('--block-size',type=int,default=20);ap.add_argument('--max-flips',type=int,default=3);ap.add_argument('--temperature',type=float,default=.15);ap.add_argument('--norm-power',type=float,default=.25);ap.add_argument('--fraction',type=float,default=1.);ap.add_argument('--elite-count',type=int,default=10);ap.add_argument('--elite-margin',type=float,default=1e-7);a=ap.parse_args()
 a.output.mkdir(parents=True,exist_ok=True);start=time.monotonic();events=a.output/'events.jsonl';rng=np.random.default_rng(a.seed)
 (a.output/'steps.md').write_text('# Stochastic nearest-plane campaign\n\nThis is a proposal heuristic. It preserves every physical angle dimension and removes only exact integer translations. The decoder RNG is independent of frame/target RNG, so equal main seeds produce matching input streams. Native polygon screening uses1e-11; GEOS checks retained elites and exact nearest-point diagnostics.\n\n```text\n'+shlex.join(['.venv/bin/python',*sys.argv])+'\n```\n')
 def emit(event,**data):
  row=dict(time=datetime.now(timezone.utc).isoformat(),event=event,seconds=time.monotonic()-start,**data);line=json.dumps(row,allow_nan=False);print(line,flush=True)
  with events.open('a')as f:f.write(line+'\n')
 p=load(a.input);(a.output/'input-snapshot.json').write_bytes(Path(a.input).read_bytes());cx,cy,radius=compute_mec_fast(p);p[:,:2]-=[cx,cy];incumbent=checked(load('solution.json'));best=incumbent.score;decoder=NativeStochastic();decoder.reset();elites=EliteArchive(a.output,a.elite_count,best+a.elite_margin,emit)
 counts=dict(frames=0,bases=0,proposals=0,unique_evaluated=0,duplicates=0,cache_resets=0,unique_center_states=0,center_cache_resets=0,alternative_decisions=0,reference_checks=0,geos_diagnostics=0,valid_improvements=0,native_fallbacks=0,within_1nm=0,within_10nm=0,within_25nm=0,within_100nm=0);timings=dict(model=0.,decode=0.,geometry=0.,reference=0.);nearest=1.;nearest_geos=1.;distance_sums=np.zeros(2);last=start;frame_index=4;batch_index=0
 while time.monotonic()-start<a.seconds:
  if (a.output.parent/'winner.json').exists():break
  t=time.perf_counter()
  if frame_index==4:
   try:base=Frame(p,radius,rng.uniform(-np.pi,np.pi),a)
   except ValueError:continue
   frame_index=0;counts['bases']+=1;norms=np.array([base.M.get_r(i,i)for i in range(45)])
  frame=base if frame_index==0 else base.quarter_turn(frame_index);frame_index+=1;counts['frames']+=1;timings['model']+=time.perf_counter()-t
  ambient,desired,target=frame.targets(rng,a.batch,a);t=time.perf_counter()
  if a.mode=='baseline':decoded=decoder.decode(target,frame.mu,frame.moves,frame.anchor_units);frame.patch(ambient,decoded)
  else:
   decoded,changes,distances=decoder.decode_random(target,frame.mu,norms,frame.moves,frame.anchor_units,seed=a.decoder_seed+batch_index,max_flips=a.max_flips,temperature=a.temperature,norm_power=a.norm_power,fraction=a.fraction);counts['alternative_decisions']+=int(changes.sum());distance_sums+=distances.sum(0)
  timings['decode']+=time.perf_counter()-t;batch_index+=1;pp,units,coeff,flags=decoded;counts['proposals']+=len(pp);keep,info=decoder.unique(units);counts['duplicates']+=int(info[0]);counts['cache_resets']+=int(info[2]);pp=pp[keep];units=units[keep];coeff=coeff[keep];ambient=ambient[keep];desired=desired[keep];target=target[keep]
  if not len(pp):continue
  center_keep,center_info=decoder.unique_centers(units);counts['unique_center_states']+=int(center_keep.sum());counts['center_cache_resets']+=int(center_info[2])
  t=time.perf_counter();areas,circles,ok=evaluate_batch(pp,centers=[0.,0.]);timings['geometry']+=time.perf_counter()-t;counts['unique_evaluated']+=len(pp);counts['native_fallbacks']+=int((ok==0).sum());violation=np.maximum(((areas-1e-6)/frame.constraint.scale).max(1),circles[:,2]-a.target)
  for threshold in (1,10,25,100):counts[f'within_{threshold}nm']+=int((violation<=threshold*1e-9).sum())
  k=int(np.argmin(violation))
  if violation[k]<nearest-1e-12:
   nearest=float(violation[k]);raw=frame.raw(coeff[k]);save(a.output/'nearest.json',raw)
   if nearest<1e-7:
    t=time.perf_counter();exact=polygon_areas(pp[k]);geos_v=float(max(((exact-1e-6)/frame.constraint.scale).max(),circles[k,2]-a.target));nearest_geos=min(nearest_geos,geos_v);counts['geos_diagnostics']+=1;timings['reference']+=time.perf_counter()-t
    emit('nearest',estimated_violation_nm=nearest*1e9,geos_violation_nm=geos_v*1e9,mec=float(circles[k,2]),maximum_area_error=float(np.max(np.abs(exact-areas[k]))),frame=counts['frames'])
  eligible=np.flatnonzero((areas.max(1)<=1e-6+1e-11)&(circles[:,2]<=elites.cap));eligible=eligible[np.argsort(circles[eligible,2])]
  for k in eligible:
   if circles[k,2]>=best-2e-12 and not elites.wants(units[k],circles[k,2]):continue
   raw=frame.raw(coeff[k]);t=time.perf_counter();validation=checked(raw);counts['reference_checks']+=1
   if validation.valid:
    elites.add(raw,units[k],validation.score,frame,coeff[k],ambient[k],target[k],desired[k],circles[k],a,counts['frames'])
    if validation.score<best-2e-12:
     save(a.output/'best.json',raw);response=check_solution(json.loads((a.output/'best.json').read_text()));(a.output/'best-api.json').write_text(json.dumps(response,indent=2)+'\n');emit('improvement',reference_score=validation.score,api=response)
     if response.get('valid'):best=float(response['score']);counts['valid_improvements']+=1
     if best<a.goal:(a.output.parent/'winner.json').write_bytes((a.output/'best.json').read_bytes())
   timings['reference']+=time.perf_counter()-t
  if time.monotonic()-last>10:
   emit('progress',counts=counts.copy(),best=best,nearest_nm=nearest*1e9,nearest_geos_nm=nearest_geos*1e9,unique_per_second=counts['unique_evaluated']/(time.monotonic()-start));last=time.monotonic()
 result=dict(mode=a.mode,counts=counts,timings=timings,best=best,nearest_nm=nearest*1e9,nearest_geos_nm=nearest_geos*1e9,unique_per_second=counts['unique_evaluated']/(time.monotonic()-start),mean_squared_distance_ratio=None if not distance_sums[0]else float(distance_sums[1]/distance_sums[0]),arguments=vars(a).copy());result['arguments']['output']=str(a.output);(a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');emit('done',**result)


if __name__=='__main__':main()
