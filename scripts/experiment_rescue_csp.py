"""Frozen low-radius nearpoint rescue; controls are external to invalid seed."""
from __future__ import annotations
import os
for k in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,itertools,json,time
from pathlib import Path
import numpy as np
import experiment_conflict_master as master
import experiment_support_bundle as support
import interval_pair_relation as relations
from interval_order_optimization import add_hamming_interval_pair
from semicircle_packing.geometry import canonicalize_theta
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.polish import load,checked,save
from arena import check_solution


def domains_at(base,moves):
 states=[base]+[base+sign*move*1e-6 for move in moves for sign in (-1,1)];domains=[]
 offsets=[(0,0,0)]+[v for v in itertools.product(range(-1,2),range(-1,2),range(-3,4))if v!=(0,0,0)]
 for piece in range(15):
  values={}
  for state in states:
   for delta in offsets:
    q=np.round(state[piece]+np.asarray(delta)*1e-6,6);q[2]=canonicalize_theta(q[2])
    if canonicalize_theta(q[2])!=q[2]:raise ValueError('Unstable raw/canonical domain alias')
    values.setdefault(tuple(np.rint(q*1e6).astype(np.int64)),q)
  domains.append(np.asarray(list(values.values())))
 return domains


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',required=True);ap.add_argument('--state',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seconds',type=float,default=300);ap.add_argument('--seed',type=int,default=3110611);ap.add_argument('--target',type=float,default=2.9485269238);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);started=time.monotonic();raw=load(a.input)
 state=np.load(a.state);assert np.array_equal(np.round(raw,6),np.round(state['placements'],6));base=np.round(raw,6);base[:,2]=[canonicalize_theta(v)for v in base[:,2]];assert np.array_equal(np.rint(base*1e6),np.rint(state['geometry']*1e6))
 moves=np.rint(state['moves']).astype(np.int64);moves[:,:,:2]-=moves[:,:1,:2];moves=np.unique(moves.reshape(-1,45),axis=0);moves=moves[np.any(moves,axis=1)].reshape(-1,15,3);save(a.output/'frozen-input.json',raw)
 meta=dict(input=a.input,state=a.state,input_sha256=hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),state_sha256=hashlib.sha256(Path(a.state).read_bytes()).hexdigest(),matching_state=True,all_physical_basis_moves=len(moves),seed=a.seed,target=a.target,total_budget_seconds=a.seconds,scope='Invalid nearpoint; external known-valid engine control; no seed-validity claim')
 (a.output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print('RESCUE',json.dumps(meta),flush=True)
 known=load('solution.json');save(a.output/'external-control-input.json',known);control=checked(known);assert control.valid;control_domains=[known[i:i+1]for i in range(15)];control_tables,_=master.exact_tables(control_domains)
 old_relation=relations.add_allowed_interval_pair
 try:
  relations.add_allowed_interval_pair=add_hamming_interval_pair
  external=support.solve_bundle(control_domains,control_tables,control.score+1e-8,str(a.output/'external-control.json'),seconds=10,expected_score=control.score,seed=a.seed,use_intervals=True)
 finally:relations.add_allowed_interval_pair=old_relation
 assert external['valid'];(a.output/'external-control-status.json').write_text(json.dumps(dict(valid=True,score=control.score,external_to_rescue_domain=True),indent=2)+'\n')
 reports=[]
 for name,selected,stage_cap in [('local',np.empty((0,15,3),dtype=np.int64),30.),('all-basis',moves,a.seconds)]:
  remaining=a.seconds-(time.monotonic()-started)
  if remaining<=1:break
  domains=domains_at(base,selected);circle=compute_mec_fast(base);bundle=support.support_bundle(domains,base,circle);prefix=a.output/(name+'.json');prefix.with_suffix('.bootstrap-cuts.json').write_text(json.dumps([master.cut_record(c)for c in bundle],indent=2)+'\n')
  print('RESCUE_STAGE',json.dumps(dict(name=name,domain_sizes=[len(d)for d in domains],physical_moves=len(selected),bootstrap_cuts=len(bundle),remaining_seconds=remaining)),flush=True)
  insert=master.add_support_cut;pair_insert=master.add_pair_constraint
  def insert_bundle(model,variables,radius,cut,serial):
   insert(model,variables,radius,cut,serial)
   if serial==0:
    for k,extra in enumerate(bundle):insert(model,variables,radius,extra,f'rescue_{k}')
  try:
   master.add_support_cut=insert_bundle;master.add_pair_constraint=add_hamming_interval_pair
   report=master.solve(domains,a.target,str(prefix),seconds=min(stage_cap,remaining),seed=a.seed,support_cuts=True,check_positive=False,band_screen=True,local_loss=True)
  finally:master.add_support_cut=insert;master.add_pair_constraint=pair_insert
  reports.append(dict(stage=name,report=report))
  candidate=prefix if report.get('valid')else prefix.with_suffix('.best-valid.json')
  if candidate.exists():
   result=checked(load(str(candidate)))
   if result.valid and result.score<checked(load('solution.json')).score:
    api=check_solution(json.loads(candidate.read_text()));candidate.with_suffix('.api.json').write_text(json.dumps(api,indent=2)+'\n');print('RESCUE_API',json.dumps(api),flush=True)
    if api.get('valid')and api.get('score',99)<a.target:break
 result=dict(seconds=time.monotonic()-started,metadata=meta,external_control_valid=True,reports=reports);(a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('RESCUE_DONE',json.dumps(result),flush=True)


if __name__=='__main__':main()
