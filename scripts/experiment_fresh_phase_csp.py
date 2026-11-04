"""Bounded independent coherent finite CSPs around full-precision packings."""
from __future__ import annotations
import os
for k in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,shlex,sys,time
from pathlib import Path
import numpy as np
import experiment_conflict_master as master
import experiment_support_bundle as support
from experiment_rescue_csp import domains_at
from interval_order_optimization import add_hamming_interval_pair
from exact_domain_pruning import IntegerCut,prune,verify_trace
from semicircle_packing.polish import load,save,checked
from semicircle_packing.geometry import canonicalize_theta
from semicircle_packing.fast_mec import compute_mec_fast
from arena import check_solution


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',action='append',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seconds',type=float,default=300);ap.add_argument('--per-phase',type=float,default=8);ap.add_argument('--target',type=float,default=2.9485269238806113);ap.add_argument('--seed',type=int,default=3110721);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);start=time.monotonic();rng=np.random.default_rng(a.seed);sources=[]
 for i,name in enumerate(a.input):
  raw=Path(name).read_bytes();(a.output/f'source-{i}.json').write_bytes(raw);p=load(name);circle=compute_mec_fast(p);p[:,:2]-=circle[:2];sources.append((name,p,circle,hashlib.sha256(raw).hexdigest()))
 (a.output/'research.md').write_text('# Fresh coherent phase CSPs\n\nFresh global rotations and subgrid translations of full-precision sources. Each phase retains all per-pieceXYZ±1/theta±3 choices, with free enclosing center, exact pair relations, Hamming interval encoding and55 balanced support inequalities. Invalid rounded bases remain explicit; they are not positive controls. Support-only integer pruning precedes exact pair preparation. Each finite model has a separate deadline within the global budget.\n\n```text\n'+shlex.join(['.venv/bin/python',*sys.argv])+'\n```\n')
 insert=master.add_support_cut;pair_insert=master.add_pair_constraint
 def hook(cuts):
  def add(model,variables,radius,cut,serial):
   insert(model,variables,radius,cut,serial)
   if serial in(0,'control'):
    for k,c in enumerate(cuts):insert(model,variables,radius,c,f'fresh_{k}')
  return add
 known=load('solution.json');validation=checked(known);assert validation.valid;known_domains=[known[i:i+1]for i in range(15)];control_bundle=support.support_bundle(known_domains,known,validation.mec)
 try:
  master.add_pair_constraint=add_hamming_interval_pair;master.add_support_cut=hook(control_bundle);control=master.positive_control(known_domains,[],validation.score)
 finally:master.add_pair_constraint=pair_insert;master.add_support_cut=insert
 (a.output/'external-control.json').write_text(json.dumps(control,indent=2)+'\n');reports=[];phase=0
 while time.monotonic()-start<a.seconds:
  phase_start=time.monotonic();name,original,source_circle,digest=sources[phase%len(sources)];angle=rng.uniform(-np.pi,np.pi);translation=rng.uniform(-.5,.5,2)*1e-6;c,s=np.cos(angle),np.sin(angle);physical=original.copy();physical[:,:2]=physical[:,:2]@np.array([[c,s],[-s,c]])+translation;physical[:,2]=np.arctan2(np.sin(physical[:,2]+angle),np.cos(physical[:,2]+angle));raw=np.round(physical,6);base=raw.copy();base[:,2]=[canonicalize_theta(v)for v in base[:,2]]
  out=a.output/f'phase-{phase:03d}';out.mkdir();save(out/'raw-base.json',raw);(out/'continuous-base.json').write_text(json.dumps([dict(x=float(x),y=float(y),theta=float(t))for x,y,t in physical],indent=2)+'\n');base_validation=checked(raw);circle=compute_mec_fast(base);domains=domains_at(base,np.empty((0,15,3),dtype=np.int64));bundle=support.support_bundle(domains,base,circle);integer=[IntegerCut(c['coefficients'],int(np.floor(a.target*c['scale']))+int(c['offset']))for c in bundle];active,trace,status=prune(list(map(len,domains)),{},integer);verify_trace(list(map(len,domains)),{},integer,active,trace,status)
  metadata=dict(phase=phase,source=name,source_sha256=digest,source_continuous_radius=source_circle[2],angle=angle,translation=translation.tolist(),seed=a.seed,base_valid=base_validation.valid,base_score=base_validation.score,base_errors=base_validation.errors,base_radius=circle[2],initial_domain_sizes=list(map(len,domains)),remaining_domain_sizes=[int(v.sum())for v in active],support_cut_count=len(bundle),support_pruning=status)
  (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');(out/'support-pruning-trace.json').write_text(json.dumps(trace)+'\n');print('FRESH_BASE',json.dumps(metadata),flush=True)
  if base_validation.valid and base_validation.score<checked(load('solution.json')).score:
   api=check_solution(json.loads((out/'raw-base.json').read_text()));(out/'base-api.json').write_text(json.dumps(api,indent=2)+'\n');metadata['api']=api;print('FRESH_API',json.dumps(api),flush=True)
   if api.get('valid')and api['score']<a.target:
    reports.append(metadata);(a.output/'summary.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,reports=reports,phases=len(reports)),indent=2)+'\n');break
  if not status['infeasible']:
   ids=[np.flatnonzero(v)for v in active];domains=[d[ids[i]]for i,d in enumerate(domains)];cuts=[dict(c,coefficients={i:v[ids[i]]for i,v in c['coefficients'].items()})for c in bundle];(out/'pose-ids.json').write_text(json.dumps([v.tolist()for v in ids])+'\n');remaining=a.seconds-(time.monotonic()-start)
   if remaining>0:
    prefix=out/'search.json'
    try:
     master.add_pair_constraint=add_hamming_interval_pair;master.add_support_cut=hook(cuts);result=master.solve(domains,a.target,str(prefix),seconds=min(a.per_phase,remaining),seed=a.seed+phase,support_cuts=True,check_positive=False,band_screen=True,local_loss=True)
    finally:master.add_pair_constraint=pair_insert;master.add_support_cut=insert
    metadata['solver']=result;candidate=prefix if result.get('valid')else prefix.with_suffix('.best-valid.json')
    if candidate.exists():
     q=load(str(candidate));check=checked(q)
     if check.valid and check.score<checked(load('solution.json')).score:
      api=check_solution(json.loads(candidate.read_text()));candidate.with_suffix('.api.json').write_text(json.dumps(api,indent=2)+'\n');metadata['api']=api;print('FRESH_API',json.dumps(api),flush=True)
  metadata['seconds']=time.monotonic()-phase_start;reports.append(metadata);(a.output/'summary.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,reports=reports,phases=len(reports)),indent=2)+'\n');print('FRESH_PHASE',json.dumps(metadata),flush=True);phase+=1
  if metadata.get('api',{}).get('valid')and metadata['api']['score']<a.target:break


if __name__=='__main__':main()
