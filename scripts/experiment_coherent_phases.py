"""Solve exact cached diagonal phase blocks, without regenerating pair geometry."""
from __future__ import annotations
import os
for k in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import experiment_conflict_master as master
from interval_order_optimization import add_hamming_interval_pair
from exact_domain_pruning import IntegerCut,pack_relations,prune,verify_trace
from semicircle_packing.polish import checked,save,load
from arena import check_solution


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prefix',default='experiments/2026-11-04/round3/conflicts/phase-pool-search');ap.add_argument('--cuts',action='append',required=True);ap.add_argument('--masks',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--target',type=float,default=2.9485269238806113);ap.add_argument('--phases',type=int,default=10);ap.add_argument('--start-phase',type=int,default=0);ap.add_argument('--phase-size',type=int,default=63);ap.add_argument('--seconds',type=float,default=120);ap.add_argument('--per-phase',type=float,default=8);ap.add_argument('--seed',type=int,default=3110711);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);started=time.monotonic()
 with np.load(a.prefix+'.domains.npz')as z:domains=[z[f'p{i}']for i in range(len(z.files))]
 assert all(len(d)==25*a.phase_size for d in domains);digest=hashlib.sha256(b''.join(d.tobytes()for d in domains)).hexdigest();masks={}
 with np.load(a.prefix+'.tables.npz')as z:
  assert str(z['_domain_digest'])==digest
  for key in z.files:
   if key.startswith('_'):continue
   i,j=map(int,key.split('_'));v=z[key];m=np.zeros((len(domains[i]),len(domains[j])),dtype=bool);m[v[:,0],v[:,1]]=True;masks[i,j]=m
 with np.load(a.masks)as z:global_masks=[z[f'p{i}']for i in range(15)]
 mask_meta=json.loads(Path(a.masks).with_name('summary.json').read_text());assert mask_meta['target']==a.target and mask_meta['domain_digest']==digest
 cuts=[]
 for path in a.cuts:
  for item in json.loads(Path(path).read_text()):item['coefficients']={int(i):np.asarray(v,dtype=np.int64)for i,v in item['coefficients'].items()};cuts.append(item)
 insert=master.add_support_cut;pair_insert=master.add_pair_constraint
 def hook(local_cuts):
  def add(model,variables,radius,cut,serial):
   insert(model,variables,radius,cut,serial)
   if serial in (0,'control'):
    for k,extra in enumerate(local_cuts):insert(model,variables,radius,extra,f'coherent_{k}')
  return add
 def project(ids):return[dict(c,coefficients={i:v[ids[i]]for i,v in c['coefficients'].items()})for c in cuts]
 try:
  master.add_pair_constraint=add_hamming_interval_pair;master.add_support_cut=hook(project([np.array([0])for _ in domains]));control=master.positive_control([d[:1]for d in domains],[])
 finally:master.add_support_cut=insert;master.add_pair_constraint=pair_insert
 metadata=dict(prefix=a.prefix,domain_digest=digest,target=a.target,cut_count=len(cuts),masks=a.masks,external_control=control,phase_order='0,-25,+25,-50,+50,...,-300,+300 microradians',pair_geometry_regenerated=0)
 (a.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');print('CONTROL',json.dumps(metadata),flush=True);reports=[]
 for phase in range(a.start_phase,min(a.start_phase+a.phases,25)):
  if time.monotonic()-started>=a.seconds:break
  begin=phase*a.phase_size;ids=[np.flatnonzero(global_masks[i][begin:begin+a.phase_size])+begin for i in range(15)];phase_start=time.monotonic();offset=0 if phase==0 else(-1 if phase%2 else 1)*((phase+1)//2)*25
  if any(not len(v)for v in ids):reports.append(dict(phase=phase,offset_microradians=offset,infeasible=True,reason='global_mask_empty_phase'));continue
  local_masks={(i,j):m[np.ix_(ids[i],ids[j])]for(i,j),m in masks.items()};local_cuts=project(ids);integer=[IntegerCut(c['coefficients'],int(np.floor(a.target*c['scale']))+int(c['offset']))for c in local_cuts];sizes=list(map(len,ids));packed=pack_relations(local_masks);active,trace,status=prune(sizes,packed,integer);verify_trace(sizes,packed,integer,active,trace,status)
  report=dict(phase=phase,offset_microradians=offset,initial_sizes=sizes,remaining_sizes=[int(x.sum())for x in active],pruning_status=status,pair_cells_checked=sum(m.size for m in local_masks.values()),pruning_seconds=time.monotonic()-phase_start)
  (a.output/f'phase-{phase:02d}.pruning-trace.json').write_text(json.dumps(dict(original_pose_ids=[v.tolist()for v in ids],trace=trace,status=status))+'\n')
  if not status['infeasible']:
   ids=[v[active[i]]for i,v in enumerate(ids)];dd=[d[ids[i]]for i,d in enumerate(domains)];tables=[]
   for(i,j),m in masks.items():
    mm=m[np.ix_(ids[i],ids[j])]
    if not mm.all():tables.append((i,j,np.column_stack(np.nonzero(mm))))
   remaining=a.seconds-(time.monotonic()-started);prefix=a.output/f'phase-{phase:02d}.json';(a.output/f'phase-{phase:02d}.pose-ids.json').write_text(json.dumps([v.tolist()for v in ids])+'\n')
   try:
    master.add_pair_constraint=add_hamming_interval_pair;master.add_support_cut=hook(project(ids));result=master.solve(dd,a.target,str(prefix),seconds=min(a.per_phase,remaining),seed=a.seed+phase,tables=tables,support_cuts=True,check_positive=False)
   finally:master.add_support_cut=insert;master.add_pair_constraint=pair_insert
   report['solver']=result
   candidate=prefix if result.get('valid')else prefix.with_suffix('.best-valid.json')
   if candidate.exists():
    validation=checked(load(str(candidate)))
    if validation.valid and validation.score<checked(load('solution.json')).score:
     api=check_solution(json.loads(candidate.read_text()));candidate.with_suffix('.api.json').write_text(json.dumps(api,indent=2)+'\n');report['api']=api;print('API',json.dumps(api),flush=True)
  report['seconds']=time.monotonic()-phase_start;reports.append(report);print('PHASE',json.dumps(report),flush=True)
  if report.get('api',{}).get('valid')and report['api']['score']<a.target:break
 result=dict(metadata=metadata,seconds=time.monotonic()-started,reports=reports,phases_tested=len(reports),scope='Same-phase diagonal blocks only; original categorical tables/cuts projected exactly');(a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('DONE',json.dumps(result),flush=True)


if __name__=='__main__':main()
