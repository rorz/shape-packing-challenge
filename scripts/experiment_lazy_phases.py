"""New coherent phase ensembles with selected uncertain-cell GEOS completion."""
from __future__ import annotations
import os
for k in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,itertools,json,shlex,sys,time
from pathlib import Path
import numpy as np
from experiment_support_bundle import support_bundle
from exact_domain_pruning import IntegerCut,prune,verify_trace
from lazy_phase_solver import solve_phase,LEADER
from semicircle_packing.geometry import canonicalize_theta
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.polish import checked,load,save


def make_domains(base,xy_width,angle_width):
 offsets=[(0,0,0)]+[v for v in itertools.product(range(-xy_width,xy_width+1),range(-xy_width,xy_width+1),range(-angle_width,angle_width+1))if v!=(0,0,0)];domains=[]
 for row in base:
  values={}
  for delta in offsets:
   p=np.round(row+np.asarray(delta)*1e-6,6);p[2]=canonicalize_theta(p[2]);assert canonicalize_theta(p[2])==p[2];values.setdefault(tuple(np.rint(p*1e6).astype(np.int64)),p)
  domains.append(np.asarray(list(values.values())))
 return domains


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',action='append',default=[]);ap.add_argument('--resume',action='append',default=[]);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seconds',type=float,default=300);ap.add_argument('--per-phase',type=float,default=15);ap.add_argument('--phases',type=int,default=100000);ap.add_argument('--xy-width',type=int,default=1);ap.add_argument('--angle-width',type=int,default=12);ap.add_argument('--target',type=float,default=2.9485269238806113);ap.add_argument('--seed',type=int,default=3110811);a=ap.parse_args();assert a.input or a.resume;a.output.mkdir(parents=True,exist_ok=True);start=time.monotonic();rng=np.random.default_rng(a.seed);sources=[]
 for i,path in enumerate(a.input):
  raw=Path(path).read_bytes();(a.output/f'source-{i}.json').write_bytes(raw);p=load(path);circle=compute_mec_fast(p);p[:,:2]-=circle[:2];sources.append((path,p,circle,hashlib.sha256(raw).hexdigest()))
 (a.output/'research.md').write_text('# Lazy exact coherent phases\n\nThe conservative native band rejects only proven incompatible cells. Every uncertain cell remains allowed until selected by CP; exact GEOS decisions are cached and false pairs become two-variable no-goods. Exact pair-feasible leaves receive a true free-center MEC check and additional enclosure cuts. Selected geometry is validated before API checks. INFEASIBLE is a statement about the finite relaxed-and-refined model; deadlines are UNKNOWN.\n\n```text\n'+shlex.join(['.venv/bin/python',*sys.argv])+'\n```\n')
 raw=load('solution.json');check=checked(raw);assert check.valid;known=np.round(raw,6);known[:,2]=[canonicalize_theta(t)for t in known[:,2]];control=solve_phase([known[i:i+1]for i in range(15)],known,check.score+1e-9,a.output/'external-control',seconds=8,seed=a.seed,control=True);assert control['valid'];reports=[];phase=0
 while time.monotonic()-start<a.seconds and phase<a.phases:
  out=a.output/f'phase-{phase:03d}';out.mkdir();metadata=dict(phase=phase,seed=a.seed,xy_width=a.xy_width,angle_width=a.angle_width)
  if phase<len(a.resume):
   previous=Path(a.resume[phase]);z=np.load(previous/'search.domains.npz');domains=[z[f'p{i}']for i in range(len(z.files))];raw=load(str(previous/'raw-base.json'));base=np.round(raw,6);base[:,2]=[canonicalize_theta(t)for t in base[:,2]];expected=json.loads((previous/'search.report.json').read_text());metadata.update(mode='saved_exhaustive_comparison',source=str(previous),expected_status=expected.get('last_status'),expected_finite_exhausted=expected.get('finite_exhausted',False))
  else:
   if not sources:break
   name,p,circle,digest=sources[(phase-len(a.resume))%len(sources)];angle=rng.uniform(-np.pi,np.pi);translation=rng.uniform(-.5,.5,2)*1e-6;co,sn=np.cos(angle),np.sin(angle);physical=p.copy();physical[:,:2]=physical[:,:2]@np.array([[co,sn],[-sn,co]])+translation;physical[:,2]=np.arctan2(np.sin(physical[:,2]+angle),np.cos(physical[:,2]+angle));raw=np.round(physical,6);base=raw.copy();base[:,2]=[canonicalize_theta(t)for t in base[:,2]];domains=make_domains(base,a.xy_width,a.angle_width);validation=checked(raw);metadata.update(mode='new_phase',source=name,source_sha256=digest,source_radius=circle[2],angle=angle,translation=translation.tolist(),base_valid=validation.valid,base_score=validation.score,base_errors=validation.errors)
  save(out/'raw-base.json',raw);circle=compute_mec_fast(base);cuts=support_bundle(domains,base,circle);integer=[IntegerCut(c['coefficients'],int(np.floor(a.target*c['scale']))+int(c['offset']))for c in cuts];active,trace,status=prune(list(map(len,domains)),{},integer);verify_trace(list(map(len,domains)),{},integer,active,trace,status);metadata.update(base_radius=circle[2],initial_sizes=list(map(len,domains)),support_pruning=status,remaining_sizes=[int(v.sum())for v in active]);(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');(out/'support-pruning.json').write_text(json.dumps(trace)+'\n');print('LAZY_BASE',json.dumps(metadata),flush=True)
  if not status['infeasible']:
   ids=[np.flatnonzero(v)for v in active];domains=[d[ids[i]]for i,d in enumerate(domains)];(out/'pose-ids.json').write_text(json.dumps([v.tolist()for v in ids])+'\n');remaining=a.seconds-(time.monotonic()-start)
   if remaining>0:metadata['solver']=solve_phase(domains,base,a.target,out/'search',seconds=min(a.per_phase,remaining),seed=a.seed+phase)
  reports.append(metadata);(a.output/'summary.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,reports=reports,phases=len(reports),external_control=control),indent=2)+'\n');print('LAZY_PHASE',json.dumps(metadata),flush=True);phase+=1
  if metadata.get('solver',{}).get('api',{}).get('valid')and metadata['solver']['api']['score']<LEADER:break


if __name__=='__main__':main()
