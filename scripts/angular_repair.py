"""Wider fixed-position angular CSP with free-center containment screening."""
from __future__ import annotations
import os
for name in ('VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import argparse,itertools,json,shlex,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from experiment_contact_csp import find_assignment
from semicircle_packing.polish import load,checked,save
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.exact_area import polygon_intersection_areas
from semicircle_packing.geometry import canonicalize_theta
from arena import check_solution


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',nargs='+',default=['experiments/2026-11-04/positions/best_continuous.json','experiments/2026-11-04/positions/angular-feedback_continuous.json']);parser.add_argument('--output',default='experiments/2026-11-04/angular');parser.add_argument('--target',type=float,default=2.94852695);parser.add_argument('--accept-score',type=float,default=2.948526966725367);parser.add_argument('--seconds',type=float,default=180);parser.add_argument('--seed',type=int,default=4110415);args=parser.parse_args()
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True);log=Path('experiments/2026-11-04/angular-repair-log.md');events=out/'events.jsonl';start=time.monotonic();rng=np.random.default_rng(args.seed)
 def emit(event,**data):
  row=dict(time=datetime.now(timezone.utc).isoformat(),event=event,seconds=time.monotonic()-start,**data);line=json.dumps(row);print(line,flush=True)
  with events.open('a') as f:f.write(line+'\n')
  with log.open('a') as f:f.write('\n- '+line+'\n')
 with log.open('a') as f:f.write('\n# Fixed-position angular repair — 2026-11-04\n\nHypothesis: continuous feasible angles may round incorrectly because piece 14 needs a different MEC center, or because floor/ceil omits a nearby angular branch. First optimize the center for each local integer angle of piece 14, then solve pairwise polygon compatibility with wider angle domains.\n\n```text\n'+shlex.join(['.venv/bin/python',*sys.argv])+'\n```\n')
 for filename in args.input:
  if time.monotonic()-start>args.seconds:break
  original=Path(filename);p=load(original);(out/(original.stem+'-input.json')).write_bytes(original.read_bytes());continuous=compute_mec_fast(p)
  # All other circles fit in the observed outward-contact enclosure. Using
  # full disks gives a center which is sufficient for every remaining angle.
  # Its lower bound is local to that outward-contact branch, not a claim about
  # arbitrary 180-degree reorientations and a different topology.
  disks=np.repeat(p[:14],2,axis=0);disks[:,2]=np.tile([0.,np.pi],14)
  radii=[]
  for step in range(-50,51):
   q=p[14].copy();q[2]=canonicalize_theta((round(q[2]*1e6)+step)*1e-6);circle=compute_mec_fast(np.vstack([disks,q]));radii.append((q[2],*circle))
  best=min(radii,key=lambda row:row[3]);emit('containment_scan',input=filename,continuous_radius=continuous[2],best_integer_angle=best[0],minimum_branch_radius=best[3],center=list(best[1:3]))
  np.savez(out/(original.stem+'-containment.npz'),scan=np.array(radii),input=p)
  if best[3]>=min(args.accept_score,args.target)+2e-12:
   emit('containment_obstruction',input=filename,excess=best[3]-args.target);continue
  allowed14=[row[0] for row in radii if row[3]<=args.target+2e-12]
  for steps in [2,5,10,50]:
   if time.monotonic()-start>args.seconds:break
   domains=[]
   for i,row in enumerate(p):
    values=allowed14 if i==14 else [canonicalize_theta((round(row[2]*1e6)+k)*1e-6) for k in range(-steps,steps+1)]
    values=sorted(set(values));q=np.tile(row,(len(values),1));q[:,2]=values;domains.append(np.ascontiguousarray(q))
   variants=np.vstack(domains);allareas=polygon_intersection_areas(variants);N=len(variants);offsets=np.r_[0,np.cumsum([len(d) for d in domains])];adj=[[]for _ in domains];maps={};nontrivial=[]
   for i,j in itertools.combinations(range(15),2):
    u=np.arange(offsets[i],offsets[i+1])[:,None];v=np.arange(offsets[j],offsets[j+1])[None,:];indices=u*(2*N-u-1)//2+(v-u-1);mask=allareas[indices]<=1e-6-1e-12
    if mask.all():continue
    nontrivial.append((i,j));adj[i].append(j);adj[j].append(i)
    maps[i,j]=[int.from_bytes(np.packbits(row,bitorder='little').tobytes(),'little') for row in mask]
    maps[j,i]=[int.from_bytes(np.packbits(row,bitorder='little').tobytes(),'little') for row in mask.T]
   # Record an explicit arc-consistency witness before optional search.
   remaining=[(1<<len(domain))-1 for domain in domains];queue=list(maps);trace=[];conflict=None
   while queue:
    i,j=queue.pop();bits=remaining[i];revised=0
    while bits:
     bit=bits&-bits;bits-=bit
     if maps[i,j][bit.bit_length()-1]&remaining[j]:revised|=bit
    if revised!=remaining[i]:
     trace.append(dict(variable=i,partner=j,before=remaining[i].bit_count(),after=revised.bit_count()))
     if not revised:conflict=(i,j);break
     remaining[i]=revised;queue.extend((k,i)for k in adj[i]if k!=j)
   if conflict is not None:
    emit('arc_consistency_conflict',input=filename,steps=steps,variable=conflict[0],partner=conflict[1],trace=trace[-12:])
   t=time.monotonic();answer,nodes=(None,0) if conflict else find_assignment(domains,adj,maps,rng,100000,initial=remaining)
   emit('csp',input=filename,steps=steps,domain_sizes=[len(d)for d in domains],nontrivial_pairs=nontrivial,nodes=nodes,satisfiable=answer is not None,solve_seconds=time.monotonic()-t,area_only=True)
   if answer is None:continue
   candidate=np.array([domains[i][k]for i,k in enumerate(answer)]);result=checked(candidate);emit('reference',input=filename,steps=steps,valid=result.valid,score=result.score,errors=result.errors)
   if result.valid and result.score<args.accept_score:
    target=out/(original.stem+'-feasible.json');save(target,candidate)
    if result.score<2.948526966725367:
     response=check_solution(json.loads(target.read_text()));target.with_suffix('.api.json').write_text(json.dumps(response,indent=2)+'\n');emit('api_improvement',path=str(target),response=response)
     if response.get('valid') and response['score']<2.9485269238856113:return
    else:emit('feasible_relay',path=str(target),score=result.score)
    break
 emit('done')

if __name__=='__main__':main()
