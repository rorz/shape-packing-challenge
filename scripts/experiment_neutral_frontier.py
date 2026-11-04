"""Bounded frontier scan of natively feasible joint center-angle edges."""
from __future__ import annotations
import os
for name in ('VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import argparse,itertools,json,shlex,sys,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from experiment_neutral_network import NativeState,canonical,position_key,polygon,radii_at,PAIRS
from experiment_contact_csp import find_assignment
from audit2_boundary_centers import boundary_centers
from semicircle_packing.polish import load,save,checked
from semicircle_packing.exact_area import polygon_intersection_areas
from breakout_geometry import evaluate_batch
from arena import check_solution


def moves_for(state):
 graph=[set()for _ in range(15)]
 for k,(i,j)in enumerate(PAIRS):
  if state.areas[k]>1e-9:graph[i].add(j);graph[j].add(i)
 clusters=[(i,)for i in range(15)]+[tuple(pair)for pair in PAIRS if pair[1]in graph[pair[0]]]
 triples={tuple(sorted((i,j,k)))for i in range(15)for j,k in itertools.combinations(graph[i],2)};clusters+=sorted(triples)
 moves=[]
 for group in clusters:
  for dx,dy in itertools.product((-1,0,1),repeat=2):
   if not(dx or dy):continue
   m=np.zeros((15,3),dtype=int);m[list(group),:2]=[dx,dy];moves.append(m)
 for name in ['experiments/sampling_sweep/b3.141616-c10.0-bias5/lll_basis.npz','experiments/lll_basis.npz']:
  if Path(name).exists():
   with np.load(name)as z:
    for m in np.rint(z['moves']).astype(int):
     if np.any(m[:,:2]-m[0,:2]):moves.extend([m,-m])
   break
 unique={}
 for m in moves:
  m=m.copy();m[:,:2]-=m[0,:2]
  if np.any(m[:,:2]):unique[tuple(m.ravel())]=m
 return list(unique.values())


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',default='solution.json');ap.add_argument('--output',required=True);ap.add_argument('--seconds',type=float,default=180);ap.add_argument('--steps',type=int,default=50);ap.add_argument('--relay',type=float,default=3e-7);ap.add_argument('--seed',type=int,default=110424);ap.add_argument('--nodes',type=int,default=10000);ap.add_argument('--max-centers',type=int,default=4);args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True);start=time.monotonic();rng=np.random.default_rng(args.seed);best=checked(load('solution.json')).score;cap=best+args.relay
 log=out/'steps.md';events=out/'events.jsonl'
 log.write_text('# Joint feasible frontier — final bounded block\n\nTest endpoint feasibility of one-piece, contact-pair, connected-triple translations and saved LLL moves. Each center proposal gets a ±'+str(args.steps)+' microradian angular CSP. Arc consistency precedes the generic free-center oracle. Native GEOS checks every proposed accepted endpoint; estimator-compatible but native-invalid table entries are cut lazily. Valid relays stay below incumbent+3e-7. No shared solution is overwritten.\n\n```text\n'+shlex.join(['.venv/bin/python',*sys.argv])+'\n```\n')
 def emit(event,**data):
  row=dict(time=datetime.now(timezone.utc).isoformat(),event=event,seconds=time.monotonic()-start,**data);line=json.dumps(row,allow_nan=False);print(line,flush=True)
  with events.open('a')as f:f.write(line+'\n')
  if event not in ('case','progress'):
   with log.open('a')as f:f.write('\n- '+line+'\n')
 state=NativeState(canonical(load(args.input)));save(out/'start.json',state.p);queue=[state];visited_centers={state.position_key};observed_centers={state.position_key};tested=set();counts=dict(proposals=0,area_unsat=0,center_excluded=0,center_incomplete=0,csp_cutoffs=0,native_patches=0,valid_center_edges=0)
 while queue and time.monotonic()-start<args.seconds:
  state=queue.pop(0);moves=moves_for(state);proposals=np.array([canonical(state.p+m*1e-6)for m in moves]);areas,circles,_=evaluate_batch(proposals,centers=[state.cx,state.cy]);energy=np.sum((np.maximum(areas-1e-6,0)*1e9)**2,axis=1);rank=energy+np.maximum(circles[:,2]-best,0)*1e8;order=np.argsort(rank)
  emit('frontier',moves=len(moves),center_states=len(visited_centers),score=state.radius)
  for number in order:
   if time.monotonic()-start>=args.seconds:break
   q=proposals[number];domain_key=tuple(np.rint(q*1e6).astype(int).ravel())
   if domain_key in tested:continue
   tested.add(domain_key);counts['proposals']+=1
   domains=[]
   for row in q:
    values=sorted({canonicalize(row[2]+k*1e-6)for k in range(-args.steps,args.steps+1)})
    d=np.tile(row,(len(values),1));d[:,2]=values;domains.append(np.ascontiguousarray(d))
   variants=np.vstack(domains);N=len(variants);offsets=np.r_[0,np.cumsum([len(d)for d in domains])];allareas=polygon_intersection_areas(variants);adj=[[]for _ in domains];maps={}
   for i,j in PAIRS:
    u=np.arange(offsets[i],offsets[i+1])[:,None];v=np.arange(offsets[j],offsets[j+1])[None,:];mask=allareas[u*(2*N-u-1)//2+v-u-1]<=1e-6+2e-11
    if mask.all():continue
    adj[i].append(j);adj[j].append(i);maps[i,j]=[int.from_bytes(np.packbits(r,bitorder='little').tobytes(),'little')for r in mask];maps[j,i]=[int.from_bytes(np.packbits(r,bitorder='little').tobytes(),'little')for r in mask.T]
   answer,nodes=find_assignment(domains,adj,maps,rng,args.nodes)
   if answer is None:
    if nodes>args.nodes:counts['csp_cutoffs']+=1
    else:counts['area_unsat']+=1
    emit('case',index=int(number),area_sat=False,nodes=nodes);continue
   records,info=boundary_centers(q,[d[:,2]for d in domains],target=cap,max_combinations=256)
   if not info['complete_geometry']:counts['center_incomplete']+=1
   if not records:counts['center_excluded']+=1;emit('case',index=int(number),area_sat=True,centers=0,geometry=info);continue
   accepted=False;polys={}
   for record in sorted(records,key=lambda r:r['radius']):
    if time.monotonic()-start>=args.seconds:break
    unary=[]
    for i,d in enumerate(domains):
     allow=radii_at(d,record['center'])<=cap+2e-12
     if i in record['fixed_angles']:allow &=np.abs(d[:,2]-record['fixed_angles'][i])<1e-12
     unary.append(sum(1<<k for k,v in enumerate(allow)if v))
    for patch in range(64):
     assignment,nodes=find_assignment(domains,adj,maps,rng,args.nodes,initial=unary)
     if assignment is None:
      if nodes>args.nodes:counts['csp_cutoffs']+=1
      break
     candidate=np.array([domains[i][a]for i,a in enumerate(assignment)]);bad=[];native_areas=np.zeros(105)
     for k,(i,j)in enumerate(PAIRS):
      if np.linalg.norm(candidate[i,:2]-candidate[j,:2])>2:continue
      for piece in [i,j]:
       tag=(piece,assignment[piece])
       if tag not in polys:polys[tag]=polygon(candidate[piece])
      value=polys[i,assignment[i]].intersection(polys[j,assignment[j]]).area;native_areas[k]=value
      if value>1e-6:bad.append((i,j,value))
     if bad:
      counts['native_patches']+=len(bad)
      for i,j,value in bad:
       if(i,j)not in maps:
        maps[i,j]=[(1<<len(domains[j]))-1]*len(domains[i]);maps[j,i]=[(1<<len(domains[i]))-1]*len(domains[j]);adj[i].append(j);adj[j].append(i)
       maps[i,j][assignment[i]]&=~(1<<assignment[j]);maps[j,i][assignment[j]]&=~(1<<assignment[i])
      continue
     native=NativeState(candidate,[polys.get((i,assignment[i]),polygon(candidate[i]))for i in range(15)],native_areas)
     if native.radius>cap+2e-12:break
     result=checked(candidate)
     if not result.valid:emit('validation_disagreement',errors=result.errors);break
     observed_centers.add(native.position_key);counts['valid_center_edges']+=1;name=out/f"valid-center-{counts['valid_center_edges']:03d}.json";save(name,candidate);emit('valid_center',path=str(name),score=result.score,native_max_area=float(native_areas.max()),improving=result.score<best,geometry=info)
     if result.score<best-2e-12:
      best=result.score;save(out/'best.json',candidate);response=check_solution(json.loads((out/'best.json').read_text()));(out/'best-api.json').write_text(json.dumps(response,indent=2)+'\n');emit('improvement',score=best,api=response)
      if response.get('valid')and response.get('score',99)<2.9485269238856113:emit('done',**counts,best=best);return
     if native.position_key not in visited_centers and len(visited_centers)<args.max_centers:visited_centers.add(native.position_key);queue.append(native)
     accepted=True;break
    if accepted:break
   emit('case',index=int(number),area_sat=True,centers=len(records),accepted=accepted,geometry=info)
  if len(visited_centers)>=args.max_centers:break
 emit('done',**counts,best=best,distinct_centers=len(observed_centers),enqueued_centers=len(visited_centers),candidate_domains_tested=len(tested),cutoff='wall_time'if time.monotonic()-start>=args.seconds else'frontier_or_center_limit')


def canonicalize(theta):
 from semicircle_packing.geometry import canonicalize_theta
 return canonicalize_theta(theta)

if __name__=='__main__':main()
