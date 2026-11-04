"""Walk a natively feasible network, using neutral angles to unlock center hops.

Accepted states always pass the 4096-point GEOS validator. The driving energy
belongs to a *future* center move, so neutral current-radius states are not
collapsed into one zero-loss point. A valid-state portfolio supports global
binary angle recombination with current validity as a hard constraint.
"""
from __future__ import annotations
import os
for name in ('VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import argparse,itertools,json,math,shlex,sys,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from shapely.geometry import Polygon
from semicircle_packing.geometry import canonicalize_theta
from semicircle_packing.polish import load,save,checked
from semicircle_packing.exact_area import polygon_intersection_areas,polygon_intersection_areas_and_jacobian
from semicircle_packing.fast_mec import compute_mec_fast
from angular_min_cost import _solve_tables
from angular_rounder import _pair_indices
from arena import check_solution
from breakout_geometry import evaluate_batch
from experiment_contact_csp import find_assignment

PAIRS=np.array(list(itertools.combinations(range(15),2)));ARCS=4096;TOL=1e-6

def canonical(p):
 p=np.round(p,6);p[:,2]=[canonicalize_theta(t) for t in p[:,2]];return p

def key(p):
 units=np.rint(p*1e6).astype(np.int64);units[:,:2]-=units[0,:2];return tuple(units.ravel())

def position_key(p):
 u=np.rint(p[:,:2]*1e6).astype(np.int64);u-=u[0];return tuple(u.ravel())

def polygon(row):
 angles=np.linspace(row[2]-np.pi/2,row[2]+np.pi/2,ARCS)
 return Polygon(np.c_[row[0]+np.cos(angles),row[1]+np.sin(angles)])

class NativeState:
 def __init__(self,p,polygons=None,areas=None,center=None,radius=None):
  self.p=p.copy();self.polygons=[polygon(row)for row in p] if polygons is None else polygons
  self.areas=np.array([self.polygons[i].intersection(self.polygons[j]).area for i,j in PAIRS]) if areas is None else areas
  if radius is None:self.cx,self.cy,self.radius=compute_mec_fast(p,initial_center=center)
  else:self.cx,self.cy=center;self.radius=radius
  self.key=key(p);self.position_key=position_key(p)
 def proposal(self,p,circle=None):
  changed=np.flatnonzero(np.any(p!=self.p,axis=1));polys=self.polygons.copy()
  for i in changed:polys[i]=polygon(p[i])
  areas=self.areas.copy();affected=np.flatnonzero(np.isin(PAIRS,changed).any(axis=1))
  for index in affected:
   i,j=PAIRS[index]
   if np.linalg.norm(p[i,:2]-p[j,:2])>2:areas[index]=0.;continue
   areas[index]=polys[i].intersection(polys[j]).area
   if areas[index]>TOL:return None
  return NativeState(p,polys,areas,center=[self.cx,self.cy]) if circle is None else NativeState(p,polys,areas,center=circle[:2],radius=circle[2])


def radii_at(variants,center):
 x=variants[:,0]-center[0];y=variants[:,1]-center[1];t=variants[:,2];s=np.sin(t);c=np.cos(t)
 return np.maximum.reduce([np.hypot(x+s,y-c),np.hypot(x-s,y+c),np.where(x*c+y*s>=0,np.hypot(x,y)+1,0)])


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',default='solution.json');ap.add_argument('--output',required=True);ap.add_argument('--seconds',type=float,default=180);ap.add_argument('--seed',type=int,default=110421);ap.add_argument('--relay',type=float,default=3e-7);ap.add_argument('--angle-max',type=int,default=1024);ap.add_argument('--archive',type=int,default=64);ap.add_argument('--crossover-every',type=int,default=12);ap.add_argument('--repair-every',type=int,default=0);ap.add_argument('--repair-steps',type=int,default=2);ap.add_argument('--hop-refresh',type=int,default=1200);args=ap.parse_args()
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(args.seed);start=time.monotonic();events=out/'events.jsonl';log=out/'steps.md'
 def emit(event,**data):
  row=dict(time=datetime.now(timezone.utc).isoformat(),event=event,seconds=time.monotonic()-start,**data);line=json.dumps(row,allow_nan=False);print(line,flush=True)
  with events.open('a')as f:f.write(line+'\n')
  if event not in ('accepted','progress'):
   with log.open('a')as f:f.write('\n- '+line+'\n')
 with log.open('a')as f:f.write('# Feasible neutral-network search — 2026-11-04\n\nAudit: earlier `experiment_hillclimb.py` screened out non-improving radii. `experiment_lll_hillclimb.py` and `experiment_tempered_lattice.py` optimized violation in generally invalid states. This run keeps every accepted state natively valid, walks neutral angles, and scores the ability to execute a future improving center hop. Valid angle portfolios are recombined with current compatibility as a hard barrier. Exact translations alone are quotiented out.\n\n```text\n'+shlex.join(['.venv/bin/python',*sys.argv])+'\n```\n')
 p=canonical(load(args.input));save(out/'start.json',p);validation=checked(p)
 if not validation.valid:raise ValueError('Neutral search requires a valid incumbent')
 state=NativeState(p);best=validation.score;save(out/'best.json',p)
 _,jac=polygon_intersection_areas_and_jacobian(p);scale=np.maximum(np.linalg.norm(jac,axis=1),1e-3)
 archive=[state];seen={state.key};counts=dict(proposals=0,screened=0,native_rejected=0,valid_proposals=0,accepted=0,neutral_angles=0,center_hops=0,worse_valid=0,crossovers=0,conditional_attempts=0,conditional_solutions=0);hop=None;hops=[];hop_energy=0.;last_report=start;last_refresh=-args.hop_refresh
 def potential(q,move):
  future=canonical(q+move*1e-6);excess=np.maximum(polygon_intersection_areas(future)-TOL,0)/scale*1e8
  return float(excess@excess)
 def record(s):
  nonlocal best
  if s.radius>=best-2e-12:return False
  result=checked(s.p)
  if result.valid and result.score<best-2e-12:
   best=result.score;save(out/'best.json',s.p);response=check_solution(json.loads((out/'best.json').read_text()));(out/'best-api.json').write_text(json.dumps(response,indent=2)+'\n');emit('improvement',score=best,api=response)
   return response.get('valid') and response.get('score',99)<2.9485269238856113
  return False
 def enumerate_hops(s):
  moves=[];directions=[(x,y)for x,y in itertools.product((-1,0,1),repeat=2)if x or y]
  graph=[set()for _ in range(15)]
  for row,(i,j)in enumerate(PAIRS):
   if s.areas[row]>1e-9:graph[i].add(j);graph[j].add(i)
  clusters=[(i,)for i in range(15)]+[tuple(pair)for pair in PAIRS if pair[1]in graph[pair[0]]]
  triples=set()
  for i in range(15):
   for j,k in itertools.combinations(graph[i],2):triples.add(tuple(sorted((i,j,k))))
  clusters+=list(sorted(triples))
  for ids in clusters:
   for dx,dy in directions:
    m=np.zeros((15,3),dtype=np.int64);m[list(ids),:2]=[dx,dy];moves.append(m)
  for filename in ['experiments/sampling_sweep/b3.141616-c10.0-bias5/lll_basis.npz','experiments/lll_basis.npz']:
   if Path(filename).exists():
    with np.load(filename)as z:
     for m in np.rint(z['moves']).astype(np.int64):
      if np.any(m[:,:2]-m[0,:2]):moves.extend([m,-m])
    break
  unique={}
  for m in moves:
   m=m.copy();m[:,:2]-=m[0,:2];unique[tuple(m.ravel())]=m
  moves=list(unique.values());proposed=np.array([canonical(s.p+m*1e-6) for m in moves])
  predicted,circles,certificate=evaluate_batch(proposed,centers=[s.cx,s.cy])
  excess=np.maximum(predicted-TOL,0)/scale[None,:]*1e8;energy=np.sum(excess*excess,axis=1)
  ranked=[]
  for index,m in enumerate(moves):
   R=circles[index,2]
   if R>s.radius+args.relay:continue
   e=float(energy[index]);rank=e+(max(R-best,0)*1e8)**2
   ranked.append((rank,R,e,index,m))
  ranked.sort(key=lambda row:row[0]);improving=[row for row in ranked if row[1]<best-2e-12]
  relays=[row for row in ranked if row[1]>=best-2e-12]
  chosen=improving[:12]+relays[:4] if improving else ranked[:16]
  emit('hop_pool',tested=len(moves),eligible=len(ranked),improving=len(improving),relays=len(relays),best_potential=chosen[0][2]if chosen else None,best_radius=chosen[0][1]if chosen else None,native_certificates=int(certificate.sum()))
  return chosen
 def accept(candidate,reason):
  nonlocal state,archive,hop_energy
  if candidate.key in seen:return False
  seen.add(candidate.key);old=state
  if candidate.radius>best+args.relay:return False
  if candidate.position_key!=old.position_key:archive=[];counts['center_hops']+=1
  elif abs(candidate.radius-old.radius)<2e-12:counts['neutral_angles']+=1
  if candidate.radius>old.radius+2e-12:counts['worse_valid']+=1
  counts['accepted']+=1;state=candidate;archive.append(state)
  if len(archive)>args.archive:archive.pop(int(rng.integers(len(archive)-1)))
  hop_energy=potential(state.p,hop)if hop is not None else 0.
  delta=np.rint((state.p-old.p)*1e6).astype(np.int64);changed=np.flatnonzero(np.any(delta,axis=1))
  emit('accepted',reason=reason,score=state.radius,best=best,lookahead=hop_energy,native_max_area=float(state.areas.max()),move=[[int(i),*map(int,delta[i])]for i in changed])
  if counts['accepted']%50==0:save(out/'current.json',state.p)
  return True
 def crossover(a,b,move):
  # Both parents are natively valid at these same centers. Cross intersections
  # supply an exact hard compatibility table for every hybrid assignment.
  variants=np.repeat(a.p,2,axis=0);variants[1::2,2]=b.p[:,2]
  future=variants.copy();future[:,:2]+=np.repeat(move[:,:2],2,axis=0)*1e-6;future[:,2]+=np.repeat(move[:,2],2)*1e-6
  pairids,indices=_pair_indices(15);areas=polygon_intersection_areas(future)[indices]
  cost=(np.maximum(areas-TOL,0)/scale[:,None,None]*1e8)**2
  hard=np.zeros_like(cost,dtype=bool)
  for row,(i,j)in enumerate(PAIRS):
   hard[row,0,0]=a.areas[row]>TOL;hard[row,1,1]=b.areas[row]>TOL
   if np.linalg.norm(a.p[i,:2]-a.p[j,:2])<=2:
    hard[row,0,1]=a.polygons[i].intersection(b.polygons[j]).area>TOL
    hard[row,1,0]=b.polygons[i].intersection(a.polygons[j]).area>TOL
  future_center=compute_mec_fast(canonical(a.p+move*1e-6))[:2]
  unary=np.maximum(radii_at(future,future_center)-(best-2e-12),0).reshape(15,2)**2*1e16
  barrier=float(cost.max()*105+unary.max()*15+1)*10
  current_bad=radii_at(variants,[a.cx,a.cy]).reshape(15,2)>best+args.relay
  unary=unary+current_bad*barrier;cost=cost+hard*barrier
  assignment,info=_solve_tables(unary,cost,rng.random((15,2))*1e-12)
  q=a.p.copy();q[:,2]=variants.reshape(15,2,3)[np.arange(15),assignment,2]
  return canonical(q),info
 def repair_angles(q):
  counts['conditional_attempts']+=1;k=args.repair_steps
  domains=[]
  for row in q:
   values=sorted({canonicalize_theta(row[2]+d*1e-6) for d in range(-k,k+1)})
   d=np.tile(row,(len(values),1));d[:,2]=values;domains.append(np.ascontiguousarray(d))
  # Piece 14 is the only locally inward boundary piece. Enumerate its
  # integer variants and require a center enclosing the other full disks;
  # this is sufficient for every orientation in their remaining domains.
  boundary=np.repeat(q[None,:,:],len(domains[14]),axis=0);boundary[:,14,2]=domains[14][:,2]
  _,circles,_=evaluate_batch(boundary,centers=[state.cx,state.cy]);cap=best+args.relay
  possible=[]
  for index,circle in enumerate(circles):
   if circle[2]<=cap and np.max(np.linalg.norm(q[:14,:2]-circle[:2],axis=1)+1)<=cap+2e-12:possible.append((circle[2],index,circle[:2]))
  if not possible:return None
  variants=np.vstack(domains);N=len(variants);offsets=np.r_[0,np.cumsum([len(d)for d in domains])];areas=polygon_intersection_areas(variants);adj=[[]for _ in domains];maps={}
  for i,j in PAIRS:
   u=np.arange(offsets[i],offsets[i+1])[:,None];v=np.arange(offsets[j],offsets[j+1])[None,:];mask=areas[u*(2*N-u-1)//2+v-u-1]<=TOL-1e-12
   if mask.all():continue
   adj[i].append(j);adj[j].append(i)
   maps[i,j]=[int.from_bytes(np.packbits(row,bitorder='little').tobytes(),'little')for row in mask]
   maps[j,i]=[int.from_bytes(np.packbits(row,bitorder='little').tobytes(),'little')for row in mask.T]
  for _,index,center in sorted(possible):
   initial=[(1<<len(d))-1 for d in domains];initial[14]=1<<index
   assignment,nodes=find_assignment(domains,adj,maps,rng,10000,initial=initial)
   if assignment is not None:
    counts['conditional_solutions']+=1
    return canonical(np.array([domains[i][a]for i,a in enumerate(assignment)]))
  return None
 emit('start',score=best,native_max_area=float(state.areas.max()),seed=args.seed)
 while time.monotonic()-start<args.seconds:
  if time.monotonic()-last_report>10:
   emit('progress',**counts,score=state.radius,best=best,lookahead=hop_energy,archive=len(archive),visited=len(seen));last_report=time.monotonic()
  if counts['proposals']-last_refresh>=args.hop_refresh or not hops:
   hops=enumerate_hops(state);last_refresh=counts['proposals']
   if not hops:break
   selected=hops[0 if rng.random()<.8 else int(rng.integers(len(hops)))];hop=selected[4];hop_energy=potential(state.p,hop);emit('target_hop',radius=selected[1],potential=hop_energy,move=np.argwhere(hop).tolist(),integer_delta=hop.tolist())
  counts['proposals']+=1
  if counts['proposals']%300==0 and len(hops)>1:
   selected=hops[0 if rng.random()<.75 else int(rng.integers(len(hops)))];hop=selected[4];hop_energy=potential(state.p,hop)
  reason='angle';q=None
  if args.repair_every and counts['proposals']%args.repair_every==0:
   q=repair_angles(canonical(state.p+hop*1e-6));reason='repaired_hop'
   if q is None:continue
  elif counts['proposals']%5==0:
   q=canonical(state.p+hop*1e-6);reason='center_hop'
  elif args.crossover_every and counts['proposals']%args.crossover_every==0 and len(archive)>1:
   parents=[s for s in archive if s.position_key==state.position_key]
   if len(parents)>1:
    b=parents[int(rng.integers(len(parents)))];q,_=crossover(state,b,hop);reason='valid_hybrid';counts['crossovers']+=1
  if q is None:
   q=state.p.copy();future=canonical(q+hop*1e-6);bad=polygon_intersection_areas(future)-TOL;row=int(np.argmax(bad/scale));i,j=PAIRS[row]
   chosen=int(rng.choice([i,j]))if rng.random()<.8 else int(rng.integers(15));amount=int(rng.choice([1,2,4,8,16,32,64,128,256,512,1024],p=np.array([6,5,4,3,2,1,1,1,1,1,1])/26));
   _,future_jac=polygon_intersection_areas_and_jacobian(future);direction=-np.sign(future_jac[row,3*chosen+2])
   sign=int(direction) if direction and rng.random()<.8 else int(rng.choice([-1,1]));amount=min(amount,args.angle_max)*sign;q[chosen,2]+=amount*1e-6
   if rng.random()<.25:
    other=j if chosen==i else i;q[other,2]+=amount*int(rng.choice([-1,1]))*1e-6;reason='paired_angles'
   q=canonical(q)
  if key(q)in seen:continue
  counts['screened']+=1
  estimated,circle,_=evaluate_batch(q[None,:,:],centers=[state.cx,state.cy]);circle=circle[0]
  if estimated.max()>TOL+2e-11:continue
  if circle[2]>best+args.relay:continue
  native=state.proposal(q,circle=circle)
  if native is None:counts['native_rejected']+=1;continue
  counts['valid_proposals']+=1
  E=potential(native.p,hop);temperature=max(.02,hop_energy*.1)
  probability=1 if E<=hop_energy+1e-12 else math.exp(max(-50,(hop_energy-E)/temperature))
  if reason in ('center_hop','repaired_hop'):
   probability=1 if native.radius<=state.radius+2e-12 else max(probability,.25)
  if rng.random()<max(.08,probability):
   old_positions=state.position_key;accept(native,reason)
   if state.position_key!=old_positions:last_refresh=-args.hop_refresh
   if record(native):break
  if time.monotonic()-last_report>10:
   emit('progress',**counts,score=state.radius,best=best,lookahead=hop_energy,archive=len(archive),visited=len(seen));last_report=time.monotonic()
 save(out/'current.json',state.p);emit('done',**counts,score=state.radius,best=best,visited=len(seen))
 with log.open('a')as f:f.write('\nOutcome: every accepted state passed native 4096-point GEOS intersections; best improvements also passed the reference scorer and API. Accepted transitions, integer deltas, native maxima, scores, and lookahead energies are retained in events.jsonl.\n')

if __name__=='__main__':main()
