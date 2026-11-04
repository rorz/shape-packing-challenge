"""Exact curved mirror-family sections and seam-aware batched lattice search."""
from __future__ import annotations
import os
for key in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,itertools,json,time
from pathlib import Path
import numpy as np
from fpylll import BKZ,GSO,IntegerMatrix,LLL
from breakout_geometry import evaluate_batch
from semicircle_packing.polish import AreaConstraints,checked,load,save
from semicircle_packing.geometry import canonicalize_theta
from semicircle_packing.fast_mec import compute_mec_fast
from arena import check_solution
LEADER=2.9485269238856113
FLATS=[(1,14),(4,8)]
ENDPOINT=12

def wrap(x):return np.arctan2(np.sin(x),np.cos(x))
def full(path,p):path.write_text(json.dumps([dict(zip(('x','y','theta'),map(float,q)))for q in p],indent=2)+'\n')
def canonical_once(p):
 q=np.round(p,6);q[:,:,2]=np.round(wrap(q[:,:,2]),6);return q

class Manifold:
 def __init__(self,p):
  self.p=p.copy();cx,cy,R=compute_mec_fast(p);self.p[:,:2]-=[cx,cy];self.z=np.r_[self.p.ravel(),R];self.c=AreaConstraints(self.p);g,j=self.c.evaluate(self.z);self.active=g<1e-8;self.D=np.zeros((2,46))
  for row,(i,k)in enumerate(FLATS):self.D[row,3*i+2]=1;self.D[row,3*k+2]=-1
  A=np.vstack([j[self.active],self.D]);_,s,v=np.linalg.svd(A,full_matrices=True);rank=int((s>1e-9).sum());null=v[rank:];rotation=np.zeros(46);rotation[0:45:3]=-self.p[:,1];rotation[1:45:3]=self.p[:,0];rotation[2:45:3]=1;rotation/=np.linalg.norm(rotation);_,_,mix=np.linalg.svd((null@rotation)[None,:],full_matrices=True);self.modes=mix[1:]@null;self.rotation=rotation
  if len(self.modes)!=2:raise ValueError(f'Expected two modes, got{len(self.modes)}; active{sum(self.active)} rank{rank}')
  self.raw=self.D@self.z;self.meta=dict(radius=R,active=int(self.active.sum()),rank=rank,raw_differences=self.raw.tolist(),flat_pairs=FLATS,endpoint=ENDPOINT,smallest_singular_values=s[-3:].tolist())
 def section(self,raw,alpha):
  desired=self.raw+wrap(np.array(raw)-self.raw);alpha=np.array(alpha);z=self.z+self.modes.T@alpha
  for iteration in range(35):
   gaps,jac=self.c.evaluate(z);delta=z-self.z;F=np.r_[gaps[self.active],self.D@z-desired,self.modes@delta-alpha,self.rotation@delta];JJ=np.vstack([jac[self.active],self.D,self.modes,self.rotation])
   if np.max(np.abs(F))<1e-12:break
   step=np.linalg.solve(JJ,-F);norm=np.linalg.norm(F)
   for power in range(12):
    trial=z+step*.5**power;gg,_=self.c.evaluate(trial);dd=trial-self.z;ff=np.r_[gg[self.active],self.D@trial-desired,self.modes@dd-alpha,self.rotation@dd]
    if np.linalg.norm(ff)<norm:z=trial;break
   else:break
  g,j=self.c.evaluate(z);U,_,_=np.linalg.svd(j[self.active,:45],full_matrices=True);w=U[:,-1];w*=np.sign(w.sum());meta=dict(raw=raw,alpha=alpha.tolist(),radius=float(z[-1]),violation=float(max(0,-g.min())),residual=float(np.max(np.abs(F))),iterations=iteration+1,min_stress=float(w.min()))
  return z[:-1].reshape(15,3),meta

def cells(p,signs):
 events=sorted({0.,2*np.pi,*[float((np.pi-p[i,2])%(2*np.pi))for pair in FLATS for i in pair]});result=[]
 for lo,hi in zip(events[:-1],events[1:]):
  q=wrap(p[:,2]+(lo+hi)*.5);actual=tuple(1 if q[i]-q[k]>0 else -1 for i,k in FLATS)
  if actual==tuple(signs):result.append((lo,hi))
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',default='experiments/2026-11-04/round2/topology/continuous-best.json');ap.add_argument('--output',type=Path,default=Path('experiments/2026-11-04/round3/mirror'));ap.add_argument('--seconds',type=float,default=300);ap.add_argument('--seed',type=int,default=3110401);ap.add_argument('--samples',type=int,default=384);ap.add_argument('--target',type=float,default=LEADER+2e-8);ap.add_argument('--block-size',type=int,default=20);ap.add_argument('--amplitude',type=float,default=.0008);ap.add_argument('--neighbor-branches',action='store_true');ap.add_argument('--smoke',action='store_true');ap.add_argument('--bootstrap',action='store_true');ap.add_argument('--boundary-bias',type=float,default=5.);ap.add_argument('--near-valid');ap.add_argument('--focus-only',action='store_true');ap.add_argument('--random-sections',action='store_true');ap.add_argument('--sigma',type=float,default=.0004);ap.add_argument('--native-dimension',type=int,choices=[43,45],default=43);ap.add_argument('--regularization',type=float,default=1e-5);ap.add_argument('--accept-seam-crossings',action='store_true');ap.add_argument('--raw-first');ap.add_argument('--raw-second');ap.add_argument('--native-flat-weight',type=float,default=.1);ap.add_argument('--branch-map');ap.add_argument('--max-branch-cost',type=float,default=20.);ap.add_argument('--frontier',action='store_true');ap.add_argument('--frontier-bin-width',type=float,default=1.);ap.add_argument('--frontier-guard',type=float,default=100.);ap.add_argument('--slack-patterns');ap.add_argument('--slack-fraction',type=float,default=.8);ap.add_argument('--slack-floor',type=float,default=.02);ap.add_argument('--slack-strength',type=float,default=0.);ap.add_argument('--slack-include-near',action='store_true');ap.add_argument('--phase-seed',type=int);ap.add_argument('--slack-quality-temperature',type=float,default=0.);a=ap.parse_args()
 if a.random_sections:return random_campaign(a)
 a.output.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(a.seed);started=time.monotonic();model=Manifold(load(a.input));(a.output/'manifold.json').write_text(json.dumps(model.meta,indent=2)+'\n');np.savez(a.output/'manifold.npz',origin=model.z,modes=model.modes,active=model.active,rotation=model.rotation);source_hash=hashlib.sha256(Path(a.input).read_bytes()).hexdigest()
 # Match the scalar canonicalizer exactly, retaining submitted raw angles separately.
 test=np.zeros((1,15,3));test[0,:,2]=np.linspace(-4,4,15);test[0,0,2]=-3.141593;test[0,1,2]=3.141593
 actual=canonical_once(test)[0,:,2];reference=np.array([canonicalize_theta(t)for t in test[0,:,2]])
 if not np.array_equal(actual,reference):raise ValueError('One-pass canonicalization audit failed')
 audit=dict(source_sha256=source_hash,canonical_once_exact=True,double_wrap_changes_negative_pi=canonicalize_theta(canonicalize_theta(-3.141593))!=canonicalize_theta(-3.141593),frame_strategy='four raw-sign cells, random submicro translations, optional endpoint grid alignment');(a.output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 near=None;near_angle=None
 if a.near_valid:
  near=canonical_once(load(a.near_valid)[None,:,:])[0];x=model.p[:,:2]-model.p[:,:2].mean(0);y=near[:,:2]-near[:,:2].mean(0);cross=np.sum((y[:,0]+1j*y[:,1])*np.conj(x[:,0]+1j*x[:,1]))+.2*np.sum(np.exp(1j*near[:,2])*np.exp(-1j*model.p[:,2]));near_angle=float(np.angle(cross)%(2*np.pi))
 initial=checked(load('solution.json'));best=float(initial.score);mirror_best=float('inf');best_violation=float('inf');best_loss=float('inf');counts=dict(frames=0,proposals=0,branch_filtered=0,checks=0,valid=0,api=0,invalid_sections=0,fallbacks=0);records=[];seed_cache={};last_report=started
 raw_options=[[(3.141570,3.141593),(3.141570,-3.141592),(-3.141616,3.141593),(-3.141616,-3.141592)]]
 branches=list(raw_options[0]);neighbors=[0]if not a.neighbor_branches else[0,-1,1]
 branches=[(x+dx*1e-6,y+dy*1e-6)for dx,dy in itertools.product(neighbors,repeat=2)for x,y in branches]
 if a.focus_only:
  if near is None:raise ValueError('focus-only requires near-valid')
  branches=[tuple(float(np.rint((near[i,2]-near[k,2])*1e6)/1e6)for i,k in FLATS)]
 A=a.amplitude;alphas=[(0,0),(A,0),(-A,0),(0,A),(0,-A),(A*.7,A*.7),(-A*.7,A*.7),(A*.7,-A*.7),(-A*.7,-A*.7),(A*1.5,0),(-A*1.5,0),(0,A*1.5),(0,-A*1.5)]
 schedule=[(branch,alpha)for alpha in alphas for branch in branches];schedule_index=0
 while time.monotonic()-started<a.seconds:
  branch,alpha=schedule[schedule_index%len(schedule)];schedule_index+=1;key=(branch,alpha)
  if key not in seed_cache:
   p,metadata=model.section(branch,alpha);index=len(seed_cache);path=a.output/f'section-{index:03d}.json';full(path,p);metadata.update(index=index,path=str(path));seed_cache[key]=(p,metadata);records.append(metadata);(a.output/'sections.json').write_text(json.dumps(records,indent=2)+'\n')
  p,metadata=seed_cache[key]
  if metadata['violation']>2e-9 or metadata['min_stress']<=0 or metadata['radius']>a.target-1e-7:counts['invalid_sections']+=1;continue
  signs=[1 if d>0 else -1 for d in branch];intervals=cells(p,signs)
  if not intervals:counts['invalid_sections']+=1;continue
  widths=np.array([hi-lo for lo,hi in intervals]);lo,hi=intervals[int(rng.choice(len(intervals),p=widths/widths.sum()))];margin=min(.001,(hi-lo)*.1);angle=rng.uniform(lo+margin,hi-margin);endpoint_mode='free'
  if near is not None and(a.focus_only or counts['frames']%3!=2):
   focused=float((near_angle+rng.normal(0,rng.choice([1e-7,1e-6,1e-5,1e-4])))%(2*np.pi))
   if any(l<focused<h for l,h in intervals):angle=focused;endpoint_mode='near_valid'
  if counts['frames']%5==0 and not(a.focus_only and near is not None):
   endpoint_target=rng.choice([-3.141592,3.141592]);candidate_angle=(endpoint_target-p[ENDPOINT,2])%(2*np.pi)
   if any(l<candidate_angle<h for l,h in intervals):angle=candidate_angle;endpoint_mode='seam'
  elif counts['frames']%2:
   endpoint_angle=float(wrap(p[ENDPOINT,2]+angle));angle+=round(endpoint_angle,6)-endpoint_angle;endpoint_mode='aligned'
  cs,sn=np.cos(angle),np.sin(angle);posed=p.copy();posed[:,:2]=posed[:,:2]@np.array([[cs,sn],[-sn,cs]]);posed[:,2]=wrap(posed[:,2]+angle);c=AreaConstraints(posed);R=metadata['radius']
  if 'frame_path'not in metadata:
   path=a.output/f'frame-{metadata["index"]:03d}.json';full(path,posed);metadata.update(frame_path=str(path),frame_angle=float(angle),raw_frame_differences=[float(posed[i,2]-posed[k,2])for i,k in FLATS],endpoint_mode=endpoint_mode);(a.output/'sections.json').write_text(json.dumps(records,indent=2)+'\n')
  g,j=c.evaluate(np.r_[posed.ravel(),R]);active=model.active;J=j[active,:45];U,S,_=np.linalg.svd(J,full_matrices=True);w=U[:,-1];w*=np.sign(w.sum())
  if w.min()<=0:counts['invalid_sections']+=1;continue
  target=min(a.target,(mirror_best if a.bootstrap else best)-2e-12);g,_=c.evaluate(np.r_[posed.ravel(),target]);g0=g[active];budget=float(w@g0)
  if budget<=0:continue
  shift=rng.uniform(-.5e-6,.5e-6,2);anchor=posed.copy();anchor[:,:2]+=shift;anchor=np.round(anchor,6);local_anchor=anchor.copy();local_anchor[:,:2]-=shift;error=(local_anchor-posed).ravel()/1e-6;b0=-g0/1e-6-J@error
  flat=np.zeros((2,45))
  for row,(i,k)in enumerate(FLATS):flat[row,3*i+2]=.1;flat[row,3*k+2]=-.1
  C=-w[:,None]*np.column_stack([J[:,0::3].sum(1),J[:,1::3].sum(1)]);Q,_=np.linalg.qr(C);P=np.eye(len(w))-Q@Q.T;W=np.vstack([P@(w[:,None]*J),flat]);regularization=1e-5 if counts['frames']%3 else 3e-5;scale=10000000;identity=round(scale*regularization);B=IntegerMatrix.from_matrix(np.column_stack([np.eye(45)*identity,np.rint(W.T*scale)]).astype(np.int64).tolist());LLL.reduction(B,delta=.99)
  if a.block_size:BKZ.reduction(B,BKZ.Param(block_size=a.block_size,max_loops=2))
  M=GSO.Mat(B,update=True);prior=np.zeros(45)
  if near is not None:
   known=near.copy();known[:,:2]+=np.round(np.mean(anchor[:,:2]-known[:,:2],axis=0),6);offset=known-anchor;offset[:,2]=wrap(offset[:,2]);prior=offset.ravel()/1e-6
  moves=np.rint(np.array([list(row)for row in B],dtype=float)[:,:45]/identity).astype(np.int64);raw_candidates=[];movement=[10,50,150][counts['frames']%3];concentration=[10.,1.,.3][(counts['frames']//3)%3];distribution=np.full(len(w),concentration);distribution[np.flatnonzero(active)>=105]*=a.boundary_bias
  for attempt in range(a.samples):
   allocation=rng.dirichlet(distribution);desired=budget*allocation/w/1e-6;contact=np.r_[P@(w*(desired+b0)),-flat@error];wanted=prior+rng.normal(0,movement,45);integer_target=np.rint(np.r_[wanted*identity,contact*scale]).astype(np.int64).tolist();coef=M.babai(integer_target);delta=np.array(coef,dtype=np.int64)@moves;raw_candidates.append(anchor+delta.reshape(15,3)*1e-6)
  raw=np.round(np.array(raw_candidates),6);actual=canonical_once(raw);branch_units=np.rint(np.array(branch)*1e6).astype(np.int64);mask=np.ones(len(actual),dtype=bool)
  for row,(i,k)in enumerate(FLATS):mask&=np.rint((actual[:,i,2]-actual[:,k,2])*1e6).astype(np.int64)==branch_units[row]
  counts['branch_filtered']+=int((~mask).sum());raw=raw[mask];actual=actual[mask];counts['frames']+=1
  if not len(actual):continue
  areas,circles,ok=evaluate_batch(actual,centers=shift);counts['proposals']+=len(actual);counts['fallbacks']+=int((ok==0).sum());negative=np.maximum((areas-1e-6)/c.scale,0);radius_violation=np.maximum(circles[:,2]-target,0);loss=np.sum(negative**2,axis=1)+3*radius_violation**2;violation=np.maximum(negative.max(axis=1),radius_violation);index=int(np.argmin(loss))
  if loss[index]<best_loss:
   best_loss=float(loss[index]);best_violation=float(violation[index]);save(a.output/'nearest-raw.json',raw[index]);np.savez(a.output/'nearest-state.npz',placements=raw[index],geometry=actual[index],origin=posed,moves=moves.reshape(45,15,3),target=target,center=circles[index,:2],branch=branch,alpha=alpha,loss=best_loss,violation=best_violation);print(json.dumps(dict(event='nearest',section=metadata['index'],branch=branch,alpha=alpha,endpoint_mode=endpoint_mode,radius=float(circles[index,2]),violation=best_violation,loss=best_loss,seconds=time.monotonic()-started,**counts)),flush=True)
  validation_cap=mirror_best if a.bootstrap else best
  promising=np.flatnonzero((areas.max(axis=1)<=1e-6+1e-11)&(circles[:,2]<validation_cap-1e-13))
  for index in promising:
   counts['checks']+=1;v=checked(raw[index])
   if not v.valid:continue
   counts['valid']+=1
   if v.score<mirror_best:
    mirror_best=v.score;save(a.output/'mirror-valid-best-raw.json',raw[index]);np.savez(a.output/'mirror-valid-state.npz',placements=raw[index],geometry=actual[index],origin=posed,moves=moves.reshape(45,15,3),target=target,center=circles[index,:2],branch=branch,alpha=alpha);print(json.dumps(dict(event='mirror_valid',score=mirror_best,section=metadata['index'],branch=branch,alpha=alpha,seconds=time.monotonic()-started)),flush=True)
   if v.score<best:
    save(a.output/'valid-best-raw.json',raw[index]);response=check_solution(json.loads((a.output/'valid-best-raw.json').read_text()));counts['api']+=1;(a.output/f'api-{counts["api"]:03d}.json').write_text(json.dumps(response,indent=2)+'\n');print(json.dumps(dict(event='api',section=metadata['index'],branch=branch,alpha=alpha,**response)),flush=True)
    if response.get('valid')and response['score']<best:best=response['score']
    if response.get('valid')and response['score']<LEADER:break
  if best<LEADER:break
  if time.monotonic()-last_report>10:
   print(json.dumps(dict(event='progress',sections=len(seed_cache),best_score=best,mirror_best=mirror_best if np.isfinite(mirror_best)else None,best_violation=best_violation,best_loss=best_loss,seconds=time.monotonic()-started,**counts)),flush=True);last_report=time.monotonic()
  if a.smoke and counts['frames']>=8:break
 result=dict(event='done',best_score=best,mirror_best=mirror_best if np.isfinite(mirror_best)else None,goal=LEADER,goal_met=best<LEADER,best_violation=best_violation,best_loss=best_loss,sections=len(seed_cache),seconds=time.monotonic()-started,**counts);(a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

def random_campaign(a):
 from types import SimpleNamespace
 from collections import Counter
 from experiment_batch_lattice import NativeBabai,Frame as FreeFrame
 from experiment_tied_batch import Frame as TiedFrame,tied_transform
 from experiment_mirror_frontier import OverlapFrontier
 from experiment_mirror_slack import SlackMixture
 a.output.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(a.seed);start=time.monotonic();model=Manifold(load(a.input));decoder=NativeBabai();decoder.reset();best=checked(load('solution.json')).score;counts=dict(frames=0,sections=0,rejected_sections=0,proposals=0,unique=0,duplicates=0,branch_rejected=0,ambiguous=0,checks=0,valid=0,api=0,elites=0,fallbacks=0);nearest=float('inf');elites={};serial=0;last=start;source_hash=hashlib.sha256(Path(a.input).read_bytes()).hexdigest();(a.output/'manifold.json').write_text(json.dumps(model.meta|dict(source_sha256=source_hash,sigma=a.sigma,dimension=a.native_dimension),indent=2)+'\n');np.savez(a.output/'manifold.npz',origin=model.z,modes=model.modes,active=model.active,rotation=model.rotation)
 def emit(event,**data):
  row=dict(event=event,seconds=time.monotonic()-start);row.update(data);print(json.dumps(row),flush=True)
  with(a.output/'events.jsonl').open('a')as stream:stream.write(json.dumps(row)+'\n')
 first=[3.141570,-3.141616]if a.raw_first is None else list(map(float,a.raw_first.split(',')));second=[3.141593,-3.141592]if a.raw_second is None else list(map(float,a.raw_second.split(',')));branches=list(itertools.product(first,second));branch_probabilities=None
 if a.branch_map:
  branch_rows=[r for r in json.loads(Path(a.branch_map).read_text())if r.get('feasible_positive_tied_stress',r.get('feasible_positive_stress',False))and r['cost_nm']<a.max_branch_cost];branches=[(r['first_raw'],r['second_raw'])for r in branch_rows];branch_probabilities=np.array([(max(a.target-r['radius'],0)/(a.target-model.z[-1]))**40 for r in branch_rows]);branch_probabilities/=branch_probabilities.sum();(a.output/'branch-selection.json').write_text(json.dumps(dict(map=a.branch_map,maximum_cost_nm=a.max_branch_cost,branches=len(branches),weight='remaining radius slack to power40'),indent=2)+'\n')
 phase_rng=rng if a.phase_seed is None else np.random.default_rng(a.phase_seed)
 frontier=OverlapFrontier(a.output/'frontier',a.frontier_bin_width,a.frontier_guard)if a.frontier else None
 slack=SlackMixture(a.slack_patterns,fraction=a.slack_fraction,floor=a.slack_floor,strength=a.slack_strength,include_near=a.slack_include_near,quality_temperature_nm=a.slack_quality_temperature)if a.slack_patterns else None
 def validate_deferred():
  nonlocal best
  if frontier is None:return
  while frontier.deferred_valid:
   point=frontier.deferred_valid.pop(0);raw=point['raw'];validation=checked(raw);counts['checks']+=1
   if not validation.valid:continue
   counts['valid']+=1
   if validation.score<best-2e-12:
    counts['api']+=1;api_input=a.output/f'api-{counts["api"]:03d}-input.json';save(api_input,raw);response=check_solution(json.loads(api_input.read_text()));response['input_sha256']=hashlib.sha256(api_input.read_bytes()).hexdigest();(a.output/f'api-{counts["api"]:03d}.json').write_text(json.dumps(response,indent=2)+'\n');emit('api',source='deferred-frontier',**response)
    if response.get('valid')and response['score']<best:best=response['score'];save(a.output/'best-raw.json',raw)
 submitted_hist=Counter();actual_hist=Counter();source_hist=Counter()
 def histogram_report():
  convert=lambda h:[dict(differences_micro=list(key),count=int(count))for key,count in sorted(h.items())]
  return dict(scope='all decoded proposals before deduplication and branch filtering',submitted_raw=convert(submitted_hist),canonical_geometry_raw=convert(actual_hist),source=convert(source_hist))
 while time.monotonic()-start<a.seconds:
  branch=branches[counts['sections']%len(branches)]if branch_probabilities is None else branches[int(phase_rng.choice(len(branches),p=branch_probabilities))];alpha=phase_rng.normal(0,a.sigma,2);p,meta=model.section(branch,alpha);counts['sections']+=1
  if a.native_dimension==43:
   _,T=tied_transform(p,FLATS);_,j=model.c.evaluate(np.r_[p.ravel(),meta['radius']]);U,_,_=np.linalg.svd(j[model.active,:45]@T,full_matrices=True);w=U[:,-1];w*=np.sign(w.sum());meta['min_full_stress']=meta['min_stress'];meta['min_stress']=float(w.min());meta['stress_dimension']=43
  if meta['violation']>2e-9 or meta['min_stress']<=0 or meta['radius']>a.target-8e-8:counts['rejected_sections']+=1;continue
  intervals=cells(p,[1 if value>0 else-1 for value in branch]);widths=np.array([hi-lo for lo,hi in intervals]);lo,hi=intervals[int(phase_rng.choice(len(intervals),p=widths/widths.sum()))];angle=phase_rng.uniform(lo,hi);endpoint_mode='free'
  if counts['frames']%2:
   t=float(wrap(p[ENDPOINT,2]+angle));angle+=round(t,6)-t;endpoint_mode='aligned'
  cs,sn=np.cos(angle),np.sin(angle);posed=p.copy();posed[:,:2]=p[:,:2]@np.array([[cs,sn],[-sn,cs]]);posed[:,2]=wrap(posed[:,2]+angle)
  settings=SimpleNamespace(target=a.target,regularization=a.regularization,flat_weight=a.native_flat_weight,block_size=a.block_size,concentration=[10.,1.,.3][counts['frames']%3],boundary_bias=a.boundary_bias,movement=[10.,50.,150.][counts['frames']%3])
  try:frame=TiedFrame(posed,meta['radius'],FLATS,settings)if a.native_dimension==43 else FreeFrame(posed,meta['radius'],0.,settings)
  except ValueError:counts['rejected_sections']+=1;continue
  # Change the grid translation phase without moving the circle-centered model.
  anchor=frame.anchor_units.reshape(15,3).copy();new_anchor=np.rint(posed*1e6).astype(np.int64);new_anchor[:,:2]=np.rint((posed[:,:2]+phase_rng.uniform(-.5e-6,.5e-6,2))*1e6).astype(np.int64)
  for (i,k),d in zip(FLATS,np.rint(np.array(branch)*1e6).astype(np.int64)):new_anchor[i,2]=new_anchor[k,2]+d
  constraint=frame.c if a.native_dimension==43 else frame.constraint;_,jac=constraint.evaluate(np.r_[posed.ravel(),meta['radius']]);J=jac[frame.active,:45];frame.b0-=J@(new_anchor-anchor).ravel();frame.anchor_units=new_anchor if a.native_dimension==43 else new_anchor.ravel()
  if a.native_dimension==45:
   frame.anchor=new_anchor*1e-6;frame.flat_offset-=frame.flat@(new_anchor-anchor).ravel()
  if slack is not None:actual,units,coef,ambient,ambiguous=slack.sample(frame,rng,a.samples,settings,decoder)
  elif a.native_dimension==43:actual,units,coef,ambient,ambiguous=frame.sample(rng,a.samples,settings,decoder)
  else:
   ambient,desired,gso=frame.targets(rng,a.samples,settings);decoded=decoder.decode(gso,frame.mu,frame.moves,frame.anchor_units);ambiguous=frame.patch(ambient,decoded);actual,units,coef,_=decoded
  counts['ambiguous']+=ambiguous;counts['proposals']+=len(actual);counts['frames']+=1
  anchor_theta=frame.anchor_units.reshape(15,3)[:,2];difference_moves=np.column_stack([frame.moves[:,3*i+2]-frame.moves[:,3*k+2]for i,k in FLATS]);submitted=coef@difference_moves+np.array([anchor_theta[i]-anchor_theta[k]for i,k in FLATS]);canonical_differences=np.column_stack([units[:,3*i+2]-units[:,3*k+2]for i,k in FLATS])
  for values,hist in [(submitted,submitted_hist),(canonical_differences,actual_hist)]:
   keys,frequencies=np.unique(values,axis=0,return_counts=True)
   for key,frequency in zip(keys,frequencies):hist[tuple(map(int,key))]+=int(frequency)
  source_hist[tuple(int(np.rint(d*1e6))for d in branch)]+=len(actual)
  if counts['frames']<=4:
   count=min(128,len(coef));expected=np.array([frame.M.babai(v.tolist())for v in ambient[:count]]);mismatch=int(np.any(expected!=coef[:count],axis=1).sum());emit('decoder_audit',frame=counts['frames'],tested=count,mismatches=mismatch)
   if mismatch:raise AssertionError('Native Babai mismatch')
  mask=np.ones(len(actual),dtype=bool)
  for(i,k),d in zip(FLATS,np.rint(np.array(branch)*1e6).astype(np.int64)):mask&=(units[:,3*i+2]-units[:,3*k+2])==d
  counts.setdefault('branch_crossings',0);counts['branch_crossings']+=int((~mask).sum())
  if not a.accept_seam_crossings:counts['branch_rejected']+=int((~mask).sum())
  permitted=np.ones(len(actual),dtype=bool)if a.accept_seam_crossings else mask;indices=np.flatnonzero(permitted);selected,info=decoder.unique(units[permitted]);counts['duplicates']+=int(info[0]);keep=np.zeros(len(actual),dtype=bool);keep[indices[selected]]=True
  actual=actual[keep];units=units[keep];coef=coef[keep]
  if not len(actual):continue
  areas,circles,ok=evaluate_batch(actual,centers=[0.,0.]);counts['unique']+=len(actual);counts['fallbacks']+=int((ok==0).sum());violation=np.maximum(np.max((areas-1e-6)/constraint.scale,axis=1),circles[:,2]-a.target);index=int(np.argmin(violation))
  def raw_at(index):
   if a.native_dimension==43:return frame.raw(coef[index])
   delta=(coef[index]@frame.moves).reshape(15,3);delta[:,:2]-=delta[0,:2].copy();return(frame.anchor_units.reshape(15,3)+delta)*1e-6
  if violation[index]<nearest:
   nearest=float(violation[index]);raw=raw_at(index);save(a.output/'nearest-raw.json',raw);np.savez(a.output/'nearest-state.npz',placements=raw,geometry=actual[index],origin=posed,moves=frame.moves.reshape(a.native_dimension,15,3),center=circles[index,:2],target=a.target,alpha=alpha,branch=branch);emit('nearest',violation=nearest,radius=float(circles[index,2]),alpha=alpha.tolist(),branch=branch,counts=counts.copy())
  frontier_valid=[]
  if frontier is not None:
   def state_at(index):return dict(origin=posed,moves=frame.moves.reshape(a.native_dimension,15,3),anchor_units=frame.anchor_units.reshape(15,3),coefficients=coef[index],center=circles[index,:2],target=a.target,alpha=alpha,branch=branch,active=frame.active,weights=frame.weights)
   frontier_valid=frontier.consider(actual,areas,circles,constraint.scale,raw_at,state_at);validate_deferred()
  cap=best+1e-7;eligible=np.union1d(np.flatnonzero((areas.max(axis=1)<=1e-6+1e-11)&(circles[:,2]<cap)),frontier_valid).astype(int);eligible=eligible[np.argsort(circles[eligible,2])]
  for index in eligible:
   xy=units[index].reshape(15,3)[:,:2];key=tuple((xy-xy[0]).ravel());estimate=float(circles[index,2])
   if key in elites and estimate>=elites[key]['score']-2e-12:continue
   if key not in elites and len(elites)>=24 and estimate>=max(r['score']for r in elites.values())-2e-12:continue
   raw=raw_at(index);validation=checked(raw);counts['checks']+=1
   if not validation.valid:continue
   counts['valid']+=1
   if key not in elites or validation.score<elites[key]['score']-2e-12:
    if len(elites)>=24 and key not in elites:worst=max(elites,key=lambda k:elites[k]['score']);elites.pop(worst)
    name=f'elite-{serial:04d}';serial+=1;path=a.output/(name+'.json');save(path,raw);np.savez(a.output/(name+'.npz'),placements=raw,geometry=actual[index],origin=posed,moves=frame.moves.reshape(a.native_dimension,15,3),center=circles[index,:2],target=a.target,alpha=alpha,branch=branch,coefficients=coef[index]);elites[key]=dict(score=validation.score,path=str(path),sidecar=str(a.output/(name+'.npz')),alpha=alpha.tolist(),branch=branch,endpoint_mode=endpoint_mode);counts['elites']+=1;(a.output/'elites.json').write_text(json.dumps(sorted(elites.values(),key=lambda r:r['score']),indent=2)+'\n');emit('elite',**elites[key])
   if validation.score<best-2e-12:
    save(a.output/'candidate-raw.json',raw);counts['api']+=1;api_input=a.output/f'api-{counts["api"]:03d}-input.json';save(api_input,raw);response=check_solution(json.loads(api_input.read_text()));response['input_sha256']=hashlib.sha256(api_input.read_bytes()).hexdigest();(a.output/f'api-{counts["api"]:03d}.json').write_text(json.dumps(response,indent=2)+'\n');emit('api',alpha=alpha.tolist(),branch=branch,**response)
    if response.get('valid')and response['score']<best:best=response['score'];save(a.output/'best-raw.json',raw)
    if best<LEADER:break
  if best<LEADER:break
  meta.update(branch=branch,angle=float(angle),endpoint_mode=endpoint_mode,dimension=a.native_dimension,counts=counts.copy())
  with(a.output/'sections.jsonl').open('a')as stream:stream.write(json.dumps(meta)+'\n')
  if counts['frames']<=64:full(a.output/f'frame-{counts["frames"]:04d}.json',posed)
  if time.monotonic()-last>10:
   if frontier is not None:frontier.flush()
   if slack is not None:(a.output/'slack-mixture.json').write_text(json.dumps(slack.report(),indent=2)+'\n')
   (a.output/'raw-differences.json').write_text(json.dumps(histogram_report(),indent=2)+'\n');emit('progress',best=best,nearest=nearest,counts=counts.copy(),unique_per_second=counts['unique']/(time.monotonic()-start),frontier=None if frontier is None else frontier.counts);last=time.monotonic()
  if a.smoke and counts['frames']>=8:break
 if frontier is not None:frontier.flush();validate_deferred()
 if slack is not None:(a.output/'slack-mixture.json').write_text(json.dumps(slack.report(),indent=2)+'\n')
 (a.output/'raw-differences.json').write_text(json.dumps(histogram_report(),indent=2)+'\n');result=dict(best=best,nearest=nearest,counts=counts,seconds=time.monotonic()-start,goal_met=best<LEADER,frontier=None if frontier is None else frontier.counts);(a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');emit('done',**result)

if __name__=='__main__':main()
