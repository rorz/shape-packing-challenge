"""Global-frame LLL sampling with native batched Babai and full geometry."""
from __future__ import annotations
import os
for name in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[name]='1'
import argparse,ctypes,hashlib,json,shlex,sys,tempfile,time,subprocess
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from fpylll import BKZ,GSO,IntegerMatrix,LLL
from experiment_lll import flat_contact_pairs
from breakout_geometry import evaluate_batch
from semicircle_packing.polish import AreaConstraints,checked,load,polish,save
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.geometry import canonicalize_theta
from arena import check_solution


class NativeBabai:
 def __init__(self):
  source=Path(__file__).with_name('_batch_lattice.cpp');digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16];target=Path(tempfile.gettempdir())/f'packing-batch-lattice-{digest}.so'
  if not target.exists():
   temporary=target.with_suffix(f'.{os.getpid()}.tmp.so');subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(temporary)],check=True,capture_output=True);temporary.replace(target)
  self.lib=ctypes.CDLL(str(target));double=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');integer=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS');flag=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
  self.lib.packing_batch_babai.argtypes=[ctypes.c_int,ctypes.c_int,double,double,integer,integer,integer,integer,double,flag];self.lib.packing_batch_babai.restype=None
  self.lib.packing_batch_unique.argtypes=[ctypes.c_int,integer,ctypes.c_int64,flag,integer];self.lib.packing_batch_unique.restype=None;self.lib.packing_batch_reset.argtypes=[];self.lib.packing_batch_reset.restype=None
 def decode(self,targets,mu,moves,anchor):
  targets=np.ascontiguousarray(targets,dtype=float);mu=np.ascontiguousarray(mu,dtype=float);moves=np.ascontiguousarray(moves,dtype=np.int64);anchor=np.ascontiguousarray(anchor,dtype=np.int64).reshape(45)
  count,n=targets.shape;coeff=np.empty((count,n),dtype=np.int64);units=np.empty((count,45),dtype=np.int64);placements=np.empty((count,15,3));ambiguous=np.empty(count,dtype=np.int32)
  self.lib.packing_batch_babai(count,n,targets,mu,moves,anchor,coeff,units,placements,ambiguous)
  return placements,units,coeff,ambiguous
 def unique(self,units,limit=1000000):
  units=np.ascontiguousarray(units,dtype=np.int64).reshape(-1,45);keep=np.empty(len(units),dtype=np.int32);info=np.empty(3,dtype=np.int64);self.lib.packing_batch_unique(len(units),units,limit,keep,info);return keep.astype(bool),info
 def reset(self):self.lib.packing_batch_reset()


class EliteArchive:
 """Bounded reference-valid starts, keeping genuinely different xy states."""
 def __init__(self,out,count,cap,emit):
  self.path=out/'elites';self.path.mkdir(exist_ok=True);self.count=count;self.cap=cap;self.emit=emit;self.entries={};self.serial=0
 @staticmethod
 def key(units):
  xy=units.reshape(15,3)[:,:2];return tuple((xy-xy[0]).ravel().tolist())
 def wants(self,units,score):
  if not self.count or score>self.cap:return False
  key=self.key(units)
  if key in self.entries:return score<self.entries[key]['score']-2e-12
  return len(self.entries)<self.count or score<max(v['score']for v in self.entries.values())-2e-12
 def add(self,p,units,score,frame,coeff,ambient,gso,desired,circle,args,index):
  if not self.wants(units,score):return False
  key=self.key(units);evicted=None
  if key in self.entries:evicted=self.entries.pop(key)
  elif len(self.entries)>=self.count:
   worst=max(self.entries,key=lambda k:self.entries[k]['score']);evicted=self.entries.pop(worst)
  if evicted:
   for name in ('file','sidecar'):(self.path/evicted[name]).unlink(missing_ok=True)
  name=f'elite-{self.serial:05d}';self.serial+=1;save(self.path/(name+'.json'),p)
  np.savez_compressed(self.path/(name+'.npz'),placement=p,raw_placement=p,canonical_placement=units.reshape(15,3)*1e-6,units=units,continuous_frame=frame.p,anchor_units=frame.anchor_units,basis=frame.reduced,moves=frame.moves,coefficients=coeff,ambient=ambient,gso_target=gso,desired_gaps=desired,J=frame.J,g0=frame.g0,weights=frame.weights,active=frame.active,circle=circle)
  arguments={key:str(value)if isinstance(value,Path)else value for key,value in vars(args).items()}
  record=dict(score=score,file=name+'.json',sidecar=name+'.npz',frame=index,angle=frame.angle,seed=args.seed,input=args.input,flat_pairs=frame.flat_pairs,arguments=arguments)
  self.entries[key]=record;records=sorted(self.entries.values(),key=lambda v:v['score']);(self.path/'manifest.json').write_text(json.dumps(dict(cap=self.cap,maximum_count=self.count,entries=records),indent=2)+'\n')
  self.emit('elite',**record,archive_size=len(records),evicted_score=None if evicted is None else evicted['score'])
  return True


class Frame:
 def __init__(self,original,radius,angle,args):
  self.angle=angle;cs,sn=np.cos(angle),np.sin(angle);p=original.copy();p[:,:2]=p[:,:2]@np.array([[cs,sn],[-sn,cs]]);p[:,2]=np.arctan2(np.sin(p[:,2]+angle),np.cos(p[:,2]+angle));self.p=p
  c=AreaConstraints(p);g,j=c.evaluate(np.r_[p.ravel(),radius]);active=g<1e-7;J=j[active,:45];U,S,_=np.linalg.svd(J,full_matrices=True);w=U[:,-1];w*=np.sign(w.sum())
  if w.min()<=0:raise ValueError('Nonpositive contact stress')
  g,_=c.evaluate(np.r_[p.ravel(),args.target]);self.g0=g[active];self.weights=w;self.budget=float(w@self.g0);self.J=J;self.active=active;self.constraint=c
  self.anchor=np.round(p,6);self.anchor_units=np.rint(self.anchor.ravel()*1e6).astype(np.int64);offset=(self.anchor-p).ravel()/1e-6;self.b0=-self.g0/1e-6-J@offset
  self.flat_pairs=flat_contact_pairs(p,c,g);flat=np.zeros((len(self.flat_pairs),45))
  for row,(i,k)in enumerate(self.flat_pairs):flat[row,3*i+2]=args.flat_weight;flat[row,3*k+2]=-args.flat_weight
  center=-np.column_stack([J[:,0::3].sum(1),J[:,1::3].sum(1)]);Q,_=np.linalg.qr(w[:,None]*center);self.projection=np.eye(len(w))-Q@Q.T;self.center_inverse=np.linalg.pinv(w[:,None]*center)
  self.flat=flat
  metric=np.vstack([self.projection@(w[:,None]*J),flat]);self.scale=10000000;self.identity=max(1,round(self.scale*args.regularization));self.flat_offset=-flat@offset
  basis=np.c_[np.eye(45)*self.identity,np.rint(metric.T*self.scale)].astype(np.int64);B=IntegerMatrix.from_matrix(basis.tolist());LLL.reduction(B,delta=.99)
  if args.block_size:BKZ.reduction(B,BKZ.Param(block_size=args.block_size,max_loops=2))
  self.B=B;self.M=GSO.Mat(B,update=True);self.reduced=np.array([list(row)for row in B],dtype=np.int64);self.moves=np.rint(self.reduced[:,:45]/self.identity).astype(np.int64)
  self.mu=np.ascontiguousarray([[self.M.get_mu(i,k)if k<i else 0 for k in range(45)]for i in range(45)])
  r=np.array([self.M.get_r(i,i)for i in range(45)]);star=np.array([self.M.to_canonical(np.eye(45)[i])for i in range(45)])
  self.transform=np.ascontiguousarray((star/r[:,None]).T)
 def quarter_turn(self,turn):
  # Signed coordinate permutations preserve the embedding Gram matrix.
  # The reference GSO stays in latent base-frame coordinates; isotropic
  # Gaussian wanted vectors have the same law after the physical permutation.
  other=object.__new__(Frame);other.__dict__=self.__dict__.copy();other.angle=self.angle+turn*np.pi/2
  other.p=self.p.copy();other.J=self.J.copy();other.moves=self.moves.copy()
  for _ in range(turn):
   other.p[:,:2]=other.p[:,:2][:,[1,0]]*[-1,1]
   reshaped=other.J.reshape(len(other.J),15,3);reshaped[:,:,:2]=reshaped[:,:,:2][:,:,[1,0]]*[-1,1]
   moved=other.moves.reshape(45,15,3);moved[:,:,:2]=moved[:,:,:2][:,:,[1,0]]*[-1,1]
  other.p[:,2]=np.arctan2(np.sin(self.p[:,2]+turn*np.pi/2),np.cos(self.p[:,2]+turn*np.pi/2))
  other.anchor=np.round(other.p,6);other.anchor_units=np.rint(other.anchor.ravel()*1e6).astype(np.int64)
  offset=(other.anchor-other.p).ravel()/1e-6
  assert np.abs(offset.reshape(15,3)[:,2]).max()<=.50000001
  other.b0=-other.g0/1e-6-other.J@offset;other.flat_offset=-other.flat@offset
  return other
 def targets(self,rng,count,args):
  alpha=np.full(len(self.weights),args.concentration);alpha[np.flatnonzero(self.active)>=105]*=args.boundary_bias;allocation=rng.dirichlet(alpha,size=count);desired=self.budget*allocation/self.weights[None,:]/1e-6
  contact=(self.weights[None,:]*(desired+self.b0))@self.projection.T
  wanted=rng.normal(0,args.movement,(count,45));ambient=np.rint(np.c_[wanted*self.identity,contact*self.scale,np.broadcast_to(self.flat_offset,(count,len(self.flat_offset)))*self.scale]).astype(np.int64)
  return ambient,desired,np.ascontiguousarray(ambient@self.transform)
 def patch(self,ambient,decoded):
  placements,units,coeff,ambiguous=decoded
  for k in np.flatnonzero(ambiguous):
   c=np.array(self.M.babai(ambient[k].tolist()),dtype=np.int64);coeff[k]=c;delta=c@self.moves;dx,dy=delta[:2].copy()
   delta=delta.reshape(15,3);delta[:,:2]-=[dx,dy];u=self.anchor_units.reshape(15,3)+delta;u[:,2]=np.rint([canonicalize_theta(t*1e-6)*1e6 for t in u[:,2]]).astype(np.int64);units[k]=u.ravel();placements[k]=u*1e-6
  return int(ambiguous.sum())
 def raw(self,coefficients):
  """Raw six-decimal input; consumers must canonicalize it exactly once."""
  delta=(np.asarray(coefficients,dtype=np.int64)@self.moves).reshape(15,3)
  delta[:,:2]-=delta[0,:2].copy()
  return (self.anchor_units.reshape(15,3)+delta)*1e-6


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',default='experiments/branch-3.141616-3.141592.json');ap.add_argument('--output',required=True);ap.add_argument('--seconds',type=float,default=300);ap.add_argument('--seed',type=int,default=3110401);ap.add_argument('--target',type=float,default=2.94852695);ap.add_argument('--goal',type=float,default=2.9485269238);ap.add_argument('--orbit-frames',type=int,choices=[1,4],default=4);ap.add_argument('--batch',type=int,default=2048);ap.add_argument('--batches-per-frame',type=int,default=1);ap.add_argument('--movement',type=float,default=50);ap.add_argument('--concentration',type=float,default=10);ap.add_argument('--boundary-bias',type=float,default=5);ap.add_argument('--regularization',type=float,default=1e-5);ap.add_argument('--flat-weight',type=float,default=.1);ap.add_argument('--block-size',type=int,default=20);ap.add_argument('--verify',type=int,default=0);ap.add_argument('--profile',action='store_true');ap.add_argument('--polish',action='store_true');ap.add_argument('--elite-count',type=int,default=10);ap.add_argument('--elite-margin',type=float,default=1e-7);args=ap.parse_args()
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True);events=out/'events.jsonl';log=out/'steps.md';rng=np.random.default_rng(args.seed);start=time.monotonic()
 def emit(event,**data):
  row=dict(time=datetime.now(timezone.utc).isoformat(),event=event,seconds=time.monotonic()-start,**data);line=json.dumps(row,allow_nan=False);print(line,flush=True)
  with events.open('a')as f:f.write(line+'\n')
  if event!='progress':
   with log.open('a')as f:f.write('\n- '+line+'\n')
 raw=Path(args.input).read_bytes();(out/'input-snapshot.json').write_bytes(raw)
 log.write_text('# Batched global-frame lattice campaign\n\nAll45 physical dimensions are decoded. Only integer translations are quotiented. A bounded 128-bit hash cache deduplicates canonical configurations, resetting at1M entries; counts mean unique within that cache epoch. Fresh random global frames are built continuously. Candidate files are written only after reference-valid score improvement.\n\n```text\n'+shlex.join(['.venv/bin/python',*sys.argv])+'\n```\n')
 original=load(args.input)
 if args.polish:
  result=polish(original);original=result.x[:-1].reshape(15,3);radius=float(result.x[-1])
 else:
  cx,cy,radius=compute_mec_fast(original);original[:,:2]-=[cx,cy]
 incumbent=checked(load('solution.json'));best=incumbent.score;decoder=NativeBabai();decoder.reset();elites=EliteArchive(out,args.elite_count,incumbent.score+args.elite_margin,emit)
 times=dict(model=0.,targets=0.,decode=0.,dedup=0.,geometry=0.,validation=0.);counts=dict(frames=0,basis_builds=0,rejected_frames=0,proposals=0,unique_evaluated=0,duplicates=0,cache_resets=0,ambiguous_patches=0,native_mec=0,fallback_mec=0,reference_checks=0,valid_improvements=0);last_report=start;best_violation=1.;base_frame=None;orbit_index=args.orbit_frames
 while time.monotonic()-start<args.seconds:
  if (out.parent/'winner.json').exists():emit('stopped_after_other_win');break
  t=time.perf_counter()
  if orbit_index>=args.orbit_frames:
   try:base_frame=Frame(original,radius,rng.uniform(-np.pi,np.pi),args)
   except ValueError:counts['rejected_frames']+=1;continue
   counts['basis_builds']+=1;orbit_index=0
  frame=base_frame if orbit_index==0 else base_frame.quarter_turn(orbit_index)
  orbit_index+=1
  times['model']+=time.perf_counter()-t;counts['frames']+=1
  if counts['frames']<=3:emit('frame',index=counts['frames'],angle=frame.angle,flat_pairs=frame.flat_pairs,active=int(frame.active.sum()),minimum_stress=float(frame.weights.min()))
  for _ in range(args.batches_per_frame):
   if time.monotonic()-start>=args.seconds:break
   t=time.perf_counter();ambient,desired,gso=frame.targets(rng,args.batch,args);times['targets']+=time.perf_counter()-t
   t=time.perf_counter();decoded=decoder.decode(gso,frame.mu,frame.moves,frame.anchor_units);patches=frame.patch(ambient,decoded);counts['ambiguous_patches']+=patches;times['decode']+=time.perf_counter()-t
   p,units,coeff,ambiguity=decoded
   if args.verify and counts['frames']==1:
    compare=min(args.verify,len(ambient));expected=np.array([frame.M.babai(v.tolist())for v in ambient[:compare]],dtype=np.int64);mismatch=np.flatnonzero(np.any(expected!=coeff[:compare],axis=1));emit('decode_verification',tested=compare,mismatches=len(mismatch),ambiguous=patches)
    if len(mismatch):raise AssertionError('Native Babai disagrees with fpylll')
   counts['proposals']+=len(p);t=time.perf_counter();keep,info=decoder.unique(units);times['dedup']+=time.perf_counter()-t;counts['duplicates']+=int(info[0]);counts['cache_resets']+=int(info[2]);p=p[keep];units=units[keep];coeff=coeff[keep];ambient=ambient[keep];gso=gso[keep];desired=desired[keep]
   if not len(p):continue
   # A geometric seed center is close enough for the certified native support
   # enumerator; unsupported cases still take the checked convex fallback.
   t=time.perf_counter();areas,circles,ok=evaluate_batch(p,centers=[0.,0.]);times['geometry']+=time.perf_counter()-t;counts['unique_evaluated']+=len(p);counts['native_mec']+=int(ok.sum());counts['fallback_mec']+=int((ok==0).sum())
   violation=np.maximum(np.max((areas-1e-6)/frame.constraint.scale[None,:],axis=1),circles[:,2]-args.target);minimum=float(violation.min());best_violation=min(best_violation,minimum)
   # The fast polygon estimator has observed errors above 2e-12. A wider
   # prefilter sends ambiguous candidates to GEOS; it is not a proof bound.
   eligible=np.flatnonzero((areas.max(axis=1)<=1e-6+1e-11)&(circles[:,2]<max(best,elites.cap)+2e-12));eligible=eligible[np.argsort(circles[eligible,2])]
   for index in eligible:
    if circles[index,2]>=best-2e-12 and not elites.wants(units[index],circles[index,2]):continue
    # Native p has already undergone the server's single canonicalization
    # pass. checked/save receive raw input to avoid a second seam transition.
    raw=frame.raw(coeff[index]);t=time.perf_counter();validation=checked(raw);counts['reference_checks']+=1
    if validation.valid:elites.add(raw,units[index],validation.score,frame,coeff[index],ambient[index],gso[index],desired[index],circles[index],args,counts['frames'])
    if validation.valid and validation.score<best-2e-12:
     save(out/'candidate.json',raw);response=check_solution(json.loads((out/'candidate.json').read_text()));(out/'candidate-api.json').write_text(json.dumps(response,indent=2)+'\n')
     if response.get('valid'):
      best=float(response['score']);save(out/'best.json',raw);(out/'best-api.json').write_text(json.dumps(response,indent=2)+'\n');counts['valid_improvements']+=1
     emit('improvement_check',reference_score=validation.score,server_best=best,api=response,counts=counts.copy())
     if response.get('valid')and response.get('score',99)<args.goal:
      (out.parent/'winner.json').write_bytes((out/'best.json').read_bytes());(out.parent/'winner-api.json').write_text(json.dumps(response,indent=2)+'\n')
      times['validation']+=time.perf_counter()-t;emit('done',counts=counts,timings=times,best=best,best_violation=best_violation,goal_met=True);return
    times['validation']+=time.perf_counter()-t
   if time.monotonic()-last_report>10:
    emit('progress',counts=counts.copy(),timings=times.copy(),best=best,best_violation=best_violation,unique_per_second=counts['unique_evaluated']/(time.monotonic()-start));last_report=time.monotonic()
 emit('done',counts=counts,timings=times,best=best,best_violation=best_violation,goal_met=best<args.goal,unique_per_second=counts['unique_evaluated']/(time.monotonic()-start))

if __name__=='__main__':main()
