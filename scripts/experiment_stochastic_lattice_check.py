"""Controls for the optional heuristic; deterministic decoder remains intact."""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from experiment_stochastic_lattice import NativeStochastic
from experiment_batch_lattice import Frame
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.polish import load
from semicircle_packing.geometry import canonicalize_theta

args=SimpleNamespace(target=2.94852698,flat_weight=.1,regularization=1e-5,block_size=20,concentration=10.,boundary_bias=5.,movement=50.)
decoder=NativeStochastic();rng=np.random.default_rng(3110521);checked=0;changed=0;records=[]
for path in ['experiments/branch-3.141616-3.141592.json','experiments/2026-11-04/round2/topology/continuous-best.json']:
 p=load(path);cx,cy,r=compute_mec_fast(p);p[:,:2]-=[cx,cy]
 for angle in (-2.4,.7):
  frame=Frame(p,r,angle,args);ambient,_,target=frame.targets(rng,1024,args);norms=np.array([frame.M.get_r(j,j)for j in range(45)])
  deterministic=decoder.decode(target,frame.mu,frame.moves,frame.anchor_units)
  control,changes,_=decoder.decode_random(target,frame.mu,norms,frame.moves,frame.anchor_units,seed=42,temperature=0)
  for actual,expected in zip(control,deterministic):assert np.array_equal(actual,expected)
  assert not changes.any();expected=np.array([frame.M.babai(a.tolist())for a in ambient]);assert np.array_equal(control[2],expected)
  sample,changes,distances=decoder.decode_random(target,frame.mu,norms,frame.moves,frame.anchor_units,seed=371911)
  repeated,repeated_changes,repeated_distances=decoder.decode_random(target,frame.mu,norms,frame.moves,frame.anchor_units,seed=371911)
  for actual,expected in zip(sample,repeated):assert np.array_equal(actual,expected)
  assert np.array_equal(changes,repeated_changes)and np.array_equal(distances,repeated_distances)
  assert changes.min()>=1 and changes.max()<=3
  for row in range(16):
   residual=target[row].copy();flips=0
   for j in range(44,-1,-1):
    value=residual[j];near=round(float(value));actual=int(sample[2][row,j]);flips+=actual!=near
    assert abs(actual-near)<=1
    if actual!=near and value!=near:assert actual-near==(1 if value>near else-1)
    residual[:j]-=actual*frame.mu[j,:j]
   assert flips==changes[row]
   raw=frame.raw(sample[2][row]);raw[:,2]=[canonicalize_theta(v)for v in raw[:,2]];np.testing.assert_allclose(raw,sample[0][row],atol=1e-15,rtol=0)
  other,_,_=decoder.decode_random(target,frame.mu,norms,frame.moves,frame.anchor_units,seed=371912)
  differences=int(np.any(other[2]!=sample[2],axis=1).sum());assert differences>900;changed+=differences;checked+=len(target)
  records.append(dict(input=path,angle=angle,zero_noise_fpylll_comparisons=len(target),different_seed_changed=differences,mean_changed_decisions=float(changes.mean()),mean_distance_ratio=float(distances[:,1].sum()/distances[:,0].sum())))
# Zero-noise path preserves native single-pass seam geometry, including 3*pi.
anchor=np.zeros(45,dtype=np.int64);anchor[2]=round(3*np.pi*1e6);moves=np.zeros((1,45),dtype=np.int64);target=np.array([[.5],[1.5],[-.5],[-1.5]])
deterministic=decoder.decode(target,np.zeros((1,1)),moves,anchor);control,_,_=decoder.decode_random(target,np.zeros((1,1)),np.ones(1),moves,anchor,seed=42,temperature=0)
for actual,expected in zip(control,deterministic):assert np.array_equal(actual,expected)
assert control[1][0,2]==-3141593 and canonicalize_theta(control[0][0,0,2])!=control[0][0,0,2]
# Center diversity is only a diagnostic: the full decoder keeps angle variants.
samples=np.zeros((4,45),dtype=np.int64);samples[1].reshape(15,3)[:,:2]+=[17,-31];samples[2,2]=1;samples[3,3]=1;decoder.reset()
assert decoder.unique(samples)[0].tolist()==[True,False,True,True]
assert decoder.unique_centers(samples)[0].tolist()==[True,False,False,True]
result=dict(seed=3110521,zero_noise_comparisons=checked+4,zero_noise_matches_fpylll=True,repeated_seed_bitwise_equal=True,different_seed_changes=changed,changed_decisions_are_conditional_alternatives=True,raw_to_single_canonical_geometry=True,seam_zero_noise_pass=True,center_diagnostic_does_not_filter_angles=True,records=records)
out=Path('experiments/2026-11-04/round3/stochastic');out.mkdir(parents=True,exist_ok=True);(out/'decoder-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
