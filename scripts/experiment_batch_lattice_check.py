"""Independent checks of native Babai, canonical output, and geometry."""
from __future__ import annotations
import os
for name in ('VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import json,time,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from experiment_batch_lattice import NativeBabai,Frame,EliteArchive
from semicircle_packing.polish import load,checked
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.geometry import canonicalize_theta
from breakout_geometry import evaluate_batch

args=SimpleNamespace(target=2.94852695,flat_weight=.1,regularization=1e-5,block_size=20,concentration=10.,boundary_bias=5.,movement=50.)
rng=np.random.default_rng(3110499);native=NativeBabai();native.reset();comparisons=mismatches=ambiguities=raw_mismatches=0;records=[]
for name in ['experiments/branch-3.141616-3.141592.json','experiments/2026-11-04/round2/topology/continuous-best.json']:
 p=load(name);cx,cy,r=compute_mec_fast(p);p[:,:2]-=[cx,cy]
 for angle in [-2.7,-.9,.4,2.8]:
  frame=Frame(p,r,angle,args);ambient,_,target=frame.targets(rng,1024,args);decoded=native.decode(target,frame.mu,frame.moves,frame.anchor_units);expected=np.array([frame.M.babai(t.tolist())for t in ambient],dtype=np.int64);raw_bad=np.any(expected!=decoded[2],axis=1);raw_mismatches+=int(raw_bad.sum());assert np.all(~raw_bad|decoded[3].astype(bool));ambiguities+=frame.patch(ambient,decoded);bad=np.flatnonzero(np.any(decoded[2]!=expected,axis=1));comparisons+=len(ambient);mismatches+=len(bad);assert not len(bad)
  for k in range(8):
   raw=frame.raw(decoded[2][k]);raw[:,2]=[canonicalize_theta(t)for t in raw[:,2]];np.testing.assert_allclose(raw,decoded[0][k],atol=1e-15,rtol=0)
  records.append(dict(input=name,angle=angle,flat_pairs=frame.flat_pairs,comparisons=len(ambient),mismatches=len(bad)))
# Verify the reused GSO against separately constructed, physically rotated
# embeddings. Reuse is exact because each quarter turn is a signed permutation.
from fpylll import GSO,IntegerMatrix
quarter_comparisons=quarter_mismatches=0
for turn in (1,2,3):
 rotated=frame.quarter_turn(turn)
 basis=frame.reduced.copy()
 for _ in range(turn):
  coordinates=basis[:,:45].reshape(45,15,3)
  coordinates[:,:,:2]=coordinates[:,:,:2][:,:,[1,0]]*[-1,1]
 independent=GSO.Mat(IntegerMatrix.from_matrix(basis.tolist()),update=True)
 ambient,_,target=rotated.targets(rng,1024,args)
 physical=ambient.copy()
 for _ in range(turn):
  coordinates=physical[:,:45].reshape(len(physical),15,3)
  coordinates[:,:,:2]=coordinates[:,:,:2][:,:,[1,0]]*[-1,1]
 expected=np.array([independent.babai(t.tolist())for t in physical],dtype=np.int64)
 decoded=native.decode(target,rotated.mu,rotated.moves,rotated.anchor_units)
 rotated.patch(ambient,decoded)
 bad=np.any(expected!=decoded[2],axis=1)
 quarter_comparisons+=len(ambient);quarter_mismatches+=int(bad.sum())
 assert not bad.any()
 assert np.allclose(rotated.moves@rotated.J.T,frame.moves@frame.J.T,atol=2e-12)
# Explicit nearest-even ties, including negative halves.
moves=np.zeros((1,45),dtype=np.int64);moves[0,0]=1;ties=np.array([[.5],[1.5],[2.5],[-.5],[-1.5],[-2.5]])
assert np.array_equal(native.decode(ties,np.zeros((1,1)),moves,np.zeros(45,dtype=np.int64))[2].ravel(),[0,2,2,0,-2,-2])
# A 3*pi raw input canonicalizes to -3.141593, whose second normalization
# becomes +3.141592. Preserve raw input so validation performs exactly one pass.
seam=object.__new__(Frame);seam.moves=np.zeros((1,45),dtype=np.int64);seam.anchor_units=np.zeros(45,dtype=np.int64);seam.anchor_units[2]=round(3*np.pi*1e6)
seam_raw=seam.raw(np.zeros(1,dtype=np.int64));seam_geometry=native.decode(np.zeros((1,1)),np.zeros((1,1)),seam.moves,seam.anchor_units)[0][0]
assert canonicalize_theta(seam_raw[0,2])==seam_geometry[0,2]
assert canonicalize_theta(seam_geometry[0,2])!=seam_geometry[0,2]
# Exact integer translations deduplicate; a one-microradian physical angle does not.
known=load('solution.json');units=np.rint(known.ravel()*1e6).astype(np.int64);samples=np.tile(units,(3,1));samples[1].reshape(15,3)[:,:2]+=[17,-31];samples[2,2]+=1;native.reset();keep,_=native.unique(samples);assert keep.tolist()==[True,False,True]
# A native zero-coefficient completion retains a known valid positive witness.
positive=native.decode(np.zeros((1,45)),frame.mu,frame.moves,units)[0];validation=checked(positive[0]);assert validation.valid
areas,circles,ok=evaluate_batch(positive,centers=compute_mec_fast(known)[:2]);assert areas.max()<1e-6+2e-12;assert abs(circles[0,2]-validation.score)<2e-12
# Archive origin/move records survive serialization; translations do not consume
# another scarce elite slot. Actual reference validity is checked by the caller.
with tempfile.TemporaryDirectory()as directory:
 archive=EliteArchive(Path(directory),2,validation.score+1e-7,lambda *a,**kw:None)
 archive_args=SimpleNamespace(seed=3110499,input='solution.json')
 assert archive.add(positive[0],units,validation.score,frame,np.zeros(45,dtype=np.int64),np.zeros(frame.reduced.shape[1],dtype=np.int64),np.zeros(45),np.zeros(len(frame.J)),circles[0],archive_args,0)
 assert not archive.wants(samples[1],validation.score)
 manifest=json.loads((Path(directory)/'elites'/'manifest.json').read_text());assert len(manifest['entries'])==1
 with np.load(Path(directory)/'elites'/manifest['entries'][0]['sidecar'])as saved:assert np.array_equal(saved['units'],units)
report=dict(seed=3110499,comparisons=comparisons,mismatches=mismatches,raw_mismatches=raw_mismatches,ambiguous_patches=ambiguities,quarter_turn_comparisons=quarter_comparisons,quarter_turn_mismatches=quarter_mismatches,single_input_canonicalization_seam_pass=True,nearest_even_ties_pass=True,only_translations_deduplicated=True,elite_archive_roundtrip_pass=True,known_valid_score=validation.score,native_radius=float(circles[0,2]),native_positive=bool(ok[0]),frames=records)
path=Path('experiments/2026-11-04/round3/batch/decoder-check.json');path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
