"""Reproducible correctness/throughput check of conditional angular 2-SAT."""
import json,itertools,time
from pathlib import Path
import numpy as np
from angular_rounder import round_angles,solve_binary_constraints,_pair_indices
from semicircle_packing.exact_area import polygon_intersection_areas
from semicircle_packing.polish import load,checked
from semicircle_packing.fast_mec import compute_mec_fast
rng=np.random.default_rng(41104);cases=0
for n in range(1,8):
 for repeat in range(50):
  unary=rng.random((n,2))>.1;tables=rng.random((n,n,2,2))>.2
  answer=solve_binary_constraints(unary,tables)
  brute=[bits for bits in itertools.product(range(2),repeat=n) if all(unary[i,bits[i]] for i in range(n)) and all(tables[i,j,bits[i],bits[j]] for i in range(n) for j in range(i+1,n))]
  assert (answer is not None)==bool(brute)
  if answer is not None:assert tuple(answer) in brute
  cases+=1
p=load('solution.json');cx,cy,r=compute_mec_fast(p);q=round_angles(p,r+1e-11,center=[cx,cy]);assert q is not None;validation=checked(q);assert validation.valid
assert round_angles(p,2.,center=[cx,cy]) is None
for repeat in range(10):
 trial=p.copy();trial[:,2]+=rng.uniform(-.3e-6,.3e-6,len(p))
 rounded=round_angles(trial,r+1e-11,center=[cx,cy]);assert rounded is not None and checked(rounded).valid
variants=np.repeat(p,2,axis=0);variants[:,2]+=rng.uniform(-.5e-6,.5e-6,len(variants));batch=polygon_intersection_areas(variants);pairs,indices=_pair_indices(len(p))
for repeat in range(60):
 row=int(rng.integers(len(pairs)));a,b=rng.integers(2,size=2);i,j=pairs[row]
 direct=polygon_intersection_areas(variants[[2*i+a,2*j+b]])[0]
 assert abs(batch[indices[row,a,b]]-direct)<1e-15
start=time.perf_counter()
for _ in range(1000):round_angles(p,r+1e-11,center=[cx,cy])
result=dict(seed=41104,randomized_cases=cases,known_valid_score=validation.score,impossible_rejected=True,nearby_continuous_angles_recovered=10,variant_index_checks=60,microseconds_with_fixed_center=(time.perf_counter()-start)*1000)
output=Path('experiments/2026-11-04/lattice/angle-rounder-check.json');output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
