"""Focused exactness tests for the standalone finite-relation prototype."""
from pathlib import Path
import sys
import numpy as np
import pytest
from ortools.sat.python import cp_model
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from interval_pair_relation import quotient,intervals,reconstruct,verify_element_encoding,add_interval_constraint,add_allowed_interval_pair
from interval_order_optimization import add_hamming_interval_pair,optimize


class Collect(cp_model.CpSolverSolutionCallback):
 def __init__(self,a,b):super().__init__();self.a=a;self.b=b;self.values=set()
 def on_solution_callback(self):self.values.add((self.value(self.a),self.value(self.b)))


@pytest.mark.parametrize('seed',range(12))
def test_interval_cp_projection_equals_every_synthetic_tuple(seed):
 rng=np.random.default_rng(seed);mask=rng.random((7,9))<.5
 # Force duplicates so the element class maps are genuinely exercised.
 mask[3]=mask[0];mask[:,7]=mask[:,1]
 if seed==0:mask[:]=False
 if seed==1:mask[:]=True
 rows,cols,small=quotient(mask);order=rng.permutation(small.shape[1]);segments=intervals(small,order)
 assert np.array_equal(reconstruct(small.shape,order,segments),small)
 assert verify_element_encoding(mask,rows,cols,order,segments)==mask.size
 model=cp_model.CpModel();a=model.new_int_var(0,6,'a');b=model.new_int_var(0,8,'b');add_interval_constraint(model,a,b,rows,cols,order,segments)
 solver=cp_model.CpSolver();solver.parameters.enumerate_all_solutions=True;solver.parameters.num_search_workers=1;collector=Collect(a,b);solver.solve(model,collector)
 assert collector.values==set(map(tuple,np.column_stack(np.nonzero(mask))))


@pytest.mark.parametrize('prefix,pair',[
 ('conflicts/network-angle3',(4,8)),('conflicts/network-angle3',(5,8)),('conflicts-original/wide',(6,10)),
])
def test_saved_actual_relation_exhaustive_element_mapping(prefix,pair):
 path=Path(__file__).parents[1]/'experiments/2026-11-04/round3'/prefix
 if not path.with_suffix('.tables.npz').exists():pytest.skip('Optional immutable experiment archive absent')
 with np.load(path.with_suffix('.domains.npz'))as data:ni,nj=(len(data[f'p{k}'])for k in pair)
 with np.load(path.with_suffix('.tables.npz'))as data:allowed=data[f'{pair[0]}_{pair[1]}']
 mask=np.zeros((ni,nj),dtype=bool);mask[allowed[:,0],allowed[:,1]]=True;rows,cols,small=quotient(mask);order=np.arange(small.shape[1]);segments=intervals(small,order)
 assert verify_element_encoding(mask,rows,cols,order,segments)==ni*nj
 assert np.array_equal(reconstruct(small.shape,order,segments)[rows][:,cols],mask)
 # Instantiate the actual CP primitive as a smoke/positive-control check.
 model=cp_model.CpModel();a=model.new_int_var(0,ni-1,'a');b=model.new_int_var(0,nj-1,'b');add_allowed_interval_pair(model,a,b,allowed,ni,nj)
 model.add(a==int(allowed[0,0]));model.add(b==int(allowed[0,1]));solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;assert solver.solve(model)==cp_model.OPTIMAL


@pytest.mark.parametrize('seed',range(6))
def test_hamming_builder_exhaustive_projection_and_tour_objective(seed):
 rng=np.random.default_rng(seed+100);mask=rng.random((7,9))<.5;mask[3]=mask[0];mask[:,7]=mask[:,1];rows,cols,small=quotient(mask)
 order,stats=optimize(small,iterations=10000);segments=intervals(small,order)
 assert stats['final_intervals']==len(segments)<=stats['initial_intervals']
 assert verify_element_encoding(mask,rows,cols,order,segments)==mask.size
 model=cp_model.CpModel();a=model.new_int_var(0,6,'a');b=model.new_int_var(0,8,'b');allowed=np.column_stack(np.nonzero(mask));add_hamming_interval_pair(model,a,b,allowed,7,9)
 solver=cp_model.CpSolver();solver.parameters.enumerate_all_solutions=True;solver.parameters.num_search_workers=1;collector=Collect(a,b);solver.solve(model,collector)
 assert collector.values==set(map(tuple,allowed))
