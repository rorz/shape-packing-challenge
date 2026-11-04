import itertools,sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from exact_domain_pruning import IntegerCut,pack_relations,prune,verify_trace


@pytest.mark.parametrize('seed',range(20))
def test_pruning_preserves_every_exhaustive_solution(seed):
 rng=np.random.default_rng(seed);sizes=[4,5,3];masks={(0,1):rng.random((4,5))<.7,(1,2):rng.random((5,3))<.7,(0,2):rng.random((4,3))<.7};cuts=[IntegerCut({i:rng.integers(-8,9,size=n,dtype=np.int64)for i,n in enumerate(sizes)},rhs=int(rng.integers(-4,8)))for _ in range(3)]
 feasible=[]
 for assignment in itertools.product(*(range(n)for n in sizes)):
  if all(mask[assignment[i],assignment[j]]for(i,j),mask in masks.items())and all(sum(int(v[assignment[i]])for i,v in c.coefficients.items())<=c.rhs for c in cuts):feasible.append(assignment)
 relations=pack_relations(masks);active,trace,status=prune(sizes,relations,cuts);assert verify_trace(sizes,relations,cuts,active,trace,status)
 for assignment in feasible:assert all(active[i][v]for i,v in enumerate(assignment))
 if status['infeasible']:assert not feasible


def test_pair_hole_and_cut_pruning_reach_fixed_point():
 masks={(0,1):np.array([[1,0,1],[0,1,0]],dtype=bool)};relations=pack_relations(masks);cuts=[IntegerCut({1:np.array([2,0,2],dtype=np.int64)},0)];active,trace,status=prune([2,3],relations,cuts)
 assert not status['infeasible'];assert active[0].tolist()==[False,True];assert active[1].tolist()==[False,True,False];assert verify_trace([2,3],relations,cuts,active,trace,status)
