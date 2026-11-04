import itertools,json,sys
from pathlib import Path
import numpy as np
from ortools.sat.python import cp_model
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from lazy_pair_oracle import PairOracle
from semicircle_packing.geometry import Semicircle,semicircle_polygon
from semicircle_packing.polish import checked


def fixture_domains():
 # Unit-disk lens near the1e-6 area threshold; some cells need exact completion.
 a=np.array([[0.,0.,t]for t in(-.00002,0.,.00002)])
 b=np.array([[x,0.,np.pi+t]for x in(1.9999173,1.9999175,1.9999177)for t in(-.00002,0.,.00002)])
 c=np.array([[0.,3.,0.],[.000001,3.,.1]])
 return[a,b,c]


def exact_assignments(domains):
 polygons=[[semicircle_polygon(Semicircle(*v))for v in d]for d in domains]
 return{a for a in itertools.product(*(range(len(d))for d in domains))if all(polygons[i][a[i]].intersection(polygons[j][a[j]]).area<=1e-6 for i,j in itertools.combinations(range(len(domains)),2))}


def test_every_oracle_cell_matches_geos_and_is_cached():
 domains=fixture_domains();oracle=PairOracle(domains);expected=exact_assignments(domains);assert oracle.stats['initial_uncertain']>0
 actual={a for a in itertools.product(*(range(len(d))for d in domains))if not oracle.check(a)};assert actual==expected
 count=oracle.stats['geos_pairs']
 for a in itertools.product(*(range(len(d))for d in domains)):oracle.check(a)
 assert oracle.stats['geos_pairs']==count


def test_lazy_pair_nogoods_and_exhaustive_tables_have_same_projection():
 domains=fixture_domains();oracle=PairOracle(domains);model=cp_model.CpModel();variables=[model.new_int_var(0,len(d)-1,f'p{i}')for i,d in enumerate(domains)]
 for i,j,allowed in oracle.relaxed_tables():model.add_allowed_assignments([variables[i],variables[j]],allowed.tolist())
 accepted=set()
 while True:
  solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;status=solver.solve(model)
  if status==cp_model.INFEASIBLE:break
  assert status in(cp_model.OPTIMAL,cp_model.FEASIBLE);a=tuple(solver.value(v)for v in variables);bad=oracle.check(a)
  if bad:
   for conflict in bad:
    i,j=conflict['pair'];model.add_forbidden_assignments([variables[i],variables[j]],[conflict['values']])
  else:accepted.add(a);model.add_forbidden_assignments(variables,[list(a)])
 assert accepted==exact_assignments(domains)


def test_known_valid_full_packing_survives_lazy_relaxation():
 path=Path(__file__).parent/'fixtures/packing_valid_20261104.json';raw=json.loads(path.read_text());p=np.array([[v['x'],v['y'],v['theta']]for v in raw]);assert checked(p).valid;oracle=PairOracle([p[i:i+1]for i in range(15)]);assert not oracle.check([0]*15)


def test_known_valid_control_survives_full_lazy_support_model(tmp_path):
 from lazy_phase_solver import solve_phase
 path=Path(__file__).parent/'fixtures/packing_valid_20261104.json';raw=json.loads(path.read_text());p=np.array([[v['x'],v['y'],v['theta']]for v in raw]);score=checked(p).score
 result=solve_phase([p[i:i+1]for i in range(15)],p,score+1e-9,tmp_path,seconds=8,control=True)
 assert result['valid']and result['status']=='CONTROL_FEASIBLE'
