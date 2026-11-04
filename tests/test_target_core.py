"""Disabled assumptions must remove only their intended necessary constraint."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from experiment_target_core import AssumedModel
from experiment_conflict_master import categorical_support_cut
from semicircle_packing.fast_mec import compute_mec_fast


def test_disabled_pair_table_leaves_all_pose_combinations_possible():
    domains=[np.zeros((2,3)),np.zeros((2,3))]
    for a in range(2):
        for b in range(2):
            m=AssumedModel(domains,4.);m.pair(0,1,np.array([[0,0]]))
            m.model.add(m.variables[0]==a);m.model.add(m.variables[1]==b)
            assert m.solve([],2)['status']in('OPTIMAL','FEASIBLE')
            result=m.solve([0],2)
            assert (result['status']in('OPTIMAL','FEASIBLE'))==(a==b==0)


def test_empty_pair_relation_can_be_disabled():
    m=AssumedModel([np.zeros((2,3)),np.zeros((2,3))],4.)
    m.pair(0,1,np.empty((0,2),dtype=int))
    assert m.solve([],2)['status']in('OPTIMAL','FEASIBLE')
    result=m.solve([0],2)
    assert result['status']=='INFEASIBLE'and result['core']==[0]


def test_radius_assumption_disables_plane_not_total_definitions():
    data=json.loads((Path(__file__).parent/'fixtures/packing_valid_20261104.json').read_text())
    base=np.array([[d['x'],d['y'],d['theta']]for d in data]);domains=[p[None]for p in base]
    cut=categorical_support_cut(domains,base,compute_mec_fast(base))
    m=AssumedModel(domains,2.94852695);m.cut(cut,0)
    assert m.solve([],2)['status']in('OPTIMAL','FEASIBLE')
    result=m.solve([0],2)
    assert result['status']=='INFEASIBLE'and result['core']==[0]


def test_sufficient_core_preserves_pair_labels():
    domains=[np.zeros((2,3))for _ in range(3)]
    equal=np.array([[0,0],[1,1]]);unequal=np.array([[0,1],[1,0]])
    m=AssumedModel(domains,4.);m.pair(0,1,equal);m.pair(1,2,equal);m.pair(0,2,unequal)
    result=m.solve([0,1,2],3)
    assert result['status']=='INFEASIBLE'and set(result['core'])=={0,1,2}
