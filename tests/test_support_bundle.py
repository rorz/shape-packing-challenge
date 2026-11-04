"""Every bootstrap inequality must remain a free-center lower bound."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from experiment_support_bundle import support_bundle,solve_bundle
from semicircle_packing.fast_mec import compute_mec_fast


def fixture():
    data=json.loads((Path(__file__).parent/'fixtures/packing_valid_20261104.json').read_text())
    return np.array([[d['x'],d['y'],d['theta']]for d in data])


def test_all_bundle_inequalities_hold_under_recombination_and_translation():
    base=fixture();rng=np.random.default_rng(110449)
    variants=base[None]+rng.integers(-100,101,(5,15,3))*1e-6
    variants[0]=base;variants[1]=base;variants[1,:,:2]+=[.11,-.07]
    domains=[variants[:,i]for i in range(15)]
    bundle=support_bundle(domains,base,compute_mec_fast(base))
    assert len(bundle)>1
    for selection in [np.ones(15,dtype=int),*[rng.integers(0,5,15)for _ in range(24)]]:
        p=np.array([d[k]for d,k in zip(domains,selection)])
        radius=compute_mec_fast(p)[2]
        for cut in bundle:
            lower=(sum(v[selection[i]]for i,v in cut['coefficients'].items())-cut['offset'])/cut['scale']
            assert lower<=radius+1e-11


def test_bootstrapped_actual_positive_control_and_small_target(tmp_path):
    base=fixture();domains=[row[None]for row in base]
    result=solve_bundle(domains,[],2.94852695,tmp_path/'bundle.json',seconds=5,expected_score=2.948526966725367)
    assert result['positive_control']['status']=='OPTIMAL'
    assert result['finite_exhausted']and not result['valid']
    assert result['candidates']==0


def test_interval_bundle_hooks_restore_main_and_keep_actual_control(tmp_path):
    import experiment_conflict_master as master
    original_pair,original_support=master.add_pair_constraint,master.add_support_cut
    base=fixture();domains=[row[None]for row in base]
    result=solve_bundle(domains,[(0,1,np.array([[0,0]]))],2.94852695,tmp_path/'combined.json',
                        seconds=5,expected_score=2.948526966725367,use_intervals=True)
    assert result['positive_control']['status']=='OPTIMAL'
    assert result['finite_exhausted']and not result['valid']
    assert master.add_pair_constraint is original_pair
    assert master.add_support_cut is original_support
