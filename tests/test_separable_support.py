"""Shared coordinate encoding must remain conservative on every pose."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import experiment_conflict_master as master
from experiment_phase_pool import phase_domains
from experiment_phase_bundle_resume import all_phase_supports
from experiment_support_bundle import solve_bundle
from separable_support import prepare,factor,piece_values,audit,Inserter
from semicircle_packing.fast_mec import compute_mec_fast


def fixture():
    data=json.loads((Path(__file__).parent/'fixtures/packing_valid_20261104.json').read_text())
    return np.array([[d['x'],d['y'],d['theta']]for d in data])


def test_every_pose_and_cut_factorization_is_conservative():
    base=fixture();domains,anchors,circles,*_=phase_domains(base,3,50e-6,1,1,100)
    cuts=all_phase_supports(domains,anchors,circles);report=audit(domains,cuts)
    assert report['max_above_true']<3e-14
    assert report['max_int64_activity']<2**62
    assert report['theta_table_entries']<sum(len(v)for c in cuts for v in c['coefficients'].values())
    prepared=prepare(domains)
    assert all(np.min(d['delta'])>=0 for d in prepared)
    assert any(np.any(factor(c,prepared)['pieces'][i]['xy']<0)for c in cuts for i in c['coefficients'])


def test_factored_integer_bounds_allow_mixed_domain_points():
    base=fixture();domains,anchors,circles,*_=phase_domains(base,3,50e-6,1,1,100)
    prepared=prepare(domains);cuts=all_phase_supports(domains,anchors,circles)
    factors=[factor(c,prepared)for c in cuts];rng=np.random.default_rng(110454)
    for _ in range(12):
        selected=[rng.integers(len(d))for d in domains]
        p=np.array([d[k]for d,k in zip(domains,selected)]);radius=compute_mec_fast(p)[2]
        for f in factors:
            lhs=sum(int(piece_values(f,prepared,i)[selected[i]])for i in f['pieces'])
            lower=(lhs-f['offset'])/f['scale']
            assert lower<=radius+1e-11


def test_actual_factored_cp_control_and_target_behaviour(tmp_path):
    base=fixture();domains=[row[None]for row in base]
    inserter=Inserter(domains);original=master.add_support_cut;master.add_support_cut=inserter
    try:
        impossible=solve_bundle(domains,[],2.94852695,tmp_path/'impossible.json',seconds=5,
                               expected_score=2.948526966725367)
        possible=solve_bundle(domains,[],2.948526967,tmp_path/'possible.json',seconds=5,
                             expected_score=2.948526966725367)
    finally:master.add_support_cut=original
    assert impossible['positive_control']['status']=='OPTIMAL'
    assert impossible['finite_exhausted']and not impossible['valid']
    assert possible['positive_control']['status']=='OPTIMAL'
    assert possible['valid']and abs(possible['score']-2.948526966725367)<1e-12


def test_cp_projection_matches_factored_integers_for_nontrivial_pose_ids():
    from ortools.sat.python import cp_model
    base=fixture();domains,anchors,circles,*_=phase_domains(base,3,50e-6,1,1,100)
    cut=all_phase_supports(domains,anchors,circles)[0];prepared=prepare(domains);f=factor(cut,prepared)
    rng=np.random.default_rng(110455)
    for _ in range(4):
        selected=[int(rng.integers(len(d)))for d in domains]
        lhs=sum(int(piece_values(f,prepared,i)[selected[i]])for i in f['pieces'])
        minimum=-( -(lhs-f['offset'])//f['ratio'])
        for level,expected in [(minimum,True),(minimum-1,False)]:
            model=cp_model.CpModel();variables=[model.new_int_var(0,len(d)-1,f'p{i}')for i,d in enumerate(domains)]
            for variable,value in zip(variables,selected):model.add(variable==value)
            radius=model.new_int_var(level,level,'radius');Inserter(domains)(model,variables,radius,cut,'test')
            solver=cp_model.CpSolver();solver.parameters.num_search_workers=1
            status=solver.solve(model)
            assert (status in(cp_model.OPTIMAL,cp_model.FEASIBLE))==expected
