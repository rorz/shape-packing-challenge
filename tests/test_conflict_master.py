"""Soundness checks for the finite placement master and geometric cuts."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from experiment_conflict_master import boundary_conflict, correlated_domains, exact_tables, overlap_conflict_core, solve, categorical_support_cut,compress_pair_relation,add_pair_constraint
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.geometry import Semicircle, semicircle_polygon


def test_full_correlated_moves_keep_angle_displacements():
    p = np.array([[0., 0., 0.], [1., 0., 1.]])
    moves = np.array([[[1, 0, 0], [1, 1, 7]], [[1, 0, 0], [1, 0, 0]]])
    domains, selected = correlated_domains(p, moves, directions=1, jitter=0)
    assert selected == [0]
    assert any(np.allclose(q, [1., 1e-6, 1.000007], atol=1e-12, rtol=0) for q in domains[1])
    assert all(q[0] == 0 for q in domains[0])


def test_local_corrections_are_additive_to_correlated_states():
    p = np.array([[0.,0.,0.],[1.,0.,1.]])
    moves = np.array([[[0,0,0],[0,7,9]]])
    domains, selected = correlated_domains(p,moves,directions=1,jitter=0,local_steps=1)
    values = {tuple(np.rint(row*1e6).astype(int)) for row in domains[1]}
    assert selected == [0]
    assert (1000000,7,1000009) in values
    assert (1000001,1,1000001) in values
    assert len(values) == 29


def test_minkowski_repairs_every_displaced_state_and_never_truncates():
    import pytest
    p=np.array([[0.,0.,0.],[1.,0.,1.]])
    moves=np.array([[[0,0,0],[0,7,9]]])
    domains,_=correlated_domains(p,moves,directions=1,jitter=3,minkowski_steps=1)
    values={tuple(np.rint(row*1e6).astype(int))for row in domains[1]}
    assert (1000001,8,1000012)in values
    assert (999999,-8,999988)in values
    with pytest.raises(ValueError,match='no truncation'):
        correlated_domains(p,moves,directions=1,jitter=3,minkowski_steps=1,max_domain=2)


def test_polygon_tables_agree_with_direct_geos():
    domains = [np.array([[0., 0., 0.], [0., .1, .1]]),
               np.array([[0., 0., 3.141592], [2.1, 0., 1.]])]
    tables, evaluations = exact_tables(domains)
    allowed = set(map(tuple, tables[0][2]))
    assert evaluations == 4
    for i, a in enumerate(domains[0]):
        for j, b in enumerate(domains[1]):
            area = semicircle_polygon(Semicircle(*a)).intersection(semicircle_polygon(Semicircle(*b))).area
            assert ((i, j) in allowed) == (area <= 1e-6)


def test_true_area_band_agrees_with_full_geos_flat_and_arc_contacts():
    import json
    fixture=Path(__file__).parent/'fixtures/packing_valid_20261104.json'
    base=np.array([[d['x'],d['y'],d['theta']]for d in json.loads(fixture.read_text())])
    rng=np.random.default_rng(110445)
    for i,j in [(1,5),(0,1)]:
        domains=[np.round(base[k]+rng.integers(-100,101,(15,3))*1e-6,6)for k in (i,j)]
        full,_=exact_tables(domains)
        def allowed(t):
            return set(map(tuple,t[0][2]))if t else set((a,b)for a in range(15)for b in range(15))
        for local in (False,True):
            stats={}
            screened,_=exact_tables(domains,band_screen=True,stats=stats,local_loss=local)
            assert allowed(full)==allowed(screened)
            assert stats['native_pairs']==225
            assert stats['screen_allowed']+stats['screen_rejected']+stats['geos_pairs']==225


def test_local_annulus_loss_bound_covers_actual_polygon_losses():
    import shapely
    from conflict_area_band import true_area_loss_band
    rng=np.random.default_rng(110446)
    a=rng.uniform(-1.5,1.5,(8,3));b=rng.uniform(-1.5,1.5,(8,3))
    cases=[(a,b)]
    # Arc tangency and almost-opposed flat wedges stress short active arcs.
    a=np.zeros((8,3));b=np.zeros((8,3));b[:,0]=2-np.geomspace(1e-8,.01,8);b[:,2]=np.pi
    cases.append((a,b))
    a=np.zeros((8,3));b=np.zeros((8,3));b[:,0]=np.linspace(-1e-5,1e-5,8);b[:,1]=.8;b[:,2]=np.pi+np.linspace(-1e-5,1e-5,8)
    cases.append((a,b))
    for a,b in cases:
        true,loss=true_area_loss_band(a,b)
        pa=np.array([semicircle_polygon(Semicircle(*p))for p in a],dtype=object)
        pb=np.array([semicircle_polygon(Semicircle(*p))for p in b],dtype=object)
        actual=shapely.area(shapely.intersection(pa[:,None],pb[None,:]))
        assert np.all(actual<=true+1e-9)
        assert np.all(true-actual<=loss+1e-9)


def test_boundary_witness_lower_bound_and_cut_survive_other_shapes():
    rng = np.random.default_rng(110443)
    for _ in range(8):
        p = rng.normal(size=(5, 3))
        circle = compute_mec_fast(p)
        ids, lower = boundary_conflict(p, circle[:2], circle[2]-.05)
        assert ids is not None and 1 <= len(ids) <= 3
        assert lower <= circle[2]+1e-10
        subset = compute_mec_fast(p[ids])
        assert subset[2] >= lower-1e-10
        assert subset[2] > circle[2]-.05


def test_known_incumbent_has_no_false_radius_cut():
    import json
    fixture = Path(__file__).parent/'fixtures/packing_valid_20261104.json'
    p = np.array([[d['x'], d['y'], d['theta']] for d in json.loads(fixture.read_text())])
    circle = compute_mec_fast(p)
    ids, lower = boundary_conflict(p, circle[:2], circle[2]+1e-9)
    assert ids is None
    assert lower <= circle[2]+1e-10


def test_pair_conflict_core_only_claims_finite_model():
    domains = [np.zeros((2,3)) for _ in range(3)]
    equal = np.array([[0,0],[1,1]])
    unequal = np.array([[0,1],[1,0]])
    result = overlap_conflict_core(domains, [(0,1,equal),(1,2,equal),(0,2,unequal)])
    assert result['status'] == 'INFEASIBLE'
    assert len(result['pairs']) == 3
    assert 'finite domains only' in result['note']


def test_pair_quotient_preserves_all_tuples_and_compresses_duplicates():
    rng=np.random.default_rng(110447)
    for rows,columns in [(7,9),(4,4),(1,8),(9,1)]:
        original=rng.integers(0,2,(rows,columns)).astype(bool)
        mask=np.repeat(np.repeat(original,3,axis=0),2,axis=1)
        r,c,small=compress_pair_relation(mask)
        assert np.array_equal(mask,small[r[:,None],c[None,:]])
        assert small.shape[0]<=rows and small.shape[1]<=columns


def test_actual_cp_pair_encoding_keeps_permutations_and_partial_quotients():
    from ortools.sat.python import cp_model
    rng=np.random.default_rng(110448)
    raw=rng.integers(0,2,(3,4)).astype(bool)
    for mask in [raw,np.repeat(raw,2,axis=0),np.repeat(raw,2,axis=1)]:
        allowed=np.column_stack(np.nonzero(mask))
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                model=cp_model.CpModel()
                a=model.new_int_var(0,mask.shape[0]-1,'a')
                b=model.new_int_var(0,mask.shape[1]-1,'b')
                add_pair_constraint(model,a,b,allowed,*mask.shape)
                model.add(a==i);model.add(b==j)
                solver=cp_model.CpSolver();solver.parameters.num_search_workers=1
                status=solver.solve(model)
                assert (status in(cp_model.OPTIMAL,cp_model.FEASIBLE))==bool(mask[i,j])


def test_free_center_benders_loop_rejects_incumbent_below_its_radius(tmp_path):
    import json
    fixture = Path(__file__).parent/'fixtures/packing_valid_20261104.json'
    p = np.array([[d['x'], d['y'], d['theta']] for d in json.loads(fixture.read_text())])
    domains = [row[None,:] for row in p]
    report = solve(domains, 2.94852695, tmp_path/'below.json', seconds=5, tables=[])
    assert report['candidates'] == 1
    assert report['cuts']['3'] == 1
    assert report['last_status'] == 'INFEASIBLE'
    assert report['finite_exhausted']
    assert not report['valid']


def test_additive_support_cut_is_lower_bound_under_domain_recombinations():
    import json
    fixture = Path(__file__).parent/'fixtures/packing_valid_20261104.json'
    base = np.array([[d['x'],d['y'],d['theta']]for d in json.loads(fixture.read_text())])
    rng=np.random.default_rng(110444)
    variants=np.round(base[None]+rng.integers(-100,101,(6,15,3))*1e-6,6)
    variants[0]=base
    domains=[variants[:,i]for i in range(15)]
    circle=compute_mec_fast(base)
    cut=categorical_support_cut(domains,base,circle)
    assert abs(cut['lower_bound']-circle[2])<1e-10
    for _ in range(32):
        selected=rng.integers(0,6,15)
        p=np.array([d[k]for d,k in zip(domains,selected)])
        lower=(sum(v[selected[i]]for i,v in cut['coefficients'].items())-cut['offset'])/cut['scale']
        assert lower<=compute_mec_fast(p)[2]+1e-11


def test_additive_bound_handles_translation_and_proves_singleton_model(tmp_path):
    import json
    fixture = Path(__file__).parent/'fixtures/packing_valid_20261104.json'
    base=np.array([[d['x'],d['y'],d['theta']]for d in json.loads(fixture.read_text())])
    variants=np.repeat(base[None],3,axis=0)
    variants[1,:,:2]+=[.1,-.17]
    variants[2,:,:2]+=[-.13,.04]
    domains=[variants[:,i]for i in range(15)]
    cut=categorical_support_cut(domains,base,compute_mec_fast(base))
    for choice in range(3):
        lower=(sum(v[choice]for v in cut['coefficients'].values())-cut['offset'])/cut['scale']
        assert lower<=compute_mec_fast(variants[choice])[2]+1e-11
    singleton=[row[None,:]for row in base]
    report=solve(singleton,2.94852695,tmp_path/'support.json',seconds=5,tables=[],support_cuts=True)
    assert report['candidates']==0 and report['support_cuts']==1
    assert report['finite_exhausted'] and not report['valid']
