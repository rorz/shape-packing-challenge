"""Gauge, coverage and rotated support checks for the genuinely new phase pool."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from experiment_phase_pool import phase_domains
from experiment_conflict_master import categorical_support_cut
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.geometry import canonicalize_theta


def fixture():
    data=json.loads((Path(__file__).parent/'fixtures/packing_valid_20261104.json').read_text())
    return np.array([[d['x'],d['y'],d['theta']]for d in data])


def test_physical_phases_have_unchanged_radius_and_grid_coverage():
    base=fixture()
    domains,anchors,circles,moves,angles,selected=phase_domains(base,phases=3,radius=50e-6,
        xy_steps=1,jitter=1,max_domain=100)
    assert np.all(moves[:,0,:2]==0)
    assert np.array_equal(anchors[:,0,:2],np.broadcast_to(base[0,:2],(3,2)))
    assert len(selected)==1
    for anchor,circle in zip(anchors,circles):
        actual=compute_mec_fast(anchor)
        assert abs(actual[2]-circles[1,2])<1e-10
        assert np.max(np.abs(np.array(actual[:2])-circle[:2]))<1e-8
        grid=np.round(anchor,6);grid[:,2]=[canonicalize_theta(t)for t in grid[:,2]]
        for i,p in enumerate(grid):assert np.any(np.all(domains[i]==p,axis=1))


def test_rotated_support_anchors_bound_all_actual_domain_choices():
    base=fixture();rng=np.random.default_rng(110452)
    domains,anchors,circles,*_=phase_domains(base,phases=3,radius=50e-6,xy_steps=1,jitter=1,max_domain=100)
    cuts=[categorical_support_cut(domains,p,c)for p,c in zip(anchors,circles)]
    for _ in range(24):
        choice=[rng.integers(len(d))for d in domains]
        p=np.array([d[k]for d,k in zip(domains,choice)])
        radius=compute_mec_fast(p)[2]
        for cut in cuts:
            bound=(sum(v[choice[i]]for i,v in cut['coefficients'].items())-cut['offset'])/cut['scale']
            assert bound<=radius+1e-11
