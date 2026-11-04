"""Cold-audit counterexamples, checked against independent valid geometry."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.angular_rounder import round_angles
from scripts.experiment_hillclimb import boundary_radii, dual_weights, support_lower_bounds, exact_overlap_free
from semicircle_packing.geometry import Semicircle, semicircle_polygon
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.polish import checked


def incumbent():
    data=json.loads((Path(__file__).parent/'fixtures/packing_valid_20261104.json').read_text())
    return np.array([[q[k]for k in('x','y','theta')]for q in data])


@pytest.mark.parametrize('fraction',[.4,.6,.8,.9])
def test_free_center_rounding_recovers_known_valid_member(fraction):
    p=incumbent();cx,cy,radius=compute_mec_fast(p);query=p.copy();query[14,2]+=fraction*1e-6
    bad_center=compute_mec_fast(query)[:2]
    fixed,info=round_angles(query,radius+1e-9,center=bad_center,return_info=True)
    assert fixed is None
    assert info['status']=='fixed_center_unsat'
    result,info=round_angles(query,radius+1e-9,return_info=True)
    assert result is not None
    assert info['recovery_used']
    assert checked(result).valid
    assert checked(result).score<=radius+1e-9


def test_unsuccessful_center_search_does_not_claim_global_unsat():
    p=np.array([[0.,0.,.2e-6],[0.,0.,.4e-6]])
    result,info=round_angles(p,1.,return_info=True)
    assert result is None
    assert not info['fixed_center']
    assert info['status']=='no_assignment_at_tested_centers'


def test_old_fixed_distance_screen_overstates_equivalent_packing_radius():
    base=incumbent();cx,cy,radius=compute_mec_fast(base);center=np.array([cx,cy])
    active,dual=dual_weights(base,center,radius)
    shifted=base.copy();shifted[:,:2]+=[.001,-.002]
    true_radius=compute_mec_fast(shifted)[2]
    old=float((boundary_radii(shifted[None],center,active)@dual).max())
    corrected=float(support_lower_bounds(shifted[None],base,center,active,dual)[0])
    assert old>true_radius+1e-9
    assert corrected<=true_radius+1e-12
    assert corrected>=true_radius-3e-12


def test_support_bound_remains_below_mec_after_independent_grid_changes():
    base=incumbent();cx,cy,radius=compute_mec_fast(base);center=np.array([cx,cy]);active,dual=dual_weights(base,center,radius)
    rng=np.random.default_rng(202611042)
    trials=base[None]+rng.integers(-3,4,(48,15,3))*1e-6
    lower=support_lower_bounds(trials,base,center,active,dual)
    actual=np.array([compute_mec_fast(p)[2]for p in trials])
    assert np.all(lower<=actual+1e-12)


def test_dual_bound_handles_a_two_point_supported_semicircle():
    p=np.array([[0.,0.,0.]])
    active,dual=dual_weights(p,np.zeros(2),1.)
    bound=support_lower_bounds(p[None],p,np.zeros(2),active,dual)
    assert bound[0]==pytest.approx(1.,abs=2e-12)


def test_cached_exact_pair_checks_match_full_validator():
    p=incumbent();polygons=[semicircle_polygon(Semicircle(*q))for q in p];cache={}
    translated=p.copy();translated[:,:2]+=[1e-6,-2e-6]
    bad=p.copy();bad[0,0]+=.01
    for candidate in (p,translated,bad):
        assert exact_overlap_free(candidate,p,polygons,cache)==checked(candidate).valid
