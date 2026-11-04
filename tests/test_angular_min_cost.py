"""Independent exhaustive checks for the native angular-cost search."""
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.angular_min_cost import _solve_tables, minimize_angles
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.polish import checked


@pytest.mark.parametrize("n",range(1,9))
def test_gray_cost_matches_independent_exhaustive_search(n):
    rng=np.random.default_rng(110440+n)
    for scale in (1.,1e-14):
        unary=rng.random((n,2))*scale;pairs=rng.random((n*(n-1)//2,2,2))*scale
        assignment,info=_solve_tables(unary,pairs)
        pair_ids=list(itertools.combinations(range(n),2))
        def cost(bits):
            return sum(unary[i,bits[i]]for i in range(n))+sum(pairs[k,bits[i],bits[j]]for k,(i,j)in enumerate(pair_ids))
        expected=min(cost(bits)for bits in itertools.product((0,1),repeat=n))
        assert cost(assignment)==pytest.approx(expected,rel=1e-13,abs=scale*1e-13)
        assert info["energy"]==pytest.approx(expected,rel=1e-13,abs=scale*1e-13)


def test_feasible_assignment_has_priority_over_preference():
    assignment,info=_solve_tables(np.array([[0.,1e-20],[0.,0.]]),np.zeros((1,2,2)),np.array([[1.,0.],[1.,0.]]))
    assert assignment.tolist()==[0,1]
    assert info["energy"]==0


def test_full_fifteen_bit_search_matches_vectorized_exhaustion():
    rng=np.random.default_rng(110441);n=15
    unary=rng.random((n,2))*1e-14;pairs=rng.random((105,2,2))*1e-14
    bits=((np.arange(1<<n)[:,None]>>np.arange(n))&1).astype(int)
    costs=unary[np.arange(n)[None,:],bits].sum(axis=1)
    for k,(i,j) in enumerate(itertools.combinations(range(n),2)):
        costs+=pairs[k,bits[:,i],bits[:,j]]
    assignment,info=_solve_tables(unary,pairs)
    assert info["enumerated"]==32768
    assert info["energy"]==pytest.approx(float(costs.min()),rel=1e-13,abs=1e-26)


def test_current_valid_packing_is_recovered_without_rounding_damage():
    root=Path(__file__).parents[1];data=json.loads((root/"solution.json").read_text())
    p=np.array([[q[k]for k in("x","y","theta")]for q in data]);cx,cy,r=compute_mec_fast(p)
    candidate,info=minimize_angles(p,r+1e-9,center=[cx,cy])
    assert np.array_equal(candidate,p)
    assert info["energy"]==0
    assert checked(candidate).valid


def test_an_infeasible_angular_family_still_returns_a_quantized_proposal():
    p=np.array([[0.,0.,.2e-6],[0.,0.,.4e-6]])
    candidate,info=minimize_angles(p,1.,center=[0.,0.])
    assert candidate.shape==(2,3)
    assert np.array_equal(candidate,np.round(candidate,6))
    assert info["energy"]>0
    assert info["overlap_pairs"]==1


def test_single_shape_has_no_pair_table_edge_case():
    candidate,info=minimize_angles(np.array([[0.,0.,.2e-6]]),1.,center=[0.,0.])
    assert candidate.tolist()==[[0.,0.,0.]]
    assert info["energy"]==0
