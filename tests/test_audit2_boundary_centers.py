import json
from pathlib import Path

import numpy as np
import pytest

from scripts.audit2_boundary_centers import boundary_centers
from semicircle_packing.fast_mec import compute_mec_fast


def base():
    data=json.loads((Path(__file__).parent/'fixtures/packing_valid_20261104.json').read_text())
    return np.array([[q[k]for k in('x','y','theta')]for q in data])


def test_detects_endpoint_generically_and_recovers_known_center_family():
    p=base();radius=compute_mec_fast(p)[2];p[14,2]+=.9e-6
    records,info=boundary_centers(p,target=radius+1e-9)
    assert info['endpoint_indices']==[14]
    assert info['complete_geometry']
    assert any(q['radius']<=radius+1e-9 for q in records)
    assert all(q['verified_invariant']for q in records)


def test_piece_order_is_irrelevant_to_boundary_detection():
    p=base();p[14,2]+=.9e-6;rng=np.random.default_rng(202611045);order=rng.permutation(15)
    original,first=boundary_centers(p)
    permuted,second=boundary_centers(p[order])
    assert second['endpoint_indices']==[int(np.flatnonzero(order==14)[0])]
    assert sorted(q['radius']for q in original)==pytest.approx(sorted(q['radius']for q in permuted),abs=2e-12)
    assert first['complete_geometry'] and second['complete_geometry']


def test_supports_wider_angle_domains_and_reports_its_limit():
    p=base();choices=[np.array([q[2]])for q in p];choices[14]=p[14,2]+np.arange(-8,9)*1e-6
    records,info=boundary_centers(p,choices,max_combinations=2)
    assert info['total_combinations']==17
    assert info['truncated']
    assert not info['complete_geometry']
