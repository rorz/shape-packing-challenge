"""Cross-check native enclosing-circle certificates against an independent solve."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from breakout_geometry import evaluate_batch
from semicircle_packing.exact_area import polygon_intersection_areas
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.polish import load

ROOT=Path(__file__).resolve().parents[1]

def test_mixed_endpoint_arc_supports_match_convex_mec():
    p=load(ROOT/'solution.json');rng=np.random.default_rng(110450)
    candidates=p+rng.normal(0,.0002,(70,15,3));candidates[0]=p
    areas,circles,ok=evaluate_batch(candidates,compute_mec_fast(p)[:2])
    assert ok.sum()>60
    for q,aa,cc in zip(candidates,areas,circles):
        assert cc[2]==pytest.approx(compute_mec_fast(q)[2],abs=5e-11)
        np.testing.assert_array_equal(aa,polygon_intersection_areas(q))

def test_empty_support_selection_falls_back_without_rejecting():
    p=load(ROOT/'solution.json');p[3,0]+=.0001
    _,circles,ok=evaluate_batch(p[None],compute_mec_fast(load(ROOT/'solution.json'))[:2],window=1e-14)
    assert ok[0]==0
    assert circles[0,2]==pytest.approx(compute_mec_fast(p)[2],abs=5e-11)

def test_rotation_and_translation_preserve_radius():
    p=load(ROOT/'solution.json');c=np.array(compute_mec_fast(p)[:2]);expected=compute_mec_fast(p)[2]
    shapes=[];centers=[]
    for angle,shift in [(0,np.array([91.,-87.])),(.813,np.array([0.,0.])),(-2.2,np.array([-8.,19.]))]:
        co,si=np.cos(angle),np.sin(angle);rotation=np.array([[co,si],[-si,co]])
        q=p.copy();q[:,:2]=q[:,:2]@rotation+shift;q[:,2]+=angle
        shapes.append(q);centers.append(c@rotation+shift)
    _,circles,ok=evaluate_batch(shapes,centers)
    np.testing.assert_allclose(circles[:,2],expected,atol=8e-12,rtol=0)

def test_small_arbitrary_instances_use_certificate_or_fallback():
    rng=np.random.default_rng(110454)
    for count in (1,2,4):
        shapes=rng.uniform(-2,2,(6,count,3))
        _,circles,_=evaluate_batch(shapes,window=4)
        for q,c in zip(shapes,circles):assert c[2]==pytest.approx(compute_mec_fast(q)[2],abs=1e-9)
