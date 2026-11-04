"""Flat-contact discovery must not depend on piece numbering or rigid frame."""
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('fpylll')
from scripts.experiment_lll import flat_contact_pairs
from semicircle_packing.polish import AreaConstraints,load
from semicircle_packing.fast_mec import compute_mec_fast

def detect(p):
    p=p.copy();cx,cy,r=compute_mec_fast(p);p[:,:2]-=[cx,cy]
    c=AreaConstraints(p);g,_=c.evaluate(np.r_[p.ravel(),r])
    return {frozenset(pair)for pair in flat_contact_pairs(p,c,g)}

def test_flat_pairs_survive_permutation_rotation_and_reflection():
    p=load(Path(__file__).parent/'fixtures/packing_valid_20261104.json')
    expected={frozenset((1,5)),frozenset((6,10))};assert detect(p)==expected
    permutation=np.random.default_rng(110460).permutation(15)
    inverse=np.argsort(permutation)
    for reflected in (False,True):
        q=p[permutation].copy()
        if reflected:q[:,1]*=-1;q[:,2]*=-1
        angle=1.783;co,si=np.cos(angle),np.sin(angle)
        q[:,:2]=q[:,:2]@np.array([[co,si],[-si,co]])+[3.,-8.];q[:,2]+=angle
        assert detect(q)=={frozenset(inverse[list(pair)])for pair in expected}
