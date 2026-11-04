"""Equal scores are not enough to merge physical packing configurations."""
from pathlib import Path
import numpy as np
from scripts.experiment_basins import congruent_packings
from semicircle_packing.polish import load

def test_archive_deduplicates_actual_rigid_equivalences():
    p=load(Path(__file__).parent/'fixtures/packing_valid_20261104.json')
    q=p[np.random.default_rng(110464).permutation(15)].copy();q[:,1]*=-1;q[:,2]*=-1
    t=2.1;co,si=np.cos(t),np.sin(t);q[:,:2]=q[:,:2]@np.array([[co,si],[-si,co]])+[17.,-31.];q[:,2]+=t
    assert congruent_packings(p,q)

def test_archive_keeps_equal_radius_noncongruent_interior_positions():
    # Identical fixed outer pieces determine either enclosing radius; moving an
    # interior piece changes geometry while leaving that radius unchanged.
    p=np.array([[-4.,0.,0.],[4.,0.,np.pi],[0.,0.,.2]])
    q=p.copy();q[2,0]+=.3
    assert not congruent_packings(p,q)

def test_archive_checks_orientations_even_when_centers_match():
    p=np.array([[-4.,0.,0.],[4.,0.,np.pi],[0.,0.,.2]])
    q=p.copy();q[2,2]+=.3
    assert not congruent_packings(p,q)
