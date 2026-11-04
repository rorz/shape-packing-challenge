"""Metamorphic checks for reduced integer branch lattices."""
import sys
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip('fpylll')

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from experiment_tied_flats import wrap_cells
from experiment_tied_batch import tied_transform


def test_constrained_branch_survives_reaction_forces_and_integer_concentration():
    """A real branch rejected by the former unconstrained stress calculation."""
    from types import SimpleNamespace
    from experiment_tied_batch import Frame
    from experiment_batch_lattice import NativeBabai
    from semicircle_packing.polish import load
    from semicircle_packing.fast_mec import compute_mec_fast
    from semicircle_packing.geometry import canonicalize_theta

    p=load(Path(__file__).parent/'fixtures/mirror_constrained_branch.json')
    cx,cy,radius=compute_mec_fast(p);p[:,:2]-=[cx,cy]
    args=SimpleNamespace(target=2.94852698,regularization=1e-5,block_size=0,
                         concentration=10,boundary_bias=1.5,movement=50)
    frame=Frame(p,radius,[(1,14),(4,8)],args)
    assert min(frame.weights)>0
    _,J=frame.c.evaluate(np.r_[p.ravel(),radius])
    full=J[frame.active,:45]
    U,_,_=np.linalg.svd(full,full_matrices=True)
    old=U[:,-1]*np.sign(U[:,-1].sum())
    assert old.min()<0  # The old gate incorrectly excluded this exact section.
    _,T=tied_transform(p,frame.pairs)
    # The section is near the continuous optimum; its smallest singular
    # value is small, not an exactly zero symbolic equilibrium.
    assert np.linalg.norm(frame.weights@full@T)<1e-7
    geometry,units,coeff,ambient,_=frame.sample(np.random.default_rng(110482),32,args,NativeBabai())
    for k in range(len(coeff)):
        assert np.array_equal(coeff[k],frame.M.babai(ambient[k].tolist()))
        raw=frame.raw(coeff[k]);expected=np.round(raw,6)
        expected[:,2]=[canonicalize_theta(t)for t in expected[:,2]]
        assert np.max(abs(geometry[k]-expected))<1e-14


def test_tied_transform_preserves_each_raw_angle_difference():
    rng=np.random.default_rng(110474);p=rng.normal(size=(15,3));pairs=[(1,14),(4,8)]
    columns,T=tied_transform(p,pairs);integer=rng.integers(-100,101,size=(200,43));moves=(integer@T.T).reshape(-1,15,3)
    assert len(columns)==43
    assert np.array_equal(T[columns],np.eye(43,dtype=np.int64))
    for i,j in pairs:assert np.array_equal(moves[:,i,2],moves[:,j,2])


def test_rotation_cells_preserve_raw_branches_and_cover_all_permitted_angles():
    rng=np.random.default_rng(110475);pairs=[(1,14),(4,8)]
    for _ in range(20):
        p=np.zeros((15,3));p[:,2]=rng.uniform(-np.pi,np.pi,15)
        expected=np.array([p[i,2]-p[j,2]for i,j in pairs]);cells=wrap_cells(p,pairs)
        for phi in rng.uniform(-np.pi,np.pi,1000):
            theta=np.arctan2(np.sin(p[:,2]+phi),np.cos(p[:,2]+phi))
            actual=np.array([theta[i]-theta[j]for i,j in pairs])
            permitted=np.max(abs(actual-expected))<1e-10
            assert permitted==any(lo<phi<hi for lo,hi in cells)
