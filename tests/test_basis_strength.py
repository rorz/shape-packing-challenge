import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("fpylll")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from experiment_basis_strength import TARGET, ambient_targets, build_worker, pose_key, set_grid_phase
from experiment_mirror_family import FLATS, Manifold
from experiment_tied_batch import Frame
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.polish import load


def _physical():
    source = Path(__file__).parent / "fixtures/recovery/mirror-continuous.json"
    model = Manifold(load(source))
    p, _ = model.section([3.141569, 3.141593], [0., 0.])
    circle = np.asarray(compute_mec_fast(p))
    p[:, :2] -= circle[:2]
    return p, circle[2]


def test_common_targets_are_identical_after_different_basis_reductions():
    p, radius = _physical()
    settings = SimpleNamespace(target=TARGET, regularization=1e-5, block_size=0,
                               concentration=1., boundary_bias=1., movement=50.)
    first = Frame(p, radius, FLATS, settings)
    settings.block_size = 20
    second = Frame(p, radius, FLATS, settings)
    for frame in (first, second):
        set_grid_phase(frame, [.25e-6, -.25e-6], radius)
    a = ambient_targets(first, np.random.default_rng(17), 32, settings)
    b = ambient_targets(second, np.random.default_rng(17), 32, settings)
    assert np.array_equal(a, b)
    assert np.array_equal(first.anchor_units, second.anchor_units)
    assert np.array_equal(first.b0, second.b0)


def test_pose_dedup_removes_only_common_integer_translation():
    units = np.arange(45, dtype=np.int64).reshape(15, 3)
    shifted = units.copy()
    shifted[:, :2] += [17, -23]
    assert pose_key(units) == pose_key(shifted)
    shifted[3, 2] += 1
    assert pose_key(units) != pose_key(shifted)


def test_worker_rejects_uncentered_physical_geometry(tmp_path):
    p, radius = _physical()
    p[:, :2] += [.1, -.2]
    path = tmp_path / "payload.npz"
    np.savez(path, physical=p, phase=np.zeros(2), radius=radius, target=TARGET)
    with pytest.raises(ValueError, match="centered"):
        build_worker(path, tmp_path / "basis.npz", 20)
