from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("fpylll")
pytest.importorskip("networkx")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from experiment_joint_fission import (
    Geometry, congruent, cut_partitions, detect_pairs, fuse_pairs, release_disks,
    schedule,
)
from semicircle_packing.polish import load


def mirror():
    return load(Path(__file__).parent / "fixtures/recovery/mirror-continuous.json")


def test_flat_detection_survives_relabeling():
    p = mirror()
    original = detect_pairs(p)
    permutation = np.random.default_rng(31).permutation(15)
    mapped = {frozenset((int(permutation[i]), int(permutation[j]))) for i, j in detect_pairs(p[permutation])}
    assert mapped == {frozenset(pair) for pair in original}
    macro, keep = fuse_pairs(p, original)
    assert macro.shape == (13, 3)
    assert len(keep) == 11
    np.testing.assert_allclose(macro[-2:, :2], [p[list(pair), :2].mean(axis=0) for pair in original])
    with pytest.raises(ValueError, match="disjoint"):
        fuse_pairs(p, [(0, 1), (1, 2)])


def test_both_zero_stagger_pairs_have_exact_disk_support():
    macro, _ = fuse_pairs(mirror(), detect_pairs(mirror()))
    cuts = [.273, 2.413]
    released = release_disks(macro, cuts, [0., 0.])
    directions = np.column_stack([np.cos(np.linspace(0, 2*np.pi, 100)), np.sin(np.linspace(0, 2*np.pi, 100))])
    for k in range(2):
        pair = released[11 + 2*k:13 + 2*k]
        supports = []
        for row in pair:
            normal = np.array([np.cos(row[2]), np.sin(row[2])])
            tangent = np.array([-normal[1], normal[0]])
            supports.append(directions @ row[:2] + np.where(directions @ normal >= 0, 1., np.abs(directions @ tangent)))
        np.testing.assert_allclose(np.maximum(*supports), directions @ macro[11+k, :2] + 1., atol=1e-14)
    shifted = release_disks(macro, cuts, [.3, -.3])
    assert np.linalg.norm(shifted[11, :2] - shifted[12, :2]) == pytest.approx(.3)
    assert np.linalg.norm(shifted[13, :2] - shifted[14, :2]) == pytest.approx(.3)


def test_congruence_preserves_normals_and_both_reflections():
    p = mirror(); theta = .371
    rotation = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    q = p.copy(); q[:, 1] *= -1; q[:, 2] *= -1
    q[:, :2] = q[:, :2] @ rotation + [3., -8.]
    q[:, 2] += theta
    q = q[np.random.default_rng(32).permutation(15)]
    assert congruent(p, q)
    q[0, 2] += .05
    assert not congruent(p, q)


def test_joint_schedule_covers_both_macros_and_four_halves():
    macro = {"cuts": [list(range(8)), list(range(8))]}
    jobs = schedule([macro, macro], 77)
    assert len(jobs) == 1152
    assert [i for i, _ in jobs[:10]] == [0, 1] * 5
    assert len(set(jobs)) == 1152
    assert jobs == schedule([macro, macro], 77)
    assert jobs != schedule([macro, macro], 78)
    assert any(move[2] != 0 and move[3] != 0 for _, move in jobs)


def test_cut_partition_cap_and_mixed_two_disk_derivatives():
    p, _ = fuse_pairs(mirror(), detect_pairs(mirror()))
    cuts, audit = cut_partitions(p, 3.15, 11)
    assert len(cuts) == 8 and len(set(cuts)) == 8
    assert all(0 <= cut < np.pi for cut in cuts)
    assert audit["partitions_omitted"] >= 0
    # Use a generic configuration away from contact switches for Jacobian QA.
    p = np.random.default_rng(88).normal(size=(13, 3))
    geometry = Geometry(np.ones(13), np.r_[np.zeros(11, dtype=int), 1, 1])
    z = np.r_[p.ravel(), 3.4]
    _, jacobian = geometry.evaluate(z)
    jacobian = jacobian.copy()
    errors = []
    for k in range(len(z)):
        d = np.zeros(len(z)); d[k] = 1e-7
        finite = (geometry.fun(z+d) - geometry.fun(z-d)) / 2e-7
        errors.append(np.max(np.abs(finite - jacobian[:, k])))
    assert max(errors) < 2e-7
