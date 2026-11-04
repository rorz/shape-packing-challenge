"""D4 center equivalence must not erase angular-grid repair opportunities."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ortools")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from experiment_angle_orbits import d4_matrices, make_angle_domains, scored_once, transform_scored
from lazy_phase_solver import submission_for_geometry
from translation_union import domain_digest


def _units(values):
    return {tuple(v) for v in np.rint(np.asarray(values) * 1e6).astype(np.int64)}


def test_d4_preserves_exact_integer_center_distances_and_rotates_directions():
    p = scored_once([[.123456, -.314159, .15], [-1.001234, .717171, -2.4],
                     [2.456789, -1.212121, 3.141592]])
    xy = np.rint(p[:, :2] * 1e6).astype(np.int64)
    squared = ((xy[:, None] - xy[None]) ** 2).sum(axis=2)
    signatures = set()
    for _, matrix in d4_matrices():
        signatures.add(tuple(matrix.ravel()))
        transformed = transform_scored(p, matrix, (37, -82))
        result_xy = np.rint(transformed[:, :2] * 1e6).astype(np.int64)
        assert np.array_equal(((result_xy[:, None] - result_xy[None]) ** 2).sum(axis=2), squared)
        expected = np.column_stack([np.cos(p[:, 2]), np.sin(p[:, 2])]) @ matrix.T
        actual = np.column_stack([np.cos(transformed[:, 2]), np.sin(transformed[:, 2])])
        assert np.allclose(actual, expected, rtol=0, atol=4e-16)
    assert len(signatures) == 8


def test_narrow_angle_domains_nest_and_fixed_centers_do_not_move():
    p = scored_once([[.123456, -.314159, .15], [-1.001234, .717171, -2.4]])
    physical = transform_scored(p, d4_matrices()[1][1])
    narrow = make_angle_domains(physical, 1)
    wide = make_angle_domains(physical, 12)
    for a, b, row in zip(narrow, wide, physical):
        assert _units(a) <= _units(b)
        assert np.all(a[:, :2] == row[:2])


def test_identity_keeps_once_scored_exceptional_alias_available():
    p = scored_once([[0., 0., -9.424778], [1., 0., 9.424778]])
    physical = transform_scored(p, np.eye(2, dtype=int))
    domains = make_angle_domains(physical, 0)
    for original, domain in zip(p, domains):
        assert tuple(np.rint(original * 1e6).astype(int)) in _units(domain)
        raw, scored = submission_for_geometry(domain)
        assert np.array_equal(scored, domain)


def test_audited_invalid_witness_orbit_contains_the_new_valid_candidate():
    path = Path(__file__).resolve().parents[1] / "docs/pilot-center-audit.json"
    audit = json.loads(path.read_text())
    source = next(row for row in audit["matching_verified_sources"] if row["sha256"] ==
                  "98fb1979c7985f5432728e93180c3261478d0b3b757fb8513c4936aad96cbde4")
    assert audit["candidate"]["sha256"] == "cefac3368caae7f9dea0c4cdc0e87d869f3a9bd76ba75292b1cd4b7fff7c8fb9"
    raw = np.array([[v[k] for k in ("x", "y", "theta")] for v in source["raw_placements"]])
    target = scored_once([[v[k] for k in ("x", "y", "theta")] for v in audit["candidate"]["raw_placements"]])
    transform = source["transformations"][0]
    matrix = np.array([[0, 1], [-1, 0]])
    physical = transform_scored(scored_once(raw), matrix, transform["integer_translation"])
    domains = make_angle_domains(physical, 1)
    base = np.asarray([d[0] for d in domains])
    changed = np.flatnonzero(np.rint((target[:, 2] - base[:, 2]) * 1e6)).tolist()
    assert changed == [4, 6, 14]
    for wanted, domain in zip(target, domains):
        assert tuple(np.rint(wanted * 1e6).astype(np.int64)) in _units(domain)
    zero = make_angle_domains(physical, 0)
    assert any(tuple(np.rint(wanted * 1e6).astype(np.int64)) not in _units(domain)
               for wanted, domain in zip(target, zero))


def test_domain_hash_includes_angle_realization_even_when_centers_match():
    physical = np.array([[.123456, -.314159, .15]])
    first = make_angle_domains(physical, 0)
    physical[0, 2] += 1e-6
    second = make_angle_domains(physical, 0)
    assert np.array_equal(first[0][:, :2], second[0][:, :2])
    assert domain_digest(first) != domain_digest(second)


def test_non_grid_centers_and_non_orthogonal_transforms_are_rejected():
    with pytest.raises(ValueError, match="six-decimal"):
        transform_scored([[.0000001, 0., 0.]], np.eye(2, dtype=int))
    with pytest.raises(ValueError, match="orthogonal"):
        transform_scored([[0., 0., 0.]], [[1, 1], [0, 1]])
