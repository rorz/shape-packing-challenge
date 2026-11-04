import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ortools")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from experiment_support_branches import FLATS, branch_choices, equality_constraints


def test_six_equalities_pin_relative_supports_and_both_flat_differences():
    p = np.arange(45, dtype=float).reshape(15, 3) / 10
    supports = [5, 10, 14]
    branch = np.array([3.141569, 3.141593])
    matrix, values = equality_constraints(p, supports, branch)
    assert matrix.shape == (6, 46)
    assert np.linalg.matrix_rank(matrix) == 6
    assert np.allclose(matrix[:4] @ np.r_[p.ravel(), 3.], values[:4])
    assert np.array_equal(values[4:], branch)
    for row, (i, j) in enumerate(FLATS, start=4):
        expected = np.zeros(46)
        expected[3 * i + 2] = 1
        expected[3 * j + 2] = -1
        assert np.array_equal(matrix[row], expected)


def test_common_translation_remains_free_but_relative_support_motion_does_not():
    p = np.zeros((15, 3))
    matrix, _ = equality_constraints(p, [5, 10, 14], [3.141569, 3.141593])
    displacement = np.zeros(46)
    displacement[:45:3] = .17
    displacement[1:45:3] = -.23
    assert np.all(matrix @ displacement == 0)
    displacement[3 * 10] += 1e-6
    assert np.max(np.abs(matrix @ displacement)) == pytest.approx(1e-6)


def test_branch_schedule_starts_at_nearest_and_covers_all_nine_integer_neighbors():
    p = np.zeros((15, 3))
    p[1, 2] = 3.1415694
    p[4, 2] = 3.1415926
    choices = branch_choices(p)
    units = np.rint(np.asarray(choices) * 1e6).astype(np.int64)
    assert len(choices) == 9
    assert units[0].tolist() == [3141569, 3141593]
    assert {tuple(row) for row in units} == {
        (x, y) for x in range(3141568, 3141571) for y in range(3141592, 3141595)}
    assert np.allclose(np.asarray(choices), units / 1e6, rtol=0, atol=1e-15)
