import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ortools")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from experiment_support_completion import pinned_domains, relative_constraints


def test_support_pins_allow_only_common_translation():
    p = np.arange(15., dtype=float).reshape(5, 3)/10
    matrix, values = relative_constraints(p, [0, 2, 4])
    shifted = p.copy()
    shifted[:, :2] += [2., -3.]
    np.testing.assert_allclose(matrix@np.r_[shifted.ravel(), 3.], values, atol=1e-14)
    shifted[2, 0] += 1e-6
    assert np.max(abs(matrix@np.r_[shifted.ravel(), 3.]-values)) > .9e-6


def test_pinned_domains_retain_support_angles_but_fix_xy():
    p = np.array([[.1234562, .3456781, .2], [.4, .5, .6]])
    fixed = np.round(p, 6)
    domains = pinned_domains(p, fixed, [0], angle_width=3)
    assert len(domains[0]) == 7
    assert np.all(domains[0][:, :2] == fixed[0, :2])
    assert len(np.unique(domains[1][:, :2], axis=0)) > 1
