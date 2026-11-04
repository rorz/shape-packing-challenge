import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fpylll")
pytest.importorskip("ortools")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from experiment_adaptive_frames import diversity_weights, fitness_values, normalized_geometry, pose_key, wrap_phases


def test_geometry_gauge_preserves_angle_identity_and_translation_equivalence():
    original = np.arange(45, dtype=np.int64).reshape(15, 3)
    translated = original.copy()
    translated[:, :2] += [123, -456]
    units, _ = normalized_geometry(np.stack([original, translated]))
    np.testing.assert_array_equal(units[0], units[1])
    assert pose_key(units[0]) == pose_key(units[1])
    changed = units[0].copy()
    changed[4, 2] += 1
    assert pose_key(changed) != pose_key(units[0])
    assert pose_key(changed, True) == pose_key(units[0], True)


def test_fixed_scales_give_identical_fitness_for_identical_geometry():
    areas = np.array([[1.2e-6, .8e-6], [1.2e-6, .8e-6]])
    circles = np.array([[0., 0., 2.95], [0., 0., 2.95]])
    primary, secondary = fitness_values(areas, circles, [.02, .05], 2.94)
    assert primary[0] == primary[1]
    assert secondary[0] == secondary[1]
    assert secondary[0] >= primary[0]


def test_sampling_weights_deweight_repeated_centers_without_exclusion():
    weights = diversity_weights([{"center_key": "a"}, {"center_key": "a"}, {"center_key": "b"}])
    np.testing.assert_allclose(weights, [.25, .25, .5])
    assert np.all(weights > 0)


def test_phase_wrap_is_periodic_and_physical_box_is_bounded():
    a = wrap_phases([5., -6., 2., .8, -.9])
    b = wrap_phases([5., -6., 2., 1.8, 1.1])
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(a, [3., -3., 2., -.2, .1])
