"""Cross-check the search MEC against the independent reference algorithm."""

import numpy as np
import pytest

from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.geometry import Semicircle
from semicircle_packing.scoring import compute_mec


@pytest.mark.parametrize("seed", range(5))
def test_fast_mec_matches_reference_on_irregular_packings(seed):
    rng = np.random.default_rng(seed)
    placements = np.column_stack((rng.normal(0, 1.5, (15, 2)), rng.uniform(-np.pi, np.pi, 15)))
    placements[:, :2] += [37.2, -11.8]
    expected = compute_mec([Semicircle(*p) for p in placements])
    actual = compute_mec_fast(placements)
    assert actual[2] == pytest.approx(expected[2], abs=1e-10)


@pytest.mark.parametrize("placements", [
    np.array([[0., 0., 0.]]),
    np.array([[0., 0., 0.], [0., 0., np.pi]]),
    np.array([[-1., 0., 0.], [1., 0., np.pi]]),
])
def test_fast_mec_degenerate_packings(placements):
    expected = compute_mec([Semicircle(*p) for p in placements])
    actual = compute_mec_fast(placements)
    assert actual[2] == pytest.approx(expected[2], abs=1e-10)
