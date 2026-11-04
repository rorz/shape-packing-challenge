import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ortools")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from experiment_archive_orbits import pose_sets, proof_covers


def test_proof_requires_superset_and_equal_or_looser_cap():
    proved = pose_sets([np.array([[0., 0., .1], [0., 0., .100001]])])
    proposed = pose_sets([np.array([[0., 0., .1]])])
    assert proof_covers(proved, 3., proposed, 3.)
    assert proof_covers(proved, 3., proposed, 2.9)
    assert not proof_covers(proved, 2.9, proposed, 3.)
    assert not proof_covers(proposed, 3., proved, 3.)


def test_center_identity_does_not_suppress_different_angles():
    first = pose_sets([np.array([[0., 0., .1]])])
    changed = pose_sets([np.array([[0., 0., .100001]])])
    assert not proof_covers(first, 3., changed, 3.)
    assert not proof_covers(first, 3., first + first, 3.)
