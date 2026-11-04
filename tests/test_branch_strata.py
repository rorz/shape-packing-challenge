from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

pytest.importorskip("fpylll")
pytest.importorskip("ortools")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from experiment_branch_strata import (
    EqualCoverage, LEADER, Manifold, NativeBabai, choose_angle, make_frame,
    reflected, select_stratum,
)
from experiment_mirror_family import FLATS, cells
from semicircle_packing.polish import load


def origin():
    return load(Path(__file__).parent / "fixtures/recovery/mirror-continuous.json")


def test_round_robin_branch_exposure_never_depends_on_cost():
    scheduler = EqualCoverage(17, 91)
    sequence = []
    for _ in range(150):
        sequence.append(scheduler.next())
        assert scheduler.visits.max() - scheduler.visits.min() <= 1
    again = EqualCoverage(17, 91)
    assert sequence == [again.next() for _ in range(150)]
    assert len({i for i, _ in sequence[:17]}) == 17


def test_stratum_uses_constrained_stress_and_exact_half_open_edges():
    rows = [{"first_raw": float(i), "second_raw": 1., "cost_nm": cost,
             "feasible_positive_stress": False, "feasible_positive_tied_stress": True}
            for i, cost in enumerate([19.999, 20., 49.999, 50., 100.])]
    assert [r["cost_nm"] for r in select_stratum(rows, 20, 50)] == [20., 49.999]
    rows[1]["feasible_positive_tied_stress"] = False
    assert [r["cost_nm"] for r in select_stratum(rows, 20, 50)] == [49.999]


def test_reflected_sections_have_negated_branch_and_matching_radius():
    p = origin(); branch = np.array([3.141562, 3.141589])
    a, meta_a = Manifold(p).section(branch, [0., 0.])
    b, meta_b = Manifold(reflected(p)).section(-branch, [0., 0.])
    assert meta_a["radius"] == pytest.approx(meta_b["radius"], abs=2e-11)
    np.testing.assert_allclose(a[:, :2], reflected(b)[:, :2], atol=2e-9)
    assert meta_a["min_stress"] < 0  # the obsolete full-J rule would reject it


def test_expensive_negative_full_j_branch_builds_positive_tied_frame():
    model = Manifold(origin()); branch = np.array([3.141562, 3.141589])
    physical, _ = model.section(branch, [0., 0.])
    rng = np.random.default_rng(92)
    angle, mode = choose_angle(physical, branch, 0., rng, local=False)
    assert mode == "global"
    settings = SimpleNamespace(target=LEADER+2e-8, regularization=1e-5, block_size=0,
                               concentration=1., boundary_bias=1., movement=50.)
    frame, posed, section, _ = make_frame(model, branch, [0., 0.], angle, rng, settings)
    assert section["min_stress"] < 0 < frame.weights.min()
    np.testing.assert_allclose([posed[i, 2]-posed[j, 2] for i, j in FLATS], branch, atol=2e-10)
    actual, units, coefficients, ambient, _ = frame.sample(rng, 16, settings, NativeBabai())
    expected = np.array([frame.M.babai(v.tolist()) for v in ambient])
    np.testing.assert_array_equal(coefficients, expected)
    for coefficients_row in coefficients:
        raw = frame.raw(coefficients_row)
        np.testing.assert_allclose([raw[i, 2]-raw[j, 2] for i, j in FLATS], branch, atol=2e-10)
