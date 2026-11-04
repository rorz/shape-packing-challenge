import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ortools")
from ortools.sat.python import cp_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from experiment_exact_center_hops import add_center_novelty, center_hash, make_hop_domains, run_graph, solve_hop
from experiment_angle_orbits import scored_once, transform_scored
from lazy_phase_solver import submission_for_geometry
from semicircle_packing.polish import checked, load


def _status(domains, blocked):
    model = cp_model.CpModel()
    variables = [model.new_int_var(0, len(d) - 1, f"p{i}") for i, d in enumerate(domains)]
    add_center_novelty(model, variables, domains, blocked)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    return solver.solve(model)


def test_angle_variants_and_common_integer_translation_cannot_count_as_hops():
    original = np.array([[0., 0., 0.], [1., .5, .2]])
    shifted = original.copy()
    shifted[:, :2] += [.000017, -.000032]
    shifted[:, 2] += 1e-6
    assert center_hash(original) == center_hash(shifted)
    assert _status([row[None] for row in shifted], [original]) == cp_model.INFEASIBLE


def test_relative_center_move_counts_and_every_visited_state_is_excluded():
    original = np.array([[0., 0., 0.], [1., .5, .2]])
    moved = original.copy()
    moved[1, 0] += 1e-6
    domains = [original[0:1], np.vstack([original[1], moved[1]])]
    assert _status(domains, [original]) in (cp_model.FEASIBLE, cp_model.OPTIMAL)
    assert _status(domains, [original, moved]) == cp_model.INFEASIBLE


def test_quarter_turns_are_distinct_center_search_states():
    original = np.array([[0., 0., 0.], [1., .5, .2]])
    turned = transform_scored(original, np.array([[0, -1], [1, 0]]))
    assert center_hash(original) != center_hash(turned)


def test_hop_domains_preserve_special_scored_alias_and_grid():
    p = scored_once([[0., 0., 9.424778], [1., .5, -9.424778]])
    physical = transform_scored(p, np.eye(2, dtype=int))
    domains = make_hop_domains(physical, xy_width=0, angle_width=1)
    for source, domain in zip(p, domains):
        assert np.any(np.all(np.isclose(source, domain, rtol=0, atol=1e-12), axis=1))
        _, scored = submission_for_geometry(domain)
        assert np.allclose(scored, domain, rtol=0, atol=1e-12)


def test_center_hash_rejects_nongrid_coordinates():
    with pytest.raises(ValueError, match="six-decimal"):
        center_hash([[.0000001, 0., 0.], [1., 0., 0.]])


def test_real_valid_control_and_same_center_exclusion(tmp_path):
    fixture = Path(__file__).parent / "fixtures/packing_valid_20261104.json"
    raw = load(fixture)
    validation = checked(raw)
    assert validation.valid
    p = scored_once(raw)
    domains = [row[None] for row in p]
    blocked = p.copy()
    blocked[1, 0] += 1e-6
    result = solve_hop(domains, p, validation.score + 1e-10, [blocked],
                       tmp_path / "positive", seconds=3)
    assert result["valid"]
    assert result["score"] == pytest.approx(validation.score, abs=1e-12)
    excluded = solve_hop(domains, p, validation.score + 1e-10, [p],
                         tmp_path / "excluded", seconds=3)
    assert not excluded["valid"]
    assert excluded["status"] == "INFEASIBLE"


def test_graph_barriers_stay_global_and_beam_handles_numpy_pose_payloads(tmp_path):
    fixtures = Path(__file__).parent / "fixtures"
    incumbent = fixtures / "packing_valid_20261104.json"
    score = checked(load(incumbent)).score
    # The expired construction budget avoids a production solve. Three genuine
    # valid starts exercise an eviction of the non-first NumPy-bearing node.
    report = run_graph(incumbent,
                       [fixtures / "recovery/random-d.json", fixtures / "recovery/neighbor-g.json"],
                       tmp_path / "graph", seconds=.0001, beam_size=2)
    assert report["relay_caps"] == [score + 50e-9, score + 100e-9]
    assert report["distinct_valid_centers"] == 3
    assert report["queue_evictions"] == 1
    assert report["queued"] == 2
    assert not report["models"]
