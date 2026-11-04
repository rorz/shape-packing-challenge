import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fpylll")
pytest.importorskip("ortools")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from experiment_rounding_events import eligible_native_indices, locate_transition, merge_domains, pop_balanced
from translation_union import domain_digest, make_translation_domains


def test_transition_records_actual_different_domains_and_bracket():
    def evaluate(p):
        physical = np.array([[p[0] * 1e-6, .37e-6, .1]])
        domains = make_translation_domains(physical, 1, 1)
        return {"parameters": p.tolist(), "domains": domains, "hash": domain_digest(domains)}
    result = locate_transition(evaluate, np.array([.37]), np.array([1.]), iterations=16)
    assert result is not None
    assert result["before"]["hash"] != result["after"]["hash"]
    assert result["low"] <= .63 <= result["high"]
    assert result["high"] - result["low"] < 2e-5
    assert result["evaluations"] < 30


def test_endpoint_union_preserves_both_sides_and_right_hint():
    left = [np.array([[0., 0., .1], [1e-6, 0., .1]])]
    right = [np.array([[2e-6, 0., .1], [1e-6, 0., .1]])]
    union = merge_domains(left, right)
    assert len(union[0]) == 3
    np.testing.assert_array_equal(union[0][0], right[0][0])
    assert domain_digest(union) == domain_digest([np.vstack([left[0], right[0]])])
    with pytest.raises(ValueError, match="max_poses"):
        merge_domains(left, right, max_poses=2)


def test_event_locator_bounds_work_when_no_change_or_deadline():
    calls = []
    def evaluate(p):
        calls.append(p.copy())
        return {"hash": "same"}
    assert locate_transition(evaluate, np.zeros(1), np.ones(1), maximum=.2) is None
    assert len(calls) == 4
    calls.clear()
    assert locate_transition(evaluate, np.zeros(1), np.ones(1), deadline=time.monotonic()-1) is None
    assert len(calls) == 1


def test_event_locator_rejects_invalid_rays():
    with pytest.raises(ValueError):
        locate_transition(lambda p: {"hash": "x"}, np.zeros(1), np.zeros(1))
    with pytest.raises(ValueError):
        locate_transition(lambda p: {"hash": "x"}, np.zeros(1), np.ones(2))


def test_event_locator_abandons_invalid_continuous_section():
    def evaluate(p):
        return {"hash": "x"} if p[0] == 0 else None
    assert locate_transition(evaluate, np.zeros(1), np.ones(1)) is None


def test_origin_quotas_prevent_priority_starvation():
    import heapq
    queue = [(0., 0, {"origin": 2}), (100., 1, {"origin": 0}),
             (200., 2, {"origin": 1}), (1., 3, {"origin": 2})]
    heapq.heapify(queue)
    assert pop_balanced(queue, 0)[2]["origin"] == 0
    assert pop_balanced(queue, 1)[2]["origin"] == 1
    assert pop_balanced(queue, 2)[1] == 0
    assert pop_balanced(queue, 0)[1] == 3  # Empty origin falls back to live work.


def test_native_eligibility_keeps_all_tied_or_nonwinning_valid_candidates():
    areas = np.array([[.9e-6, .9e-6], [1e-6 + 5e-12, .9e-6],
                      [1e-6 + 2e-11, .9e-6], [.9e-6, .9e-6]])
    circles = np.array([[0., 0., 2.9], [0., 0., 2.9],
                        [0., 0., 2.8], [0., 0., 3.1]])
    assert eligible_native_indices(areas, circles, 2.9).tolist() == [0, 1]
