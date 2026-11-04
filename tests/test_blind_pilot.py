"""The blind comparison must grade independently and preserve its decision rule."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fpylll")
pytest.importorskip("ortools")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_blind_pilot import center_keys, decide, grade_arm
from semicircle_packing.polish import checked, load, save


def test_center_novelty_quotients_labels_reflections_and_grid_translation():
    p = np.array([[.1, .2, 0], [.3, .8, 1], [-.4, .7, 2]])
    transformed = p[::-1].copy()
    transformed[:, :2] = transformed[:, :2][:, ::-1] * [-1, 1] + [3, -2]
    assert center_keys(p)[0] != center_keys(transformed)[0]
    assert center_keys(p)[1] == center_keys(transformed)[1]
    transformed[0, 0] += 1e-6
    assert center_keys(p)[1] != center_keys(transformed)[1]


def arm(seconds, score=3.):
    return {"first_qualifying_seconds": seconds, "censored_seconds": seconds if seconds is not None else 120,
            "best_score": score}


def test_gate_needs_both_latency_advantage_and_no_score_regression():
    pairs = [{"native": arm(10), "hybrid": arm(5)} for _ in range(3)]
    assert decide(pairs)["decision"] == "GO"
    pairs[0]["hybrid"]["best_score"] += 2e-11
    assert decide(pairs)["decision"] == "NO_GO"
    pairs[0]["hybrid"]["best_score"] = 3.
    pairs[1]["hybrid"] = arm(None)
    assert decide(pairs)["decision"] == "NO_GO"


def test_independent_grader_rejects_score_guard_and_deadline_miscredit(tmp_path):
    fixture = Path(__file__).parent / "fixtures/packing_valid_20261104.json"
    p = load(fixture)
    score = checked(p).score
    label, _ = center_keys(p)
    save(tmp_path / "valid.json", p)
    report = {
        "score_cap": score, "method": "native", "problem_id": "test",
        "valid_records": [{"file": "valid.json", "score": score, "seconds": 1,
                           "center_hash": label, "qualifying": True}],
        "counts": {}, "exact_seconds": 0, "exact_budget_overrun": 0,
        "solver_records": [], "seconds": 2, "budget_seconds": 2,
    }
    audit = {"blocked_reads": [], "network_attempts": 0}
    (tmp_path / "access-audit.json").write_text(json.dumps(audit))
    (tmp_path / "report.json").write_text(json.dumps(report))
    assert grade_arm(tmp_path, (set(), set()), 2, score)["qualifying_centers"] == 1
    report["score_cap"] -= 1.5e-11  # Neither double the benchmark guard nor add the solver's guard.
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(AssertionError, match="qualification"):
        grade_arm(tmp_path, (set(), set()), 2, report["score_cap"])
    report["score_cap"] = score
    report["valid_records"][0]["seconds"] = 2.01
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(AssertionError, match="late"):
        grade_arm(tmp_path, (set(), set()), 2, score)
