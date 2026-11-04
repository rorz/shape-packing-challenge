"""Blind-worker input isolation, deterministic proposals, and grid identity."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fpylll")
pytest.importorskip("ortools")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import blind_packing_worker as worker
from benchmark_recovery import fit_frame, scored_once


def origin_problem():
    path = Path(__file__).parent / "fixtures/recovery/random-d-origin.json"
    values = json.loads(path.read_text())
    return {"id": "control-origin", "origin": [[p[k] for k in ("x", "y", "theta")] for p in values],
            "branch": [3.141570, -3.141592], "score_cap": 2.9485269867883153}


def test_blind_problem_rejects_target_payload_and_nonfinite_data(tmp_path):
    path = tmp_path / "problem.json"
    problem = origin_problem()
    path.write_text(json.dumps(problem))
    _, origin, branch = worker.read_problem(path)
    assert origin.shape == (15, 3) and branch.shape == (2,)
    problem["target"] = problem["origin"]
    path.write_text(json.dumps(problem))
    with pytest.raises(ValueError, match="keys must be exactly"):
        worker.read_problem(path)
    del problem["target"]
    problem["origin"][0][0] = float("nan")
    path.write_text(json.dumps(problem))
    with pytest.raises(ValueError, match="finite"):
        worker.read_problem(path)


def first_batches(stream, count):
    found = []
    for _ in range(20):
        batch = stream.next()
        if batch is not None:
            found.append(batch)
        if len(found) == count:
            return found
    pytest.fail("No usable deterministic frames")


def test_native_prefix_is_unchanged_by_fitting_own_proposal():
    problem = origin_problem()
    origin, branch = np.array(problem["origin"]), np.array(problem["branch"])
    baseline = worker.ProposalStream(origin, branch, problem["score_cap"], 3111001)
    expected = [b[3]["proposal_hash"] for b in first_batches(baseline, 2)]
    hybrid = worker.ProposalStream(origin, branch, problem["score_cap"], 3111001)
    first = first_batches(hybrid, 1)[0]
    assert len(first[1])
    fit_frame(hybrid.model.p, first[1][0], model=hybrid.model, branch=branch.tolist())
    second = first_batches(hybrid, 1)[0]
    assert expected == [first[3]["proposal_hash"], second[3]["proposal_hash"]]
    frame, actual, coefficients, _ = first
    for geometry, coefficient in zip(actual[:8], coefficients[:8]):
        assert np.array_equal(np.rint(scored_once(frame.raw(coefficient)) * 1e6).astype(np.int64),
                              np.rint(geometry * 1e6).astype(np.int64))


def test_center_identity_quotients_integer_translation_and_ignores_angles():
    p = np.array([[.1, -.2, .3], [.4, .5, -.6]])
    q = p.copy()
    q[:, :2] += [.000003, -.000002]
    q[:, 2] += .001
    assert worker.center_hash(p) == worker.center_hash(q)
    q[0, 0] += 1e-6
    assert worker.center_hash(p) != worker.center_hash(q)


def test_domain_identity_is_order_independent_but_retains_piece_boundaries():
    domains = [np.array([[0., 0., .1], [1e-6, 0., .1]]), np.array([[2., 0., .2]])]
    assert worker.domain_hash(domains) == worker.domain_hash([domains[0][::-1], domains[1]])
    assert worker.domain_hash(domains) != worker.domain_hash(domains[::-1])


def test_worker_refuses_overwrite_and_unknown_method(tmp_path):
    path = tmp_path / "problem.json"
    path.write_text(json.dumps(origin_problem()))
    output = tmp_path / "results"
    output.mkdir()
    (output / "keep.txt").write_text("prior evidence")
    with pytest.raises(ValueError, match="overwrite"):
        worker.run(path, output, seconds=.1)
    with pytest.raises(ValueError, match="Method"):
        worker.run(path, tmp_path / "unused", method="oracle", seconds=.1)


def test_queue_selection_uses_identity_not_numpy_dictionary_equality():
    first = {"geometry": np.zeros((15, 3)), "stage": 0, "serial": 0, "loss": 2.}
    second = {"geometry": np.ones((15, 3)), "stage": 0, "serial": 1, "loss": 1.}
    pending = [first, second]
    assert worker.take_candidate(pending, 0) is second
    assert len(pending) == 1 and pending[0] is first
