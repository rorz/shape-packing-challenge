"""Offline feasibility must validate the selected submission and its radius."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
pytest.importorskip('ortools')
import experiment_support_bundle
import lazy_phase_solver
from semicircle_packing.polish import checked, load


FIXTURE = Path(__file__).parent / 'fixtures' / 'packing_valid_20261104.json'


def forbid_external_dependencies(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail('Offline feasibility accessed the incumbent or remote API')

    monkeypatch.setattr(lazy_phase_solver, 'load', forbidden)
    monkeypatch.setattr(lazy_phase_solver, 'check_solution', forbidden)
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize('mode,status', [
    ({'local_only': True}, 'LOCAL_FEASIBLE'),
    ({'control': True}, 'CONTROL_FEASIBLE'),
])
def test_offline_recovery_returns_reference_checked_raw_assignment(tmp_path, monkeypatch, mode, status):
    p = load(FIXTURE)
    score = checked(p).score
    forbid_external_dependencies(monkeypatch, tmp_path)
    domains = [p[i:i + 1].copy() for i in range(15)]
    # Index zero is only a generic solver hint. It must not be mistaken for
    # either the held-out assignment or an already validated control.
    bad_pose = p[0].copy()
    bad_pose[0] += 100
    domains[0] = np.stack([bad_pose, p[0]])
    output = tmp_path / 'recovery'
    result = lazy_phase_solver.solve_phase(
        domains, p, score + 1e-9, output, seconds=8, **mode)
    assert result['valid'] and result['status'] == status
    assert result['assignment'] == [1] + [0] * 14
    assert result['counts']['reference_checks'] >= 1
    assert result['best_score'] is None
    assert result['score'] <= score + 1e-9 + lazy_phase_solver.ACCEPTANCE_GUARD
    assert np.array_equal(np.asarray(result['raw_placements']), p)
    assert np.array_equal(np.asarray(result['scored_placements']), p)
    assert checked(load(output / 'solution.json')).valid
    assert json.loads((output / 'summary.json').read_text()) == result
    assert not (tmp_path / 'solution.json').exists()
    assert not (output / 'best-api.json').exists()


@pytest.mark.parametrize('mode', [{'local_only': True}, {'control': True}])
def test_pair_feasible_leaf_above_target_is_not_accepted(tmp_path, monkeypatch, mode):
    p = load(FIXTURE)
    target = checked(p).score - 1e-7
    forbid_external_dependencies(monkeypatch, tmp_path)
    # Deliberately weaken the initial model so its sole pair-feasible leaf
    # reaches the final reference check. A positive control still needs to
    # satisfy the requested enclosure radius, even with no useful initial cut.
    monkeypatch.setattr(experiment_support_bundle, 'support_bundle', lambda *args: [])
    output = tmp_path / 'too-small'
    result = lazy_phase_solver.solve_phase(
        [p[i:i + 1] for i in range(15)], p, target, output, seconds=8, **mode)
    assert not result['valid']
    assert result['status'] == 'INFEASIBLE'
    assert result['counts']['reference_checks'] >= 1
    assert not (output / 'solution.json').exists()


def test_scored_wrap_aliases_are_encoded_without_changing_geometry():
    p = np.array([[0., 0., -3.141593], [1., 0., 3.141593]])
    raw, scored = lazy_phase_solver.submission_for_geometry(p)
    assert np.array_equal(raw[:, 2], [9.424778, -9.424778])
    assert np.array_equal(scored, p)
    with pytest.raises(ValueError, match='six-decimal'):
        lazy_phase_solver.submission_for_geometry([[0.0000001, 0., 0.]])
