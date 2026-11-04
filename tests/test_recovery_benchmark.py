"""Coverage diagnostics must retain held-out separation and real grid geometry."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fpylll")
pytest.importorskip("ortools")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from benchmark_recovery import (
    DEFAULT_MANIFEST, calibrated_profile, fit_frame, membership,
    required_widths, run_benchmark, scored_once, transform, verify_fixture,
)
from experiment_lazy_phases import make_domains
from experiment_mirror_family import FLATS, Manifold
from semicircle_packing.polish import checked


def test_calibration_never_widens_from_heldout_targets():
    cases = [
        {"split": "calibration", "family": {"required_widths": {"xy": 2, "angle": 3}}},
        {"split": "heldout", "family": {"required_widths": {"xy": 999, "angle": 999}}},
    ]
    assert calibrated_profile(cases, {"xy": 1, "angle": 12}) == {"xy": 2, "angle": 12}
    with pytest.raises(ValueError, match="calibration"):
        calibrated_profile(cases[1:], {"xy": 1, "angle": 12})


def test_membership_does_not_repair_or_inject_a_missing_target():
    base = np.array([[0., 0., .2], [3., 0., -.3]])
    domains = make_domains(base, 1, 2)
    snapshots = [domain.copy() for domain in domains]
    target = base.copy()
    target[1, 0] += 2e-6
    assert membership(domains, target)["missing_pieces"] == [1]
    assert required_widths(base, target)["xy"] == 2
    for before, after in zip(snapshots, domains):
        assert np.array_equal(before, after)


def test_required_angular_width_matches_production_wrap_boundary():
    base = np.array([[0., 0., 3.141592]])
    target = scored_once([[0., 0., 3.141593]])
    assert target[0, 2] == -3.141592
    assert required_widths(base, target)["angle"] == 1
    assert membership(make_domains(base, 0, 1), target)["covered"]
    assert not membership(make_domains(base, 0, 0), target)["covered"]


def test_known_incumbent_needs_physical_modes_before_small_box_covers_it():
    manifest = json.loads(DEFAULT_MANIFEST.read_text())
    folder = DEFAULT_MANIFEST.parent
    seed = verify_fixture(folder, manifest["seed"])
    case = manifest["cases"][0]
    raw = verify_fixture(folder, case, "target")
    assert checked(raw).valid
    target = scored_once(raw)
    model = Manifold(seed)
    branch = [round(float(target[i, 2] - target[j, 2]), 6) for i, j in FLATS]
    section, _ = model.section(branch, [0, 0])
    rigid, _ = fit_frame(section, target)
    family, metadata = fit_frame(model.p, target, model=model, branch=branch)
    assert metadata["success"]
    assert metadata["section"]["violation"] < 1e-9
    assert not membership(make_domains(scored_once(rigid), 1, 12), target)["covered"]
    assert membership(make_domains(scored_once(family), 1, 12), target)["covered"]


def test_rigid_fit_is_invariant_to_common_grid_translation():
    manifest = json.loads(DEFAULT_MANIFEST.read_text())
    seed = verify_fixture(DEFAULT_MANIFEST.parent, manifest["seed"])
    target = transform(seed, .71, [.012345, -.098765])
    fitted, info = fit_frame(seed, target)
    shifted = target.copy()
    shifted[:, :2] += [.003, -.002]
    fitted_shift, info_shift = fit_frame(seed, shifted)
    assert info["rms_grid_units"] < 1e-6
    assert info_shift["rms_grid_units"] < 1e-6
    assert np.allclose(fitted_shift[:, :2] - fitted[:, :2], [.003, -.002], atol=1e-12, rtol=0)


def test_fixture_hash_check_catches_changed_controls(tmp_path):
    path = tmp_path / "wrong.json"
    path.write_text("[]\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_fixture(tmp_path, {"path": path.name, "sha256": "0" * 64})


def test_domain_limit_is_checked_before_any_domain_allocation(tmp_path, monkeypatch):
    import experiment_lazy_phases

    def forbidden(*args, **kwargs):
        pytest.fail("Allocated an oversized domain before enforcing the cap")

    monkeypatch.setattr(experiment_lazy_phases, "make_domains", forbidden)
    output = tmp_path / "oversized"
    with pytest.raises(ValueError, match="exceeding --max-poses"):
        run_benchmark(DEFAULT_MANIFEST, output, xy_width=1000, solve=False)
    assert not output.exists()
