from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("fpylll")
pytest.importorskip("ortools")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from experiment_physical_frame_pool import (
    all_frame_supports, assignment_sources, fit_sources, merge_domains,
    scored_once, support_module, use_supports,
)
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.polish import load


def inputs():
    root = Path(__file__).parent / "fixtures/recovery"
    return load(root / "mirror-continuous.json"), [load(root / name) for name in ("incumbent.json", "random-d.json", "fresh-baseline.json")]


def test_alignment_preserves_physical_deformations_and_flat_geometry():
    seed, raw = inputs()
    aligned, fits, common = fit_sources(seed, raw)
    assert any(r["maximum_labeled_pair_distance_change_from_current"] > 1e-5 for r in fits[1:])
    assert len({tuple(r["fit"]["alpha"]) for r in fits}) == 3
    for p, fit in zip(aligned, fits):
        assert compute_mec_fast(p)[2] == pytest.approx(fit["section"]["radius"], abs=2e-9)
        delta = np.array(fit["aligned_flat_differences"]) - fit["source_scored_flat_differences"]
        np.testing.assert_allclose(np.arctan2(np.sin(delta), np.cos(delta)), 0., atol=2e-10)
    assert common["angle"] == fits[0]["fit"]["angle"]


def test_mixed_union_keeps_incumbent_and_every_source_without_truncation():
    seed, raw = inputs(); aligned, _, _ = fit_sources(seed, raw)
    incumbent = scored_once(raw[0])
    domains, provenance, build = merge_domains(aligned, incumbent)
    np.testing.assert_array_equal(np.array([d[0] for d in domains]), incumbent)
    assert all(-1 in p[0] for p in provenance)
    for source in range(3):
        assert all(any(source in ids for ids in piece) for piece in provenance)
    assert max(map(len, domains)) <= 500
    assert sum(build["source_new_pose_totals_in_input_order"]) > 0
    with pytest.raises(ValueError, match="exceeding|max_poses"):
        merge_domains(aligned, incumbent, max_poses=10)


def test_all_physical_supports_remain_bounds_for_mixed_choices():
    seed, raw = inputs(); aligned, _, _ = fit_sources(seed, raw)
    incumbent = scored_once(raw[0]); domains, provenance, _ = merge_domains(aligned, incumbent)
    cuts = all_frame_supports(domains, [incumbent, *aligned])
    assert cuts
    rng = np.random.default_rng(3111901)
    for _ in range(12):
        assignment = [int(rng.integers(len(d))) for d in domains]
        p = np.array([d[value] for d, value in zip(domains, assignment)])
        radius = compute_mec_fast(p)[2]
        for cut in cuts:
            bound = (sum(values[assignment[i]] for i, values in cut["coefficients"].items()) - cut["offset"]) / cut["scale"]
            assert bound <= radius + 1e-11
    classified = assignment_sources({"valid": True, "assignment": [0] * 15}, provenance)
    assert -1 in classified["common_source_domain_indices"]
    assert not classified["requires_cross_source_domain_choices"]


def test_private_support_hook_restores_after_failure():
    original = support_module.support_bundle
    with pytest.raises(RuntimeError):
        with use_supports(["sentinel"]):
            assert support_module.support_bundle(None, None, None) == ["sentinel"]
            raise RuntimeError("test")
    assert support_module.support_bundle is original
