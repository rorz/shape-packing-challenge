"""Translation unions must contain production neighborhoods at every phase."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from translation_union import domain_digest, make_translation_domains, translation_axis_bounds
from semicircle_packing.geometry import canonicalize_theta


def units(domain):
    return {tuple(row) for row in np.rint(np.asarray(domain) * 1e6).astype(np.int64)}


def production_domains(physical, phase, xy_width, angle_width):
    pytest.importorskip("ortools")
    from experiment_lazy_phases import make_domains

    shifted = physical.copy()
    shifted[:, :2] += phase
    base = np.round(shifted, 6)
    base[:, 2] = [canonicalize_theta(theta) for theta in base[:, 2]]
    return make_domains(base, xy_width, angle_width)


def test_random_common_translation_domains_are_contained():
    rng = np.random.default_rng(3110907)
    physical = rng.uniform(-3, 3, size=(15, 3))
    union = [units(d) for d in make_translation_domains(physical, 1, 3)]
    phases = list(itertools.product([-0.5e-6, 0., 0.5e-6], repeat=2))
    phases.extend(rng.uniform(-0.5e-6, 0.5e-6, size=(40, 2)))
    for phase in phases:
        for want, allowed in zip(production_domains(physical, phase, 1, 3), union):
            assert units(want) <= allowed


def test_grid_ties_negative_zero_and_large_coordinates_are_contained():
    values = [0., -0., 1e-6, -1e-6, 2e-6, -2e-6, 1.25e-6, -1.25e-6,
              1e-12, -1e-12, 0.123456, -0.123456, 123456.123456, -123456.123456]
    physical = np.array([[v, -v, 0.25] for v in values])
    union = [units(d) for d in make_translation_domains(physical, 1, 1)]
    endpoints = [-0.5e-6, np.nextafter(-0.5e-6, 0.), 0.,
                 np.nextafter(0.5e-6, 0.), 0.5e-6]
    for phase in itertools.product(endpoints, repeat=2):
        for want, allowed in zip(production_domains(physical, phase, 1, 1), union):
            assert units(want) <= allowed
    assert translation_axis_bounds(1e-6, 1) == (-1, 3)
    assert translation_axis_bounds(-1e-6, 1) == (-3, 1)


def test_generic_domain_has_48_poses_and_preserves_zero_phase_first():
    physical = np.array([[0.12345625, -0.76543225, 0.23456725]])
    domain = make_translation_domains(physical)[0]
    assert domain.shape == (48, 3)
    assert np.array_equal(domain[0], np.round(physical[0], 6))
    assert len(units(domain)) == len(domain)


def test_nesting_and_plain_pi_match_production():
    physical = np.array([[0.12345625, 0.43215625, theta]
                         for theta in (np.pi, -np.pi, 3.141592, -3.141592)])
    narrow = make_translation_domains(physical, angle_width=1)
    medium = make_translation_domains(physical, angle_width=3)
    wide = make_translation_domains(physical, angle_width=12)
    for a, b, c in zip(narrow, medium, wide):
        assert units(a) <= units(b) <= units(c)
    for want, allowed in zip(production_domains(physical, [0., 0.], 1, 3), medium):
        assert units(want) <= units(allowed)


def test_exceptional_wrap_alias_is_preserved_and_all_rows_encode():
    pytest.importorskip("ortools")
    from lazy_phase_solver import submission_for_geometry

    physical = np.array([[0.12345625, 0.43215625, theta] for theta in (-9.424778, 9.424778)])
    domains = make_translation_domains(physical, angle_width=3)
    assert domains[0][0, 2] == 3.141593
    assert domains[1][0, 2] == -3.141593
    for domain in domains:
        raw, scored = submission_for_geometry(domain)
        assert np.array_equal(scored, domain)
        assert np.array_equal(np.array([canonicalize_theta(v) for v in raw[:, 2]]), domain[:, 2])


def test_cap_is_enforced_before_angle_or_product_allocation(monkeypatch):
    import translation_union

    def forbidden(*args, **kwargs):
        pytest.fail("Allocated angle neighborhood before checking pose cap")

    monkeypatch.setattr(translation_union, "_angles", forbidden)
    with pytest.raises(ValueError, match="exceeding max_poses"):
        make_translation_domains([[0., 0., 0.]], angle_width=1_000_000)
    with pytest.raises(ValueError, match="exceeding max_poses"):
        make_translation_domains([[0., 0., 0.]], xy_width=1_000_000)


@pytest.mark.parametrize("kwargs", [{"xy_width": -1}, {"angle_width": 1.5}, {"max_poses": 0}])
def test_bad_widths_and_caps_are_rejected(kwargs):
    with pytest.raises(ValueError):
        make_translation_domains([[0., 0., 0.]], **kwargs)


def test_digest_deduplicates_row_order_but_keeps_piece_labels():
    domains = make_translation_domains([[.12345625, .65432125, .2], [1.12345625, .65432125, .3]])
    reordered = [np.concatenate([d[::-1], d[:1]]) for d in domains]
    assert domain_digest(domains) == domain_digest(reordered)
    assert domain_digest(domains) != domain_digest(domains[::-1])
