"""Independent area checks, including coincident boundaries and tiny contacts."""

import math

import numpy as np
import pytest

from semicircle_packing.exact_area import intersection_areas, intersection_areas_and_jacobian
from semicircle_packing.geometry import Semicircle, semicircle_polygon


def test_coincident_circles_count_shared_arcs_once():
    for delta in [0, 0.2, math.pi/2, math.pi-1e-6, math.pi, -1.7]:
        placements = np.array([[0, 0, 0], [0, 0, delta]])
        expected = (math.pi-abs(delta))/2
        assert intersection_areas(placements)[0] == pytest.approx(expected, abs=1e-12)


def test_tiny_arc_arc_lens_and_flat_arc_cap():
    d = 1.9999
    lens = 2*math.acos(d/2) - d/2*math.sqrt(4-d*d)
    actual = intersection_areas([[0, 0, 0], [d, 0, math.pi]])[0]
    assert actual == pytest.approx(lens, abs=1e-13)
    h = 1e-4
    cap = math.acos(1-h)-(1-h)*math.sqrt(2*h-h*h)
    actual = intersection_areas([[0, 0, 0], [-1+h, 0, 0]])[0]
    assert actual == pytest.approx(cap, abs=1e-13)


def test_areas_match_random_independent_polygons():
    rng = np.random.default_rng(524)
    for _ in range(60):
        p = rng.normal(size=(2, 3))
        p[:, 2] *= 3
        a, b = [semicircle_polygon(Semicircle(*s), 4096) for s in p]
        actual = intersection_areas(p)[0]
        expected = a.intersection(b).area
        # Inscribed polygons have an area deficit of up to 2e-7 per half disk.
        assert actual >= expected-1e-12
        assert actual == pytest.approx(expected, abs=3e-7)


def test_area_analytic_jacobian_and_rigid_motion_invariants():
    rng = np.random.default_rng(145)
    for _ in range(50):
        p = rng.normal(size=(2, 3))
        p[:, 2] *= 3
        _, jac = intersection_areas_and_jacobian(p)
        finite = np.column_stack([
            (intersection_areas((p.ravel()+s).reshape(2, 3))
             - intersection_areas((p.ravel()-s).reshape(2, 3)))/(2e-6)
            for s in np.eye(6)*1e-6
        ])
        np.testing.assert_allclose(jac, finite, atol=1e-7)
        grad = jac.reshape(2, 3)
        np.testing.assert_allclose(grad[:, :2].sum(axis=0), 0, atol=1e-11)
        torque = -p[:, 1]*grad[:, 0]+p[:, 0]*grad[:, 1]+grad[:, 2]
        assert torque.sum() == pytest.approx(0, abs=1e-11)


def test_shared_flat_subgradient_preserves_translation_invariance():
    p = np.array([[0., 0., 0.], [0., .5, 0.]])
    area, jac = intersection_areas_and_jacobian(p)
    assert area[0] > 0
    np.testing.assert_allclose(jac[0, :2]+jac[0, 3:5], 0, atol=1e-12)
    finite = np.column_stack([
        (intersection_areas((p.ravel()+s).reshape(2, 3))
         - intersection_areas((p.ravel()-s).reshape(2, 3)))/(2e-7)
        for s in np.eye(6)*1e-7
    ])
    np.testing.assert_allclose(jac, finite, atol=1e-7)


def test_analytic_chord_correction_tracks_reference_contact_areas():
    from semicircle_packing.exact_area import polygon_intersection_areas
    # Arc-arc and flat-arc overlaps at the arena's one-millionth area scale.
    rng = np.random.default_rng(391)
    for _ in range(40):
        theta = rng.uniform(-math.pi, math.pi)
        direction = np.array([math.cos(theta), math.sin(theta)])
        distance = 2-rng.uniform(4e-5, 2e-4)
        p = np.array([[0., 0., theta], [*(direction*distance), theta+math.pi]])
        a, b = [semicircle_polygon(Semicircle(*q), 4096) for q in p]
        expected = a.intersection(b).area
        estimated = polygon_intersection_areas(p)[0]
        assert estimated == pytest.approx(expected, abs=2e-12)


def test_chord_correction_with_flat_arc_contacts():
    from semicircle_packing.exact_area import polygon_intersection_areas
    for theta in np.linspace(-2, 2, 12):
        distance = -1+1e-4
        p = np.array([[0, 0, theta], [distance*math.cos(theta), distance*math.sin(theta), theta]])
        a, b = [semicircle_polygon(Semicircle(*q), 4096) for q in p]
        expected = a.intersection(b).area
        assert polygon_intersection_areas(p)[0] == pytest.approx(expected, abs=2e-12)


def test_chord_correction_jacobian_includes_contact_motion_and_grid_phase():
    from semicircle_packing.exact_area import (
        polygon_intersection_areas,
        polygon_intersection_areas_and_jacobian,
    )
    p = np.array([[0., 0., 0.154], [1.9999, 0., math.pi+0.17]])
    _, polygon_jac = polygon_intersection_areas_and_jacobian(p)
    _, circle_jac = intersection_areas_and_jacobian(p)

    def correction(q):
        return polygon_intersection_areas(q)-intersection_areas(q)

    finite = np.column_stack([
        (correction((p.ravel()+s).reshape(2, 3))
         - correction((p.ravel()-s).reshape(2, 3)))/(2e-8)
        for s in np.eye(6)*1e-8
    ])
    np.testing.assert_allclose(polygon_jac-circle_jac, finite, atol=1e-9)
