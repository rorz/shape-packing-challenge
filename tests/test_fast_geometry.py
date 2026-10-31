"""Independent checks of the analytic search constraints and derivatives."""

import math

import numpy as np
import pytest

from semicircle_packing.fast_geometry import FastGeometry
from semicircle_packing.geometry import (
    Semicircle,
    farthest_boundary_point_from,
    semicircle_polygon,
)


@pytest.fixture(scope="module")
def geometry():
    return FastGeometry(2)


@pytest.mark.parametrize(
    "placements,gap",
    [
        ([0, 0, 0, 2.1, 0, math.pi], 0.1),
        ([0, 0, 0, 1.9, 0, math.pi], -0.1),
        ([0, 0, 0, 0, 0, math.pi], 0),
        ([0, 0, 0, 0, 0, 0], -1),
        ([0, 0, 0, -0.2, 0, math.pi], 0.2),
    ],
)
def test_separating_gap_at_known_contacts(geometry, placements, gap):
    values, _ = geometry.evaluate(np.r_[placements, 4.0])
    assert values[0] == pytest.approx(gap, abs=1e-12)


def test_random_pairs_against_independent_polygons(geometry):
    rng = np.random.default_rng(291)
    for _ in range(80):
        z = rng.normal(size=7)
        z[[2, 5]] *= 3
        z[-1] = 4
        gaps = geometry.constraints(z)
        a, b = (semicircle_polygon(Semicircle(*p), 1024) for p in z[:6].reshape(2, 3))
        if gaps[0] > 1e-5:
            assert gaps[0] == pytest.approx(a.distance(b), abs=3e-6)
        elif gaps[0] < -1e-4:
            assert a.intersection(b).area > 1e-10
        for i in range(2):
            far = farthest_boundary_point_from(Semicircle(*z[3*i:3*i+3]), 0, 0)
            assert gaps[1+i] == pytest.approx(4-math.hypot(*far), abs=1e-12)


def test_analytic_jacobian_matches_finite_differences(geometry):
    rng = np.random.default_rng(167)
    for _ in range(40):
        z = rng.normal(size=7)
        _, jac = geometry.evaluate(z)
        step = np.eye(7) * 1e-6
        finite = np.column_stack([
            (geometry.constraints(z+s)-geometry.constraints(z-s))/(2e-6)
            for s in step
        ])
        np.testing.assert_allclose(jac, finite, rtol=1e-5, atol=1e-7)


def test_penalty_gradient_aggregates_constraints(geometry):
    z = np.array([0, 0, 0.3, 1.4, 0.1, 3.5, 1.8])
    value, gradient, dr = geometry.penalty(z[:-1].reshape(2, 3), z[-1])
    numeric = []
    for step in np.eye(7)*1e-6:
        a,b = z+step,z-step
        va = geometry.penalty(a[:-1].reshape(2, 3), a[-1])[0]
        vb = geometry.penalty(b[:-1].reshape(2, 3), b[-1])[0]
        numeric.append((va-vb)/(2e-6))
    assert value > 0
    np.testing.assert_allclose(np.r_[gradient.ravel(), dr], numeric, atol=1e-7)


def test_boundary_keeps_both_tied_endpoints_active(geometry):
    z = np.array([2., 0., math.pi, -2., 0., 0., math.sqrt(5)])
    gaps, jac = geometry.boundary(z)
    np.testing.assert_allclose(gaps[[0, 1, 3, 4]], 0, atol=1e-12)
    assert gaps[2] == gaps[5] == 1  # radial arc extreme points inward
    assert jac[0, 2]*jac[1, 2] < 0  # opposing endpoint rotation derivatives


def test_all_boundary_constraint_gradients(geometry):
    rng = np.random.default_rng(379)
    for _ in range(30):
        z = rng.normal(size=7)
        gaps, jac = geometry.boundary(z)
        finite = np.column_stack([
            (geometry.boundary_constraints(z+s)-geometry.boundary_constraints(z-s))/(2e-6)
            for s in np.eye(7)*1e-6
        ])
        np.testing.assert_allclose(jac, finite, atol=1e-7)
        assert min(0, gaps.min()) == pytest.approx(min(0, geometry.constraints(z)[1:].min()))
