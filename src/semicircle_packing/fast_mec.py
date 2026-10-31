"""Fast convex enclosing-circle solve with analytic semicircle constraints.

This is a search accelerator. Final promoted solutions are still checked by
the reference scorer and the arena API.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.optimize import minimize

from .fast_geometry import FastGeometry


@lru_cache(maxsize=8)
def _geometry(count: int) -> FastGeometry:
    return FastGeometry(count)


def compute_mec_fast(placements, initial_center=None) -> tuple[float, float, float]:
    """Return (center_x, center_y, radius), solving only three variables.

    Accepts an n-by-3 array or a sequence of Semicircle objects. A final
    analytic containment pass raises the returned radius to cover numerical
    constraint residuals. Optimizer failures fall back to the reference solve.
    """
    if isinstance(placements, np.ndarray):
        points = np.asarray(placements, dtype=np.float64)
    else:
        points = np.array([(p.x, p.y, p.theta) for p in placements], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("Expected a nonempty n-by-3 placement array")
    if not np.isfinite(points).all():
        raise ValueError("Placements must be finite")

    anchor = np.mean(points[:, :2], axis=0) if initial_center is None else np.asarray(initial_center, dtype=float)
    local = points.copy()
    local[:, :2] -= anchor
    geometry = _geometry(len(points))
    last_variables = last_values = last_jacobian = None

    def evaluate(variables):
        nonlocal last_variables, last_values, last_jacobian
        if last_variables is not None and np.array_equal(variables, last_variables):
            return last_values, last_jacobian
        translated = local.copy()
        translated[:, :2] -= variables[:2]
        gaps, jac = geometry.boundary(np.r_[translated.reshape(-1), variables[2]])
        small_jac = np.column_stack((-jac[:, :-1:3].sum(axis=1),
                                     -jac[:, 1:-1:3].sum(axis=1), jac[:, -1]))
        last_variables, last_values, last_jacobian = variables.copy(), gaps, small_jac
        return gaps, small_jac

    initial_radius = -float(np.min(evaluate(np.zeros(3))[0]))
    result = minimize(
        lambda variables: variables[2], np.array([0.0, 0.0, initial_radius]),
        jac=lambda variables: np.array([0.0, 0.0, 1.0]), method="SLSQP",
        bounds=[(None, None), (None, None), (0.0, None)],
        constraints={"type": "ineq", "fun": lambda v: evaluate(v)[0],
                     "jac": lambda v: evaluate(v)[1]},
        options={"ftol": 2e-13, "maxiter": 50},
    )
    if not result.success or not np.isfinite(result.x).all():
        from .geometry import Semicircle
        from .scoring import compute_mec
        return compute_mec([Semicircle(*p) for p in points])
    center = result.x[:2] + anchor
    residual = float(np.min(evaluate(result.x)[0]))
    radius = result.x[2] + max(0.0, -residual)
    return float(center[0]), float(center[1]), float(radius)
