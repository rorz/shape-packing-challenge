"""Tolerance-aware constrained refinement using analytic intersection areas."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

for variable in ('VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(variable, '1')

import numpy as np
from scipy.optimize import minimize

from .exact_area import intersection_areas, intersection_areas_and_jacobian, polygon_intersection_areas_and_jacobian
from .fast_geometry import FastGeometry
from .geometry import Semicircle, semicircle_polygon, canonicalize_theta
from .scoring import validate_and_score


def load(path):
    with open(path) as f:
        return np.array([[v['x'], v['y'], v['theta']] for v in json.load(f)])


def save(path, placements):
    data = [dict(x=float(x), y=float(y), theta=float(t))
            for x, y, t in np.round(placements, 6)]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(target)


def polygon_areas(p):
    polygons = [semicircle_polygon(Semicircle(*v)) for v in p]
    return np.array([polygons[i].intersection(polygons[j]).area
                     for i in range(len(p)) for j in range(i + 1, len(p))])


class AreaConstraints:
    def __init__(self, placements, allowance=1e-6, polygon_correction=True, fast_polygon=True):
        self.fast_polygon = fast_polygon
        self.n = len(placements)
        self.fast = FastGeometry(self.n)
        self.allowance = np.full(self.fast.n_pairs, allowance)
        if polygon_correction and not fast_polygon:
            self.allowance += intersection_areas(placements) - polygon_areas(placements)
        self.last = None
        self.calls = 0
        # Normalize contact constraints to units of displacement.
        _, jac = self._areas_and_jac(np.r_[placements.ravel(), 3.0])
        self.scale = np.maximum(np.linalg.norm(jac, axis=1), 1e-3)

    def _areas_and_jac(self, z):
        p = z[:-1].reshape(self.n, 3)
        area_function = polygon_intersection_areas_and_jacobian if self.fast_polygon else intersection_areas_and_jacobian
        areas, derivatives = area_function(p)
        jac = np.column_stack([derivatives, np.zeros(len(areas))])
        return areas, jac

    def evaluate(self, z):
        if self.last is not None and np.array_equal(z, self.last):
            return self.g, self.j
        areas, jac = self._areas_and_jac(z)
        boundary, bj = self.fast.boundary(z)
        self.g = np.r_[(self.allowance - areas) / self.scale, boundary]
        self.j = np.vstack([-jac/self.scale[:, None], bj])
        self.last = z.copy()
        self.calls += 1
        return self.g, self.j

    def fun(self, z):
        return self.evaluate(z)[0]

    def jac(self, z):
        return self.evaluate(z)[1]


def polish(placements, *, allowance=1e-6, maxiter=300, polygon_correction=True, fast_polygon=True):
    """Return scipy result; last vector element is enclosing radius at origin."""
    p = np.asarray(placements, dtype=float)
    constraint = AreaConstraints(p, allowance, polygon_correction, fast_polygon=fast_polygon)
    z = np.r_[p.ravel(), 4.]
    z[-1] -= constraint.fast.constraints(z)[-len(p):].min()
    objective_jac = np.zeros(len(z))
    objective_jac[-1] = 1
    result = minimize(lambda v: v[-1], z, jac=lambda v: objective_jac,
                      constraints={'type': 'ineq', 'fun': constraint.fun,
                                   'jac': constraint.jac}, method='SLSQP',
                      options={'maxiter': maxiter, 'ftol': 2e-13})
    result.geometry_calls = constraint.calls
    result.max_violation = float(-min(0, constraint.fun(result.x).min()))
    return result


def checked(placements):
    p = np.round(placements, 6)
    p[:, 2] = [canonicalize_theta(t) for t in p[:, 2]]
    return validate_and_score([Semicircle(*v) for v in p])


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='?', default='solution.json')
    parser.add_argument('--output', default='experiments/polished.json')
    parser.add_argument('--allowance', type=float, default=1e-6)
    parser.add_argument('--maxiter', type=int, default=300)
    args = parser.parse_args()
    start = time.perf_counter()
    result = polish(load(args.input), allowance=args.allowance, maxiter=args.maxiter)
    p = result.x[:-1].reshape(-1, 3)
    save(args.output, p)
    validation = checked(p)
    print(json.dumps(dict(radius=result.x[-1], success=result.success,
                          message=str(result.message), iterations=result.nit,
                          geometry_calls=result.geometry_calls,
                          max_violation=result.max_violation,
                          seconds=time.perf_counter()-start,
                          valid=validation.valid, score=validation.score,
                          errors=validation.errors, output=args.output), indent=2), flush=True)


if __name__ == '__main__':
    main()
