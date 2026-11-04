"""Conditional six-decimal orientation rounding by binary implication solving.

For fixed centers, every pair of adjacent orientation choices gives a 2x2
compatibility table. Each forbidden assignment is a two-literal clause;
containment at a fixed enclosing-circle center supplies unary clauses.
"""
from __future__ import annotations

from functools import lru_cache
import time
import numpy as np

from semicircle_packing.exact_area import polygon_intersection_areas
from semicircle_packing.fast_geometry import FastGeometry
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.geometry import canonicalize_theta


def solve_binary_constraints(unary: np.ndarray, compatibility: np.ndarray, *, return_conflicts=False):
    """Return a binary assignment or None for pairwise forbidden states.

    unary[i,a] says state a of variable i is allowed. compatibility[i,j,a,b]
    is consulted only for i<j. Tarjan's SCC algorithm solves the resulting 2-SAT.
    """
    unary = np.asarray(unary, dtype=bool)
    n = len(unary)
    graph = [[] for _ in range(2*n)]
    for i in range(n):
        for a in range(2):
            if not unary[i, a]:
                graph[2*i+a].append(2*i+1-a)
        for j in range(i+1, n):
            for a in range(2):
                for b in range(2):
                    if not compatibility[i, j, a, b]:
                        graph[2*i+a].append(2*j+1-b)
                        graph[2*j+b].append(2*i+1-a)
    index = np.full(2*n, -1, dtype=int)
    low = np.zeros(2*n, dtype=int)
    component = np.full(2*n, -1, dtype=int)
    stack, on_stack = [], np.zeros(2*n, dtype=bool)
    counter = components = 0

    def visit(vertex):
        nonlocal counter, components
        index[vertex] = low[vertex] = counter
        counter += 1
        stack.append(vertex)
        on_stack[vertex] = True
        for following in graph[vertex]:
            if index[following] < 0:
                visit(following)
                low[vertex] = min(low[vertex], low[following])
            elif on_stack[following]:
                low[vertex] = min(low[vertex], index[following])
        if low[vertex] == index[vertex]:
            while True:
                following = stack.pop()
                on_stack[following] = False
                component[following] = components
                if following == vertex:
                    break
            components += 1

    for vertex in range(2*n):
        if index[vertex] < 0:
            visit(vertex)
    conflicts=np.flatnonzero(component[::2] == component[1::2]).tolist()
    if conflicts:
        return (None,conflicts) if return_conflicts else None
    assignment=(component[::2] > component[1::2]).astype(np.int8)
    return (assignment,[]) if return_conflicts else assignment


@lru_cache(maxsize=8)
def _geometry(n):
    return FastGeometry(n)


@lru_cache(maxsize=8)
def _pair_indices(n):
    count = 2*n
    pairs, indices = [], []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((i, j))
            table = np.empty((2, 2), dtype=int)
            for a in range(2):
                for b in range(2):
                    u, v = 2*i+a, 2*j+b
                    table[a, b] = u*(2*count-u-1)//2+(v-u-1)
            indices.append(table)
    return np.asarray(pairs,dtype=int).reshape(-1,2), np.asarray(indices,dtype=int).reshape(-1,2,2)


def round_angles(continuous_p, target, center=None, *, return_info=False,
                 area_guard=1e-12, containment_tolerance=2e-12,
                 max_boundary_combinations=64):
    """Round fixed-position orientations, returning a packing or None.

    Input centers must already be on the six-decimal grid. If center is omitted,
    the continuous-orientation MEC is tried first, then small boundary-angle
    families and an angular minimum-cost completion supply alternative centers.
    A free-center failure means no
    assignment was found at the tested centers, not global angular UNSAT.
    Polygon estimates are guarded; the caller must reference-validate a saved
    improvement before promoting it.
    """
    started = time.perf_counter()
    p = np.asarray(continuous_p, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.isfinite(p).all():
        raise ValueError('Expected a finite n-by-3 placement array')
    if np.max(np.abs(p[:, :2]-np.round(p[:, :2], 6))) > 1e-10:
        raise ValueError('Positions must be quantized before angular rounding')
    n = len(p)
    fixed_center = center is not None
    if center is None:
        center = np.array(compute_mec_fast(p)[:2])
    else:
        center = np.asarray(center, dtype=float)
    theta = np.arctan2(np.sin(p[:, 2]), np.cos(p[:, 2]))
    units = theta*1e6
    nearest = np.rint(units)
    units = np.where(np.abs(units-nearest) < 1e-7, nearest, units)
    choices = np.column_stack([np.floor(units), np.ceil(units)])*1e-6
    choices = np.array([[canonicalize_theta(a) for a in row] for row in choices])
    variants = np.repeat(p, 2, axis=0)
    variants[:, 2] = choices.ravel()
    variants[:, :2] = np.round(variants[:, :2], 6)
    areas = polygon_intersection_areas(variants)
    pairs, indices = _pair_indices(n)
    tables = areas[indices] <= 1e-6-area_guard
    compatibility = np.ones((n, n, 2, 2), dtype=bool)
    compatibility[pairs[:, 0], pairs[:, 1]] = tables
    def at_center(query):
        relative = variants.copy()
        relative[:, :2] -= query
        gaps, _ = _geometry(2*n).boundary(np.r_[relative.ravel(), float(target)])
        unary = (gaps.reshape(2*n, 3).min(axis=1) >= -containment_tolerance).reshape(n, 2)
        assignment, conflicts = solve_binary_constraints(unary, compatibility, return_conflicts=True)
        return assignment, conflicts, unary

    tested_centers = [center.tolist()]
    assignment, conflicts, unary = at_center(center)
    recovery_used = False
    boundary_info = None
    if assignment is None and not fixed_center:
        try:
            from .angular_boundary_centers import boundary_centers
        except ImportError:
            from angular_boundary_centers import boundary_centers
        records, boundary_info = boundary_centers(p, choices, target=target+containment_tolerance,
                                                 max_combinations=max_boundary_combinations)
        for record in records:
            query = np.asarray(record['center'],dtype=float)
            if any(np.linalg.norm(query-np.asarray(previous))<=1e-14 for previous in tested_centers):
                continue
            recovery_used = True
            center = query
            tested_centers.append(center.tolist())
            assignment, conflicts, unary = at_center(center)
            if assignment is not None:break
    if assignment is None and not fixed_center and n <= 20:
        # Import only on failure: angular_min_cost shares this module's pair
        # index helper, but its minimum-cost solve does not call round_angles.
        try:
            from .angular_min_cost import minimize_angles
        except ImportError:
            from angular_min_cost import minimize_angles
        _, completion = minimize_angles(p, target)
        query = np.asarray(completion['center'], dtype=float)
        if np.linalg.norm(query-center) > 1e-14:
            recovery_used = True
            center = query
            tested_centers.append(center.tolist())
            assignment, conflicts, unary = at_center(center)
    info = dict(satisfiable=assignment is not None, center=center.tolist(),
                forbidden_pairs=int(np.count_nonzero(~tables)),
                forbidden_unary=int(np.count_nonzero(~unary)), conflicts=conflicts,
                fixed_center=fixed_center, tested_centers=tested_centers,
                recovery_used=recovery_used,
                boundary_oracle=boundary_info,
                native_family_covered=bool(boundary_info and boundary_info['complete_geometry']),
                status=('sat' if assignment is not None else
                        'fixed_center_unsat' if fixed_center else 'no_assignment_at_tested_centers'),
                seconds=time.perf_counter()-started)
    candidate = None
    if assignment is not None:
        candidate = p.copy()
        candidate[:, :2] = np.round(candidate[:, :2], 6)
        candidate[:, 2] = choices[np.arange(n), assignment]
        # Independent table checks protect the implication/SCC convention.
        assert np.all(unary[np.arange(n), assignment])
        assert np.all(compatibility[pairs[:, 0], pairs[:, 1], assignment[pairs[:, 0]], assignment[pairs[:, 1]]])
        info['assignment'] = assignment.tolist()
    return (candidate, info) if return_info else candidate
