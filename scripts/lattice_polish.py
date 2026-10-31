#!/usr/bin/env python3
"""Optimize six-decimal coordinates with a local mixed-integer contact model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from rounding_search import area_model, exact_valid, write_solution
from semicircle_packing.geometry import Semicircle, semicircle_polygon
from semicircle_packing.exact_area import intersection_areas, intersection_areas_and_jacobian
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.scoring import compute_mec


MICRO = 1e-6


def flat_contact_cuts(base, steps, margin=2e-11, samples=17):
    """Tangent planes along the narrow convex wedge at opposed flat edges."""
    pairs = [(i,j) for i in range(15) for j in range(i+1,15)]
    cuts = []
    for index, (i,j) in enumerate(pairs):
        angle_error = abs(abs(np.arctan2(np.sin(base[i,2]-base[j,2]),
                                        np.cos(base[i,2]-base[j,2]))) - np.pi)
        if angle_error > 1e-3 or np.linalg.norm(base[i,:2]-base[j,:2]) > 2+4*steps*MICRO:
            continue
        for delta in np.linspace(-2*steps*MICRO, 2*steps*MICRO, samples):
            candidate = base.copy()
            candidate[i,2] += delta
            normal = np.array([np.cos(candidate[i,2]), np.sin(candidate[i,2])])
            width = (100+4*steps)*MICRO
            lo, hi = -width, width
            origin = candidate[i,:2].copy()
            bracketed = False
            for _ in range(8):
                candidate[i,:2] = origin + lo*normal
                low_area = intersection_areas(candidate)[index]
                candidate[i,:2] = origin + hi*normal
                high_area = intersection_areas(candidate)[index]
                if low_area > 1e-6-margin and high_area <= 1e-6-margin:
                    bracketed = True
                    break
                lo *= 2
                hi *= 2
            if not bracketed:
                continue
            for _ in range(30):
                mid = (lo+hi)/2
                candidate[i,:2] = origin + mid*normal
                if intersection_areas(candidate)[index] > 1e-6-margin:
                    lo = mid
                else:
                    hi = mid
            candidate[i,:2] = origin + hi*normal
            _, jac = intersection_areas_and_jacobian(candidate)
            actual = semicircle_polygon(Semicircle(*candidate[i])).intersection(
                semicircle_polygon(Semicircle(*candidate[j]))).area
            cuts.append((candidate, np.array([actual]), jac[index:index+1]))
    return cuts


def containment_model(base, center, radius):
    rows, gaps = [], []
    for i, (x, y, theta) in enumerate(base):
        position = np.array([x, y]) - center
        normal = np.array([np.cos(theta), np.sin(theta)])
        tangent = np.array([-np.sin(theta), np.cos(theta)])
        for sign in (-1, 1):
            endpoint = position + sign * tangent
            distance = np.linalg.norm(endpoint)
            direction = endpoint / distance
            row = np.zeros(48)
            row[3*i:3*i+2] = -direction
            row[3*i+2] = sign * direction.dot(normal)
            row[45:47] = direction
            row[47] = 1
            rows.append(row)
            gaps.append(radius - distance)
        if position.dot(normal) > -1e-4:
            distance = np.linalg.norm(position)
            direction = position / distance
            row = np.zeros(48)
            row[3*i:3*i+2] = -direction
            row[45:47] = direction
            row[47] = 1
            rows.append(row)
            gaps.append(radius - distance - 1)
    return np.array(gaps), np.array(rows)


def solve(base, steps=5, seconds=20, margin=2e-11, cuts=(), target=None):
    base = np.round(base, 6)
    pairs, areas, area_jac = area_model(base)
    cx, cy, radius = compute_mec_fast(base)
    gaps, contain_matrix = containment_model(base, np.array([cx, cy]), radius)
    # Integer variables: 45 micro-unit placement steps. Continuous variables:
    # two enclosing-center steps and one radius step, also in micro-units.
    area_rows = [area_jac]
    area_limits = [1e-6 - margin - areas]
    for anchor, old_areas, old_jac in cuts:
        area_rows.append(old_jac)
        area_limits.append(1e-6 - margin - old_areas - old_jac @ (base - anchor).reshape(-1))
    area_matrix = np.zeros((sum(len(rows) for rows in area_rows), 48))
    area_matrix[:, :45] = np.vstack(area_rows) * MICRO / 1e-8
    matrix = np.vstack([area_matrix, contain_matrix])
    lower = np.r_[np.full(len(area_matrix), -np.inf), -gaps / MICRO]
    upper = np.r_[np.concatenate(area_limits) / 1e-8, np.full(len(gaps), np.inf)]
    cost = np.r_[np.zeros(47), 1.0]
    bounds = Bounds(np.r_[np.full(45, -steps, dtype=float), -10 * steps, -10 * steps, -3 * steps],
                    np.r_[np.full(45, steps, dtype=float), 10 * steps, 10 * steps, 3 * steps])
    # Integer translation is a pure symmetry. Pinning one center removes two
    # otherwise unbounded families of equivalent branch-and-bound choices.
    bounds.lb[:2] = 0
    bounds.ub[:2] = 0
    if np.max(areas) <= 1e-6 - margin:
        bounds.ub[-1] = 0
    if target is not None:
        bounds.ub[-1] = min(bounds.ub[-1], (target - radius) / MICRO)
        cost[:] = 0
    result = milp(cost, integrality=np.r_[np.ones(45), np.zeros(3)], bounds=bounds,
                  constraints=LinearConstraint(matrix, lower, upper),
                  options={"time_limit": seconds, "mip_rel_gap": 0.0})
    info = {"message": result.message, "success": bool(result.success), "base_radius": radius}
    if result.x is None:
        return None, info
    candidate = np.round(base + np.rint(result.x[:45]).reshape(15, 3) * MICRO, 6)
    info["predicted_radius"] = radius + result.x[-1] * MICRO
    polygons = [semicircle_polygon(Semicircle(*p)) for p in candidate]
    actual_areas = [polygons[i].intersection(polygons[j]).area for i,j in pairs]
    info["valid"] = all(area <= 1e-6 for area in actual_areas)
    info["max_area"] = max(actual_areas)
    info["invalid_pairs"] = [[i,j,area] for (i,j),area in zip(pairs,actual_areas) if area > 1e-6]
    info["score"] = compute_mec_fast(candidate)[2]
    info["max_step"] = float(np.abs(result.x[:45]).max())
    info["gap"] = float(result.mip_gap)
    return candidate, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("experiments/polished.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/lattice_best.json"))
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--margin", type=float, default=2e-11)
    parser.add_argument("--target", type=float)
    parser.add_argument("--flat-cuts", action="store_true")
    args = parser.parse_args()
    base = np.array([[p[k] for k in ("x", "y", "theta")] for p in json.loads(args.input.read_text())])
    best = float("inf")
    cuts = flat_contact_cuts(np.round(base,6), args.steps, args.margin) if args.flat_cuts else []
    started = time.perf_counter()
    for iteration in range(args.rounds):
        candidate, info = solve(base, args.steps, args.seconds, args.margin, cuts, args.target)
        print(json.dumps({"round": iteration, "elapsed": time.perf_counter()-started, **info}), flush=True)
        if candidate is None:
            break
        write_solution(args.output.with_name(args.output.stem + "_last.json"), candidate)
        if info["valid"] and info["score"] < best:
            reference_score = compute_mec([Semicircle(*p) for p in candidate])[2]
            if reference_score < best:
                best = reference_score
                write_solution(args.output, candidate)
        # Retain tangent cuts at bad near-parallel flat contacts. Their wedge
        # area is locally convex; one tangent alone permits crossing the kink.
        new_pairs, new_areas, new_jac = area_model(candidate)
        active = np.array([area > 1e-6 - args.margin for area in new_areas])
        if active.any():
            cuts.append((candidate.copy(), new_areas[active], new_jac[active]))
        else:
            base = candidate
            cuts = []


if __name__ == "__main__":
    main()
