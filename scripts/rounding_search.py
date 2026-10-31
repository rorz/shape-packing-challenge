#!/usr/bin/env python3
"""Search rigid transforms before 6dp rounding, using linearized area screening.

Rigid motion preserves the continuous packing. Its phase relative to the
submission grid changes the quantization error. We screen millions of such
phases cheaply, then check promising candidates with the original 4096-point
polygon geometry and the original enclosing-circle scorer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from semicircle_packing.geometry import Semicircle, semicircle_polygon
from semicircle_packing.scoring import compute_mec
from semicircle_packing.exact_area import intersection_areas_and_jacobian


def polygon(p):
    return semicircle_polygon(Semicircle(*p))


def area_model(base, step=2e-7):
    polygons = [polygon(p) for p in base]
    pairs = [(i, j) for i in range(15) for j in range(i) if np.linalg.norm(base[i, :2] - base[j, :2]) < 2.00001]
    areas = np.array([polygons[i].intersection(polygons[j]).area for i, j in pairs])
    _, full_jac = intersection_areas_and_jacobian(base)
    pair_index = {(i,j):k for k,(i,j) in enumerate((i,j) for i in range(15) for j in range(i+1,15))}
    jac = np.array([full_jac[pair_index[j,i]] for i,j in pairs])
    return pairs, areas, jac


def exact_valid(candidate, pairs):
    polys = [polygon(p) for p in candidate]
    return all(polys[i].intersection(polys[j]).area <= 1e-6 for i, j in pairs)


def write_solution(path, candidate):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([dict(zip(("x", "y", "theta"), map(float, p))) for p in candidate], indent=2) + "\n")


def search(base, iterations, seed, output, seconds=None):
    t0 = time.perf_counter()
    pairs, areas, jac = area_model(base)
    rng = np.random.default_rng(seed)
    center_x, center_y, continuous = compute_mec([Semicircle(*p) for p in base])
    rounded_base = np.round(base, 6)
    best = compute_mec([Semicircle(*p) for p in rounded_base])[2] if exact_valid(rounded_base, pairs) else float('inf')
    center = np.array([center_x, center_y])
    attempted = screened = valid = 0
    print(f"continuous={continuous:.16f} baseline={best:.16f} contacts={len(pairs)} model_seconds={time.perf_counter()-t0:.3f}", flush=True)
    while attempted < iterations and (seconds is None or time.perf_counter() - t0 < seconds):
        count = min(8192, iterations - attempted)
        angles = rng.uniform(-np.pi, np.pi, count)
        cs, sn = np.cos(angles), np.sin(angles)
        shifts = rng.uniform(-0.5e-6, 0.5e-6, (count, 2))
        rotated = np.empty((count, 15, 3))
        rotated[:, :, 0] = base[None, :, 0] * cs[:, None] - base[None, :, 1] * sn[:, None] + shifts[:, None, 0]
        rotated[:, :, 1] = base[None, :, 0] * sn[:, None] + base[None, :, 1] * cs[:, None] + shifts[:, None, 1]
        rotated[:, :, 2] = base[None, :, 2] + angles[:, None]
        rounded = np.round(rotated, 6)
        rounded[:,:,2] = np.round(np.arctan2(np.sin(rounded[:,:,2]),np.cos(rounded[:,:,2])),6)
        error = rounded - rotated
        error[:,:,2] = np.arctan2(np.sin(error[:,:,2]),np.cos(error[:,:,2]))
        inverse = error.copy()
        inverse[:, :, 0] = error[:, :, 0] * cs[:, None] + error[:, :, 1] * sn[:, None]
        inverse[:, :, 1] = -error[:, :, 0] * sn[:, None] + error[:, :, 1] * cs[:, None]
        predicted_areas = inverse.reshape(count, 45) @ jac.T + areas
        # Allow a small model error; every retained candidate is checked exactly.
        eligible = np.flatnonzero(np.max(predicted_areas, axis=1) <= 1.00003e-6)
        attempted += count
        screened += len(eligible)
        for index in eligible:
            candidate = rounded[index]
            if not exact_valid(candidate, pairs):
                continue
            valid += 1
            _, _, score = compute_mec([Semicircle(*p) for p in candidate])
            if score < best:
                best = score
                write_solution(output, candidate)
                print(f"best={best:.16f} samples={attempted} screened={screened} valid={valid} seconds={time.perf_counter()-t0:.3f}", flush=True)
        if attempted % 262144 == 0:
            print(f"progress samples={attempted} screened={screened} valid={valid} best={best:.16f} seconds={time.perf_counter()-t0:.3f}", flush=True)
    print(json.dumps({"samples": attempted, "screened": screened, "valid": valid, "best": best, "seconds": time.perf_counter()-t0}), flush=True)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("solution.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/rounding_best.json"))
    parser.add_argument("--iterations", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args()
    base = np.array([[p[k] for k in ("x", "y", "theta")] for p in json.loads(args.input.read_text())])
    search(base, args.iterations, args.seed, args.output, args.seconds)


if __name__ == "__main__":
    main()
