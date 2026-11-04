"""Pin competitive integer boundary triangles, then repack and quantize jointly."""
from __future__ import annotations
import os
for _key in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import itertools
import json
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from breakout_geometry import evaluate_batch
from experiment_angle_orbits import scored_once
from lazy_phase_solver import ACCEPTANCE_GUARD, solve_phase
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.polish import AreaConstraints, checked, load, polygon_areas
from translation_union import make_translation_domains, domain_digest

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def relative_constraints(p, supports):
    """Fix support-center differences; leave common translation free."""
    matrix = np.zeros((2 * (len(supports) - 1), 3 * len(p) + 1))
    for k, piece in enumerate(supports[1:]):
        for axis in (0, 1):
            matrix[2*k+axis, 3*piece+axis] = 1
            matrix[2*k+axis, 3*supports[0]+axis] = -1
    return matrix, matrix @ np.r_[p.ravel(), 0.]


def complete(p, supports, *, iterations=150):
    circle = compute_mec_fast(p)
    local = p.copy()
    local[:, :2] -= circle[:2]
    constraint = AreaConstraints(local)
    matrix, values = relative_constraints(local, supports)
    z = np.r_[local.ravel(), circle[2] + 1e-7]
    direction = np.zeros_like(z)
    direction[-1] = 1
    result = minimize(lambda v: v[-1], z, jac=lambda v: direction,
                      method="SLSQP", constraints=[
                          {"type": "ineq", "fun": constraint.fun, "jac": constraint.jac},
                          {"type": "eq", "fun": lambda v: matrix @ v - values,
                           "jac": lambda v: matrix}],
                      options={"maxiter": iterations, "ftol": 2e-13})
    q = result.x[:-1].reshape(-1, 3).copy()
    q[:, :2] += p[supports[0], :2] - q[supports[0], :2]
    error = float(np.max(np.abs(q[supports, :2] - p[supports, :2])))
    return q, {"success": bool(result.success), "message": str(result.message),
               "radius": float(result.x[-1]), "native_mec": list(compute_mec_fast(q)),
               "model_violation": float(max(0., -constraint.fun(result.x).min())),
               "relative_pin_error": error, "iterations": int(result.nit),
               "calls": constraint.calls}


def pinned_domains(p, fixed, supports, *, angle_width=1):
    domains = make_translation_domains(p, xy_width=1, angle_width=angle_width, max_poses=2000)
    for index in supports:
        angles = list(dict.fromkeys(float(v) for v in domains[index][:, 2]))
        domains[index] = np.array([[*fixed[index, :2], angle] for angle in angles])
    return domains


def run(input_path, output, *, target=2.9485269238806113, seconds=180, limit=32, extent=2,
        exclude_reports=()):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Use a new output directory")
    output.mkdir(parents=True)
    (output / "source-snapshot").mkdir()
    source_hashes = {}
    for name in ("experiment_support_completion.py", "lazy_phase_solver.py", "translation_union.py"):
        path = ROOT / "scripts" / name
        source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copy2(path, output / "source-snapshot" / name)
    raw = load(input_path)
    validation = checked(raw)
    if not validation.valid:
        raise ValueError("Support completion requires a valid starting packing")
    p = scored_once(raw)
    input_sha = hashlib.sha256(Path(input_path).read_bytes()).hexdigest()
    excluded = set()
    exclusion_records = []
    for path in exclude_reports:
        old = json.loads(Path(path).read_text())
        if old["input_sha256"] != input_sha:
            raise ValueError("Previous offsets came from a different input")
        excluded.update(tuple(row["offsets"]) for row in old["rows"])
        exclusion_records.append({"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()})
    circle = np.asarray(compute_mec_fast(p))
    radial = p[:, :2] - circle[:2]
    distances = np.linalg.norm(radial, axis=1)
    normals = np.column_stack([np.cos(p[:, 2]), np.sin(p[:, 2])])
    supports = np.flatnonzero((abs(distances+1-circle[2]) < 2e-9) &
                             ((radial*normals).sum(axis=1) > 0)).tolist()
    if len(supports) != 3:
        raise ValueError(f"This diagnostic needs three radial supports, found{supports}")
    started = time.monotonic()
    offsets = np.array(list(itertools.product(range(-extent, extent+1), repeat=4)))
    poses = np.repeat(p[None], len(offsets), axis=0)
    poses[:, supports[1], :2] += offsets[:, :2] * 1e-6
    poses[:, supports[2], :2] += offsets[:, 2:] * 1e-6
    # Submicrounit representation noise must not change the intended pins.
    poses[:, :, :2] = np.round(poses[:, :, :2], 6)
    areas, circles, _ = evaluate_batch(poses, centers=circle[:2])
    scale = AreaConstraints(p).scale
    deficits = np.maximum((areas-1e-6)/scale, 0).max(axis=1)
    eligible = np.flatnonzero(circles[:, 2] < target)
    fresh = [i for i in eligible if tuple(offsets[i]) not in excluded]
    selected = sorted(fresh, key=lambda i: (deficits[i], np.sum(offsets[i]**2), circles[i, 2]))[:limit]
    report = {"scope": "Boundary-center conditioned continuous completion, then finite all-piece integer repair; no API or promotion",
              "input": str(input_path), "input_sha256": input_sha,
              "input_score": validation.score, "target": target, "budget_seconds": seconds,
              "supports": supports, "extent": extent, "triangles": len(offsets),
              "subtarget_triangles": len(eligible), "selected": len(selected),
              "excluded_prior_offsets": len(excluded), "previous_offset_reports": exclusion_records,
              "source_hashes": source_hashes, "rows": []}
    best = None
    for serial, index in enumerate(selected):
        if time.monotonic()-started >= seconds:
            break
        before = time.monotonic()
        fixed = poses[index]
        q, result = complete(fixed, supports)
        row = {"serial": serial, "offsets": offsets[index].tolist(),
               "initial_radius": float(circles[index, 2]), "initial_deficit": float(deficits[index]),
               "completion_seconds": time.monotonic()-before, "completion": result, "repairs": []}
        write(output / f"completion-{serial:03d}.json",
              [{k: float(v) for k, v in zip(("x", "y", "theta"), pose)} for pose in q])
        if result["relative_pin_error"] <= 1e-10 and result["native_mec"][2] <= target+1e-10 and result["model_violation"] <= 2e-8:
            exact_areas = polygon_areas(q)
            row["continuous_geos_max_area"] = float(exact_areas.max())
            row["continuous_geos_bad_pairs"] = int((exact_areas > 1e-6).sum())
            for width in (1, 3):
                remaining = seconds - (time.monotonic()-started)
                if remaining < .1:
                    break
                domains = pinned_domains(q, fixed, supports, angle_width=width)
                folder = output / f"repair-{serial:03d}-w{width}"
                result = solve_phase(domains, np.array([d[0] for d in domains]), target, folder,
                                     seconds=min(3 if width == 1 else 5, remaining),
                                     seed=3111701+serial*2+width, local_only=True)
                row["repairs"].append({"width": width, "domain_sha256": domain_digest(domains), "result": result})
                if result.get("valid"):
                    v = checked(load(folder / "solution.json"))
                    if not v.valid or v.score > target+ACCEPTANCE_GUARD:
                        raise AssertionError("Returned candidate failed independent validation")
                    if best is None or v.score < best:
                        best = v.score
                        shutil.copy2(folder / "solution.json", output / "best.json")
                        report["best"] = {"score": best, "serial": serial, "width": width, "path": "best.json"}
                    break
        report["rows"].append(row)
        report["seconds"] = time.monotonic()-started
        write(output / "report.json", report)
        print(json.dumps({"serial": serial, "radius": row["completion"]["native_mec"][2],
                          "violation": row["completion"]["model_violation"],
                          "repairs": [r["result"]["status"] for r in row["repairs"]], "best": best}), flush=True)
        if best is not None:
            break
    report.update(seconds=time.monotonic()-started, status="VALID_CANDIDATE" if best is not None else
                  "COMPLETE" if len(report["rows"]) == len(selected) else "BUDGET_EXHAUSTED")
    write(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"solution.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--extent", type=int, default=2)
    parser.add_argument("--exclude-report", type=Path, action="append", default=[])
    args = parser.parse_args()
    result = run(args.input, args.output, seconds=args.seconds, limit=args.limit, extent=args.extent,
                 exclude_reports=args.exclude_report)
    print(json.dumps({"status": result["status"], "seconds": result["seconds"], "best": result.get("best")}), flush=True)


if __name__ == "__main__":
    main()
