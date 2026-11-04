"""Pinned integer support triangles with integer flat differences before repair.

Continuous completion is only a proposal. Saved integer candidates require the
original reference checker; this runner never calls the arena or promotes files.
"""
from __future__ import annotations

import os
for _name in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import itertools
import json
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from experiment_support_completion import pinned_domains, relative_constraints, write
from lazy_phase_solver import ACCEPTANCE_GUARD, solve_phase
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.polish import AreaConstraints, checked, load, polygon_areas
from translation_union import domain_digest

ROOT = Path(__file__).resolve().parents[1]
FLATS = ((1, 14), (4, 8))
TARGET = 2.9485269238806113


def equality_constraints(p, supports, branch):
    relative, values = relative_constraints(p, supports)
    flat = np.zeros((len(FLATS), 3 * len(p) + 1))
    for row, (i, j) in enumerate(FLATS):
        flat[row, 3 * i + 2] = 1
        flat[row, 3 * j + 2] = -1
    return np.vstack([relative, flat]), np.r_[values, np.asarray(branch, dtype=float)]


def branch_choices(p):
    nearest = np.rint([1e6 * (p[i, 2] - p[j, 2]) for i, j in FLATS]).astype(np.int64)
    offsets = sorted(itertools.product((-1, 0, 1), repeat=2),
                     key=lambda v: (abs(v[0]) + abs(v[1]), v))
    return [(nearest + offset) / 1e6 for offset in offsets]


def complete_branch(p, fixed, supports, branch, *, iterations=180):
    local = np.asarray(p, dtype=float).copy()
    circle = compute_mec_fast(local)
    local[:, :2] -= circle[:2]
    # Meet the exact branch equations at initialization; all other poses remain
    # movable, and overlap constraints determine the needed collective response.
    for wanted, (i, j) in zip(branch, FLATS):
        change = wanted - (local[i, 2] - local[j, 2])
        local[i, 2] += change / 2
        local[j, 2] -= change / 2
    pinned = fixed.copy()
    pinned[:, :2] -= circle[:2]
    matrix, values = equality_constraints(pinned, supports, branch)
    constraints = AreaConstraints(local)
    z = np.r_[local.ravel(), compute_mec_fast(local)[2] + 1e-7]
    direction = np.zeros_like(z)
    direction[-1] = 1
    result = minimize(lambda v: v[-1], z, jac=lambda v: direction, method="SLSQP",
                      constraints=[{"type": "ineq", "fun": constraints.fun, "jac": constraints.jac},
                                   {"type": "eq", "fun": lambda v: matrix @ v - values,
                                    "jac": lambda v: matrix}],
                      options={"maxiter": iterations, "ftol": 2e-13})
    q = result.x[:-1].reshape(-1, 3).copy()
    q[:, :2] += fixed[supports[0], :2] - q[supports[0], :2]
    flat_actual = np.array([q[i, 2] - q[j, 2] for i, j in FLATS])
    model = {"success": bool(result.success), "message": str(result.message),
             "radius_variable": float(result.x[-1]), "native_mec": list(compute_mec_fast(q)),
             "model_violation": float(max(0., -constraints.fun(result.x).min())),
             "relative_pin_error": float(np.max(np.abs(q[supports, :2] - fixed[supports, :2]))),
             "flat_error": float(np.max(np.abs(flat_actual - branch))),
             "flat_actual": flat_actual.tolist(), "iterations": int(result.nit),
             "calls": constraints.calls}
    return q, model


def run(source_directory, output, *, seconds=240, target=TARGET, seed=3111801, retry_failed=False):
    source_directory, output = Path(source_directory), Path(output)
    if not np.isfinite(seconds) or seconds <= 0 or not np.isfinite(target) or target <= 0:
        raise ValueError("Budget and target must be positive and finite")
    if output.exists():
        raise ValueError("Output directory must be new")
    output.mkdir(parents=True)
    (output / "source-snapshot").mkdir()
    (output / "inputs").mkdir()
    source_hashes = {}
    for name in ("experiment_support_branches.py", "experiment_support_completion.py",
                 "lazy_phase_solver.py", "lazy_pair_oracle.py", "translation_union.py"):
        source = ROOT / "scripts" / name
        shutil.copy2(source, output / "source-snapshot" / name)
        source_hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    source_report = source_directory / "report.json"
    old = json.loads(source_report.read_text())
    supports = old["supports"]
    if len(supports) != 3:
        raise ValueError("Exactly three support centers are required")
    inputs = []
    for row in old["rows"]:
        if retry_failed and row.get("continuous_eligible") is not False:
            continue
        path = source_directory / f"completion-{row['serial']:03d}.json"
        p = load(path)
        if p.shape != (15, 3) or not np.all(np.isfinite(p)):
            raise ValueError("Expected finite 15-piece continuous completions")
        fixed = p.copy()
        fixed[supports, :2] = np.round(fixed[supports, :2], 6)
        if np.max(np.abs(fixed[supports, :2] - p[supports, :2])) > 1e-9:
            raise ValueError("Saved continuous source no longer has integer support pins")
        frozen = output / "inputs" / path.name
        shutil.copy2(path, frozen)
        branches = [row["branch"]] if retry_failed else [b.tolist() for b in branch_choices(p)]
        inputs.append((p, fixed, {"serial": row["serial"], "path": str(path),
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "branches": branches}))
    report = {"scope": "Six-equality continuous proposal followed by exact finite integer repair; no API or promotion",
              "source_report": str(source_report),
              "source_report_sha256": hashlib.sha256(source_report.read_bytes()).hexdigest(),
              "source_hashes": source_hashes, "inputs": [entry[2] for entry in inputs],
              "supports": supports, "flats": [list(pair) for pair in FLATS], "target": target,
              "budget_seconds": seconds, "seed": seed, "retry_failed": retry_failed,
              "scheduled": sum(len(v[2]["branches"]) for v in inputs),
              "continuous_eligibility": {"model_violation_at_most": 1e-9,
                                          "relative_pin_error_at_most": 1e-10,
                                          "flat_error_at_most": 1e-10,
                                          "native_mec_at_most": target},
              "rows": []}
    started = time.monotonic()
    seen = {}
    best = None
    for branch_index, input_index in itertools.product(range(1 if retry_failed else 9), range(len(inputs))):
        if time.monotonic() - started >= seconds:
            break
        p, fixed, source = inputs[input_index]
        branch = np.asarray(source["branches"][branch_index])
        serial = len(report["rows"])
        before = time.monotonic()
        q, model = complete_branch(p, fixed, supports, branch, iterations=300 if retry_failed else 180)
        row = {"serial": serial, "input_index": input_index, "branch_index": branch_index,
               "branch": branch.tolist(), "completion": model,
               "completion_seconds": time.monotonic() - before, "repairs": []}
        write(output / f"completion-{serial:03d}.json",
              [dict(zip(("x", "y", "theta"), map(float, point))) for point in q])
        eligible = (model["relative_pin_error"] <= 1e-10 and model["flat_error"] <= 1e-10
                    and model["model_violation"] <= 1e-9 and model["native_mec"][2] <= target)
        row["continuous_eligible"] = eligible
        if eligible:
            exact_areas = polygon_areas(q)
            row["continuous_geos_max_area"] = float(exact_areas.max())
            row["continuous_geos_bad_pairs"] = int((exact_areas > 1e-6).sum())
            for width in (1, 3):
                remaining = seconds - (time.monotonic() - started)
                if remaining <= 0:
                    break
                domains = pinned_domains(q, fixed, supports, angle_width=width)
                digest = domain_digest(domains)
                if digest in seen:
                    row["repairs"].append(dict(width=width, domain_sha256=digest,
                                               status="DUPLICATE_DOMAIN", previous=seen[digest]))
                    continue
                folder = output / f"repair-{serial:03d}-w{width}"
                result = solve_phase(domains, np.asarray([d[0] for d in domains]), target, folder,
                                     seconds=min(3 if width == 1 else 5, remaining),
                                     seed=seed + serial * 2 + width, local_only=True)
                seen[digest] = {"serial": serial, "width": width, "status": result["status"]}
                row["repairs"].append(dict(width=width, domain_sha256=digest, result=result))
                if result.get("valid"):
                    check = checked(load(folder / "solution.json"))
                    assert check.valid and check.score <= target + ACCEPTANCE_GUARD
                    best = float(check.score)
                    shutil.copy2(folder / "solution.json", output / "candidate.json")
                    report["best"] = {"score": best, "serial": serial, "width": width,
                                      "path": "candidate.json"}
                    break
        report["rows"].append(row)
        report["seconds"] = time.monotonic() - started
        write(output / "report.json", report)
        print(json.dumps({"serial": serial, "input": input_index, "branch_index": branch_index,
                          "radius": model["native_mec"][2], "violation": model["model_violation"],
                          "eligible": eligible,
                          "repairs": [v.get("status", v.get("result", {}).get("status")) for v in row["repairs"]],
                          "best": best}), flush=True)
        if best is not None:
            break
    report.update(seconds=time.monotonic() - started, completed=len(report["rows"]),
                  distinct_exact_domains=len(seen), status="VALID_CANDIDATE" if best is not None else
                  "COMPLETE" if len(report["rows"]) == report["scheduled"] else "BUDGET_EXHAUSTED")
    write(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=240)
    parser.add_argument("--seed", type=int, default=3111801)
    parser.add_argument("--retry-failed", action="store_true",
                        help="Warm-restart only numerically ineligible rows at their same recorded branch")
    args = parser.parse_args()
    result = run(args.source_directory, args.output, seconds=args.seconds, seed=args.seed,
                 retry_failed=args.retry_failed)
    print(json.dumps({key: result[key] for key in ("status", "seconds", "completed", "distinct_exact_domains")}), flush=True)


if __name__ == "__main__":
    main()
