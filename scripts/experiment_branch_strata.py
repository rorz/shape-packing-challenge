"""Offline equal-coverage native search on expensive continuous flat branches.

The mapped continuous costs define sampling strata, not feasibility bounds or
evidence that those costs predict integer packing quality. Both reflected
families and their actual J@T contact stresses are evaluated independently.
"""
from __future__ import annotations

import os
for _name in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import Counter
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmark_recovery import fit_frame, scored_once, transform
from breakout_geometry import evaluate_batch
from experiment_batch_lattice import NativeBabai
from experiment_mirror_family import FLATS, Manifold, cells
from experiment_mirror_frontier import OverlapFrontier
from experiment_tied_batch import Frame
from semicircle_packing.polish import checked, load, save

LEADER = 2.9485269238856113
INCUMBENT = 2.948526950912552
ROOT = Path(__file__).resolve().parents[1]
STRATA = (("cost20-50", 20., 50.), ("cost50-100", 50., 100.), ("cost100-126", 100., float("inf")))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path); temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def select_stratum(rows, lower, upper):
    selected = [dict(row) for row in rows if row.get("feasible_positive_tied_stress") and lower <= row["cost_nm"] < upper]
    return sorted(selected, key=lambda row: (round(row["first_raw"] * 1e6), round(row["second_raw"] * 1e6)))


class EqualCoverage:
    """Every completed pass visits every branch exactly once."""
    def __init__(self, count, seed):
        if count < 1:
            raise ValueError("A stratum must contain branches")
        self.rng = np.random.default_rng(seed)
        self.count = count; self.order = []; self.visits = np.zeros(count, dtype=int)

    def next(self):
        if not self.order:
            self.order = list(map(int, self.rng.permutation(self.count)))
        index = self.order.pop()
        visit = int(self.visits[index]); self.visits[index] += 1
        return index, visit


def reflected(poses):
    p = np.asarray(poses, dtype=float).copy()
    p[:, 1] *= -1; p[:, 2] *= -1
    return p


def choose_angle(physical, raw_branch, preferred, rng, *, local=True, sigma=.0003):
    intervals = cells(physical, np.sign(raw_branch))
    if not intervals:
        raise ValueError("No valid raw-sign wrap cell")
    if local:
        for _ in range(12):
            angle = float((preferred + rng.normal(0., sigma)) % (2 * np.pi))
            if any(lo + 1e-12 < angle < hi - 1e-12 for lo, hi in intervals):
                return angle, "local"
    widths = np.array([hi - lo for lo, hi in intervals])
    lo, hi = intervals[int(rng.choice(len(intervals), p=widths / widths.sum()))]
    return float(rng.uniform(lo, hi)), "global_fallback" if local else "global"


def make_frame(model, raw_branch, alpha, angle, rng, settings):
    physical, section = model.section(raw_branch, alpha)
    if max(section["violation"], section["residual"]) > 2e-9:
        raise ValueError("Invalid continuous section")
    if section["radius"] >= settings.target:
        raise ValueError("Continuous section exceeds sampling target")
    posed = transform(physical, angle, [0., 0.])
    differences = np.array([posed[i, 2] - posed[j, 2] for i, j in FLATS])
    if np.max(np.abs(differences - raw_branch)) > 2e-10:
        raise ValueError("Posed raw branch mismatch")
    frame = Frame(posed, section["radius"], FLATS, settings)
    if frame.budget <= 0:
        raise ValueError("Nonpositive tied contact budget")
    anchor = frame.anchor_units.copy()
    translation = rng.uniform(-.5e-6, .5e-6, 2)
    changed = np.rint(posed * 1e6).astype(np.int64)
    changed[:, :2] = np.rint((posed[:, :2] + translation) * 1e6).astype(np.int64)
    for (i, j), difference in zip(FLATS, np.rint(raw_branch * 1e6).astype(np.int64)):
        changed[i, 2] = changed[j, 2] + difference
    _, jacobian = frame.c.evaluate(np.r_[posed.ravel(), section["radius"]])
    frame.b0 -= jacobian[frame.active, :45] @ (changed - anchor).ravel()
    frame.anchor_units = changed
    return frame, posed, section, translation


def run_block(rows, origin, preferred_angle, output, *, seconds, seed, control=False, target=LEADER+2e-8):
    output = Path(output); output.mkdir(parents=True)
    models = [Manifold(origin), Manifold(reflected(origin))]
    scheduler = EqualCoverage(len(rows), seed)
    phase_rng = np.random.default_rng(seed + 1); sample_rng = np.random.default_rng(seed + 2)
    decoder = NativeBabai(); decoder.reset()
    frontier = OverlapFrontier(output / "frontier")
    branch_records = [{"index": i, "raw": [r["first_raw"], r["second_raw"]],
                       "nominal_cost_nm": r["cost_nm"], "attempts": 0, "frames": 0,
                       "reflected_attempts": 0, "evaluated": 0, "reference_valid": 0,
                       "minimum_actual_radius": None, "maximum_actual_radius": None,
                       "minimum_tied_stress": None, "negative_full_J_frames": 0} for i, r in enumerate(rows)]
    counts = Counter(); rejections = Counter(); orientations = Counter(); valid_records = []
    validated = set(); best_score = None; best_deficit = None
    begin = time.monotonic(); deadline = begin + seconds; last = begin
    def elapsed():
        return time.monotonic() - begin
    def emit(kind, **data):
        row = {"event": kind, "seconds": elapsed(), **data}
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if kind in ("valid", "progress", "done"):
            print(json.dumps(row, allow_nan=False), flush=True)
    def register(raw, branch_index, source):
        nonlocal best_score
        key = np.rint(raw * 1e6).astype("<i8").tobytes()
        if key in validated:
            return
        validated.add(key)
        result = checked(raw); counts["reference_checks"] += 1
        if not result.valid:
            return
        counts["reference_valid"] += 1
        if branch_index is not None:
            branch_records[branch_index]["reference_valid"] += 1
        late = elapsed() > seconds
        path = output / f"valid-{len(valid_records):04d}.json"; save(path, raw)
        record = {"file": path.name, "sha256": sha(path), "score": float(result.score),
                  "seconds": elapsed(), "branch_index": branch_index, "source": source,
                  "validation_after_deadline": late}
        valid_records.append(record)
        if best_score is None or result.score < best_score:
            best_score = float(result.score); save(output / "best.json", raw)
        emit("valid", **record)
    def report(status):
        return {"status": status, "seed": seed, "budget_seconds": seconds, "seconds": elapsed(),
                "sampling_target": target, "leader_target": LEADER, "sigma_physical": .0002,
                "sigma_orientation": .0003, "requested_local_fraction": .75,
                "control": control, "counts": dict(counts), "rejections": dict(rejections),
                "orientations": dict(orientations), "best_score": best_score,
                "best_native_combined_deficit": best_deficit, "branches": branch_records,
                "valid_records": valid_records, "frontier_counts": frontier.counts,
                "coverage": {"branches": len(rows), "attempted_branches": int(np.count_nonzero(scheduler.visits)),
                             "minimum_attempts": int(scheduler.visits.min()), "maximum_attempts": int(scheduler.visits.max())},
                "scope": "Equal branch attempts, not equal valid proposals or statistically matched branch-quality inference"}
    while time.monotonic() < deadline:
        branch_index, visit = scheduler.next(); row = rows[branch_index]; branch_record = branch_records[branch_index]
        reflect = visit % 2
        branch_record["attempts"] += 1; branch_record["reflected_attempts"] += reflect
        branch = np.array([row["first_raw"], row["second_raw"]]) * (-1 if reflect else 1)
        model = models[reflect]
        alpha = np.zeros(2) if control and visit == 0 else phase_rng.normal(0., .0002, 2)
        settings = SimpleNamespace(target=target, regularization=1e-5, block_size=20,
                                   concentration=(10., 1., .3)[visit % 3], boundary_bias=1.,
                                   movement=(10., 50., 150.)[visit % 3])
        try:
            physical, pre_section = model.section(branch, alpha)
            if max(pre_section["violation"], pre_section["residual"]) > 2e-9:
                raise ValueError("Invalid continuous section")
            preferred = (-preferred_angle if reflect else preferred_angle) + (visit // 2 % 4) * np.pi / 2
            if control and visit == 0:
                angle, mode = 0., "control_origin"
            else:
                angle, mode = choose_angle(physical, branch, preferred, phase_rng, local=bool(phase_rng.random() < .75))
            frame, posed, section, translation = make_frame(model, branch, alpha, angle, phase_rng, settings)
        except (ValueError, np.linalg.LinAlgError) as error:
            rejections[str(error)] += 1; counts["rejected_frames"] += 1
            continue
        counts["frames"] += 1; branch_record["frames"] += 1; orientations[mode] += 1
        radius = float(section["radius"])
        for name, fn in (("minimum_actual_radius", min), ("maximum_actual_radius", max)):
            old = branch_record[name]; branch_record[name] = radius if old is None else fn(old, radius)
        minimum_stress = float(frame.weights.min())
        old = branch_record["minimum_tied_stress"]
        branch_record["minimum_tied_stress"] = minimum_stress if old is None else min(old, minimum_stress)
        branch_record["negative_full_J_frames"] += int(section["min_stress"] <= 0)
        actual, units, coeff, ambient, ambiguous = frame.sample(sample_rng, 2048, settings, decoder)
        counts["proposals"] += len(actual); counts["ambiguous_patches"] += ambiguous
        if counts["frames"] == 1:
            expected = np.array([frame.M.babai(v.tolist()) for v in ambient[:64]], dtype=np.int64)
            mismatches = int(np.any(expected != coeff[:64], axis=1).sum())
            emit("native_control", compared=64, mismatches=mismatches, minimum_tied_stress=minimum_stress)
            if mismatches:
                raise AssertionError("Native decoder differs from fpylll")
        allowed = np.ones(len(actual), dtype=bool)
        for (i, j), difference in zip(FLATS, np.rint(branch * 1e6).astype(np.int64)):
            allowed &= units[:, 3*i+2] - units[:, 3*j+2] == difference
        counts["branch_rejected"] += int((~allowed).sum())
        indices = np.flatnonzero(allowed); keep, duplicate = decoder.unique(units[allowed]); indices = indices[keep]
        actual, coeff = actual[indices], coeff[indices]
        counts["duplicates"] += int(duplicate[0]); counts["unique_evaluated"] += len(actual)
        branch_record["evaluated"] += len(actual)
        if not len(actual):
            continue
        areas, circles, _ = evaluate_batch(actual, centers=[0., 0.])
        deficit = np.maximum(np.maximum((areas - 1e-6) / frame.c.scale, 0.).max(axis=1), np.maximum(circles[:, 2] - LEADER, 0.))
        index = int(np.argmin(deficit))
        if best_deficit is None or float(deficit[index]) < best_deficit:
            best_deficit = float(deficit[index]); save(output / "nearest-raw.json", frame.raw(coeff[index]))
            write_json(output / "nearest.json", {"native_combined_deficit": best_deficit, "radius": float(circles[index, 2]),
                       "branch_index": branch_index, "reflected": bool(reflect), "raw_branch": branch.tolist(),
                       "physical_alpha": alpha.tolist(), "angle": angle, "actual_continuous_radius": radius})
        def state_at(k):
            return {"origin": posed, "anchor": frame.anchor_units, "moves": frame.moves,
                    "coefficients": coeff[k], "branch": branch, "alpha": alpha,
                    "branch_index": np.array(branch_index), "reflected": np.array(reflect),
                    "rotation": np.array(angle), "translation_phase": translation}
        frontier_valid = frontier.consider(actual, areas, circles, frame.c.scale,
                                            lambda k: frame.raw(coeff[k]), state_at)
        eligible = set(map(int, np.flatnonzero((areas.max(axis=1) <= 1e-6+1e-11) & (circles[:, 2] <= INCUMBENT+1e-7))))
        eligible.update(frontier_valid)
        for index in sorted(eligible, key=lambda k: circles[k, 2]):
            if time.monotonic() >= deadline:
                break
            register(frame.raw(coeff[index]), branch_index, "native")
        while frontier.deferred_valid:
            point = frontier.deferred_valid.pop()
            register(point["raw"], int(point["state"]["branch_index"]), "deferred_geos")
        if time.monotonic() - last >= 10:
            write_json(output / "report.json", report("RUNNING"))
            emit("progress", best_score=best_score, best_deficit=best_deficit, counts=dict(counts),
                 covered_branches=int(np.count_nonzero(scheduler.visits)))
            last = time.monotonic()
    frontier.flush()
    while frontier.deferred_valid:
        point = frontier.deferred_valid.pop()
        register(point["raw"], int(point["state"]["branch_index"]), "final_geos_flush")
    final = report("COMPLETE"); write_json(output / "report.json", final)
    emit("done", best_score=best_score, best_deficit=best_deficit, counts=dict(counts), coverage=final["coverage"])
    return final


def run(branch_map, origin_path, incumbent_path, output, *, seconds=90., control_seconds=8., seed=3111701):
    if not 0 < seconds <= 90 or not 0 < control_seconds <= 10:
        raise ValueError("Use strata in (0,90] seconds and control in (0,10] seconds")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be fresh")
    output.mkdir(parents=True, exist_ok=True)
    rows = json.loads(Path(branch_map).read_text()); origin = load(origin_path); incumbent = scored_once(load(incumbent_path))
    for name, source in (("branch-map.json", branch_map), ("origin.json", origin_path), ("incumbent.json", incumbent_path)):
        shutil.copy2(source, output / name)
    snapshots = output / "source-snapshot"; snapshots.mkdir()
    source_hashes = {}
    for name in ("experiment_branch_strata.py", "experiment_mirror_family.py", "experiment_tied_batch.py", "experiment_batch_lattice.py", "_batch_lattice.cpp", "experiment_mirror_frontier.py", "breakout_geometry.py", "_breakout_geometry.cpp", "benchmark_recovery.py"):
        source = ROOT / "scripts" / name; shutil.copy2(source, snapshots / name); source_hashes[name] = sha(source)
    model = Manifold(origin)
    current_branch = np.array([incumbent[i, 2] - incumbent[j, 2] for i, j in FLATS])
    fitted, fit = fit_frame(origin, incumbent, model=model, branch=current_branch.tolist())
    if not fit["success"] or max(fit["section"]["violation"], fit["section"]["residual"]) > 2e-9:
        raise ValueError("Known branch control fit failed")
    # Warm up native geometry/decoder before timing the blocks.
    NativeBabai(); evaluate_batch(origin[None])
    result = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "source_hashes": source_hashes,
              "inputs": {str(path): sha(path) for path in (branch_map, origin_path, incumbent_path)},
              "seed": seed, "seconds_per_stratum": seconds, "control_seconds": control_seconds,
              "known_branch_fit": fit, "sampling_policy": "Shuffled round-robin, one attempt per branch/pass; no cost-weighted sampling",
              "strata": {}, "scope": "Offline coverage intervention; no API/global promotion; costs are nominal alpha0 section costs, not causal performance estimates",
              "command": [sys.executable, *sys.argv]}
    control_row = [{"first_raw": float(current_branch[0]), "second_raw": float(current_branch[1]), "cost_nm": 0.}]
    result["control"] = run_block(control_row, fitted, 0., output / "control", seconds=control_seconds,
                                  seed=seed, control=True, target=INCUMBENT+3e-8)
    write_json(output / "report.json", result)
    for index, (name, lower, upper) in enumerate(STRATA):
        selected = select_stratum(rows, lower, upper)
        if not selected:
            raise ValueError(f"Empty stratum {name}")
        result["strata"][name] = run_block(selected, origin, fit["angle"], output / name,
            seconds=seconds, seed=seed+100*(index+1))
        write_json(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-map", type=Path, default=Path("experiments/2026-11-04/round3/mirror/branch-map/branches.json"))
    parser.add_argument("--origin", type=Path, default=Path("tests/fixtures/recovery/mirror-continuous.json"))
    parser.add_argument("--incumbent", type=Path, default=Path("experiments/pantheon-pilot/candidate-2.948526950912552.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=90.)
    parser.add_argument("--control-seconds", type=float, default=8.)
    parser.add_argument("--seed", type=int, default=3111701)
    args = parser.parse_args()
    result = run(args.branch_map, args.origin, args.incumbent, args.output,
                 seconds=args.seconds, control_seconds=args.control_seconds, seed=args.seed)
    print(json.dumps({name: {"best_score": r["best_score"], "coverage": r["coverage"]} for name, r in result["strata"].items()}, indent=2))


if __name__ == "__main__":
    main()
