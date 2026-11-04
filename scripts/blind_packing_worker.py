"""Offline, fixed-budget native-versus-hybrid packing search.

The only problem data is a continuous origin, a flat-angle branch and a score
cap. Hybrid fitting sees only candidates produced by this run. Neither method
reads target placements, fitted benchmark parameters or the workspace incumbent.
"""
from __future__ import annotations

import os
for _key in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmark_recovery import fit_frame, scored_once, transform, write_json
from breakout_geometry import evaluate_batch
from experiment_batch_lattice import NativeBabai
from experiment_mirror_family import FLATS, Manifold, cells, wrap
from experiment_tied_batch import Frame
from lazy_phase_solver import ACCEPTANCE_GUARD, solve_phase
from semicircle_packing.polish import AreaConstraints, checked, save

SCORE_GUARD = 1e-11
ANGLE_LADDER = (1, 3, 12)
DEFAULT_EXACT_SECONDS = (1., 2., 6.)
EXACT_FRACTION = .25
BATCH_SIZE = 2048


def read_problem(path):
    data = json.loads(Path(path).read_text())
    expected = {"id", "origin", "branch", "score_cap"}
    if set(data) != expected:
        raise ValueError(f"Blind problem keys must be exactly {sorted(expected)}")
    origin = np.asarray(data["origin"], dtype=float)
    branch = np.asarray(data["branch"], dtype=float)
    if origin.shape != (15, 3) or branch.shape != (2,) or not np.isfinite(origin).all() or not np.isfinite(branch).all():
        raise ValueError("Expected a finite 15-pose origin and two flat differences")
    if not isinstance(data["id"], str) or not np.isfinite(data["score_cap"]) or data["score_cap"] <= 0:
        raise ValueError("Problem id and score cap are invalid")
    return data, origin, branch


def center_hash(p):
    xy = np.rint(np.asarray(p)[:, :2] * 1e6).astype(np.int64)
    xy -= xy[0].copy()
    return hashlib.sha256(xy.astype("<i8").tobytes()).hexdigest()


def domain_hash(domains):
    h = hashlib.sha256()
    for d in domains:
        units = np.rint(d * 1e6).astype(np.int64)
        # Content identity, not the generator's heuristic row ordering.
        order = np.lexsort((units[:, 2], units[:, 1], units[:, 0]))
        h.update(np.asarray(units.shape, dtype="<i8").tobytes())
        h.update(units[order].astype("<i8").tobytes())
    return h.hexdigest()


def take_candidate(pending, desired_stage):
    """Select by scalar priority without comparing NumPy arrays in dicts."""
    choices = [c for c in pending if c["stage"] == desired_stage]
    candidate = min(choices or pending, key=lambda c: (c["loss"], c["serial"]))
    pending[:] = [item for item in pending if item is not candidate]
    return candidate


def warmup(origin=None):
    """Compile every native kernel before either arm's timed search."""
    from breakout_geometry import kernel as geometry_kernel
    from conflict_area_band import kernel as band_kernel
    from interval_order_optimization import optimize
    NativeBabai()
    geometry_kernel()
    band_kernel()
    optimize(np.eye(3, dtype=bool), seconds=.001, iterations=1)
    if origin is not None:
        AreaConstraints(origin).evaluate(np.r_[origin.ravel(), 4.])
        evaluate_batch(origin[None])


class ProposalStream:
    """Identical deterministic native proposal prefix for both methods."""

    def __init__(self, origin, branch, cap, seed):
        self.model = Manifold(origin)
        self.branch = branch
        self.cap = cap
        self.phase_rng = np.random.default_rng(seed)
        self.proposal_rng = np.random.default_rng(seed + 1)
        self.decoder = NativeBabai()
        self.decoder.reset()
        self.index = 0

    def next(self):
        index = self.index
        self.index += 1
        alpha = np.zeros(2) if index == 0 else self.phase_rng.normal(0, .0004, 2)
        physical, section = self.model.section(self.branch.tolist(), alpha)
        if max(section["violation"], section["residual"]) > 2e-9 or section["radius"] > self.cap - 8e-8:
            return None
        # Exploit the supplied origin's orientation, with a fixed global tail.
        if index == 0:
            angle = 0.
        elif index % 4:
            angle = float(self.phase_rng.normal(0, .0004))
        else:
            intervals = cells(physical, np.sign(self.branch))
            if not intervals:
                return None
            widths = np.array([hi - lo for lo, hi in intervals])
            lo, hi = intervals[int(self.phase_rng.choice(len(intervals), p=widths / widths.sum()))]
            angle = float(self.phase_rng.uniform(lo, hi))
        physical = transform(physical, angle, [0., 0.])
        actual_branch = np.array([physical[i, 2] - physical[j, 2] for i, j in FLATS])
        if np.max(np.abs(actual_branch - self.branch)) > 2e-10:
            return None
        settings = SimpleNamespace(
            target=self.cap, regularization=1e-5, block_size=20,
            concentration=(10., 1., .3)[index % 3], boundary_bias=1.,
            movement=(10., 50., 150.)[index % 3],
        )
        try:
            frame = Frame(physical, section["radius"], FLATS, settings)
        except ValueError:
            return None
        if frame.budget <= 0:
            return None
        # Exactly the existing native phase update, shared by both methods.
        anchor = frame.anchor_units.copy()
        shift = self.phase_rng.uniform(-.5e-6, .5e-6, 2)
        changed = np.rint(physical * 1e6).astype(np.int64)
        changed[:, :2] = np.rint((physical[:, :2] + shift) * 1e6).astype(np.int64)
        for (i, j), difference in zip(FLATS, np.rint(self.branch * 1e6).astype(np.int64)):
            changed[i, 2] = changed[j, 2] + difference
        _, jacobian = frame.c.evaluate(np.r_[physical.ravel(), section["radius"]])
        frame.b0 -= jacobian[frame.active, :45] @ (changed - anchor).ravel()
        frame.anchor_units = changed
        actual, units, coefficients, _, ambiguous = frame.sample(
            self.proposal_rng, BATCH_SIZE, settings, self.decoder)
        branch_units = np.rint(self.branch * 1e6).astype(np.int64)
        allowed = np.ones(len(actual), dtype=bool)
        for (i, j), difference in zip(FLATS, branch_units):
            allowed &= units[:, 3 * i + 2] - units[:, 3 * j + 2] == difference
        indices = np.flatnonzero(allowed)
        keep, duplicate_info = self.decoder.unique(units[allowed])
        indices = indices[keep]
        metadata = {
            "index": index, "alpha": alpha.tolist(), "angle": angle,
            "proposals": len(actual), "branch_rejected": int((~allowed).sum()),
            "ambiguous": ambiguous, "duplicates": int(duplicate_info[0]),
            "proposal_hash": hashlib.sha256(units.astype("<i8").tobytes()).hexdigest(),
        }
        return frame, actual[indices], coefficients[indices], metadata


def run(problem_path, output, *, method="native", seconds=120, seed=3111001,
        exact_seconds=DEFAULT_EXACT_SECONDS, already_warm=False):
    if method not in ("native", "hybrid"):
        raise ValueError("Method must be native or hybrid")
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Search duration must be positive and finite")
    if len(exact_seconds) != 3 or any(not np.isfinite(v) or v <= 0 for v in exact_seconds):
        raise ValueError("Three positive exact-stage budgets are required")
    problem_path, output = Path(problem_path), Path(output)
    problem, origin, branch = read_problem(problem_path)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite an existing run")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "problem.json", problem)
    warming = time.monotonic()
    if not already_warm:
        warmup(origin)
    warmup_seconds = time.monotonic() - warming
    started = time.monotonic()
    deadline = started + seconds
    cap = float(problem["score_cap"])
    stream = ProposalStream(origin, branch, cap, seed)
    records, late_records, solver_records, frame_records = [], [], [], []
    valid_centers, qualified_centers = {}, set()
    seen_candidates, seen_domains = set(), set()
    pending = []
    best_score, first_qualifying = None, None
    exact_spent = 0.
    counts = {key: 0 for key in ("frames", "rejected_frames", "proposals", "unique_evaluated",
                                "branch_rejected", "duplicates", "ambiguous", "reference_checks",
                                "reference_valid", "late_valid", "candidate_nominations", "queue_dropped",
                                "qualified_repairs_skipped", "duplicate_domains")}
    last_report = started

    def elapsed():
        return time.monotonic() - started

    def emit(event, **values):
        row = {"event": event, "seconds": elapsed(), **values}
        with (output / "events.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        if event in ("progress", "valid", "done"):
            print(json.dumps(row, allow_nan=False), flush=True)

    def register(raw, source):
        nonlocal best_score, first_qualifying
        validation = checked(raw)
        counts["reference_checks"] += 1
        if not validation.valid:
            return False
        counts["reference_valid"] += 1
        score, key = float(validation.score), center_hash(raw)
        timestamp = elapsed()
        scored = scored_once(raw)
        actual_branch = [round(float(scored[i, 2] - scored[j, 2]), 6) for i, j in FLATS]
        if timestamp > seconds:
            counts["late_valid"] += 1
            name = f"late-valid-{len(late_records):05d}.json"
            save(output / name, raw)
            late_records.append({"score": score, "seconds": timestamp, "source": source,
                                 "center_hash": key, "file": name, "branch": actual_branch})
            return False
        qualifies = score <= cap + SCORE_GUARD
        if qualifies:
            qualified_centers.add(key)
            if first_qualifying is None:
                first_qualifying = timestamp
        previous = valid_centers.get(key)
        is_better = previous is None or score < previous - 2e-12
        # Keep every reference-valid submission, including angle variants.
        # Aggregate center counts remain deduplicated independently.
        name = f"valid-{len(records):05d}.json"
        save(output / name, raw)
        record = {"score": score, "seconds": timestamp, "source": source,
                  "qualifying": qualifies, "center_hash": key, "file": name,
                  "new_center": previous is None,
                  "branch": actual_branch,
                  "same_requested_branch": bool(np.max(np.abs(np.array(actual_branch) - branch)) < 5e-7)}
        records.append(record)
        emit("valid", **record)
        if is_better:
            valid_centers[key] = score
        if best_score is None or score < best_score:
            best_score = score
            save(output / "best.json", raw)
        return True

    def summary():
        return {
            "schema_version": 1, "problem_id": problem["id"], "method": method,
            "seed": seed, "budget_seconds": seconds, "seconds": elapsed(),
            "warmup_seconds": warmup_seconds,
            "score_cap": cap, "qualification_guard": SCORE_GUARD,
            "best_score": best_score, "first_qualifying_seconds": first_qualifying,
            "unique_valid_centers": len(valid_centers),
            "qualifying_center_hashes": sorted(qualified_centers),
            "valid_centers": valid_centers, "valid_records": records,
            "late_valid_records": late_records,
            "counts": counts, "exact_seconds": exact_spent,
            "exact_fraction_limit": EXACT_FRACTION,
            "exact_budget_overrun": max(0., exact_spent - EXACT_FRACTION * seconds),
            "solver_records": solver_records, "frame_records": frame_records,
            "parameters": {"batch": BATCH_SIZE, "sigma": .0004,
                           "local_rotation_fraction": .75, "boundary_bias": 1.,
                           "angle_ladder": list(ANGLE_LADDER),
                           "seconds_per_width": list(exact_seconds),
                           "native_validation_cap": cap + 1e-7},
            "scope": "Blind origin-to-score pilot; fitting uses only this run's own proposals; no API or workspace incumbent reads",
        }

    def repair_one():
        nonlocal exact_spent
        from translation_union import make_translation_domains

        already_qualified = [c for c in pending if c["center_hash"] in qualified_centers]
        counts["qualified_repairs_skipped"] += len(already_qualified)
        pending[:] = [c for c in pending if c["center_hash"] not in qualified_centers]
        available = min(EXACT_FRACTION * elapsed() - exact_spent,
                        EXACT_FRACTION * seconds - exact_spent,
                        deadline - time.monotonic())
        if not pending or available < .25:
            return
        # A repeating ladder allocation ensures wide repairs are not starved by
        # an endless stream of new narrow candidates.
        desired_stage = (0, 0, 1, 2)[len(solver_records) % 4]
        choices = [c for c in pending if c["stage"] == desired_stage]
        candidate = min(choices or pending, key=lambda c: (c["loss"], c["serial"]))
        stage = candidate["stage"]
        budget = float(exact_seconds[stage])
        # Accrue the full stage allowance before starting, including a small
        # fitting/build reserve. Actual elapsed cost is charged on every path.
        if available < budget + .15:
            return
        candidate = take_candidate(pending, desired_stage)
        begin = time.monotonic()
        width = ANGLE_LADDER[stage]
        attempt = {"candidate": candidate["serial"], "center_hash": candidate["center_hash"],
                   "angle_width": width, "native_loss": candidate["loss"],
                   "source_frame": candidate["frame"]}
        try:
            if "physical" not in candidate:
                physical, fit = fit_frame(stream.model.p, candidate["geometry"],
                                           model=stream.model, branch=branch.tolist())
                if not fit["success"] or max(fit["section"]["violation"], fit["section"]["residual"]) > 1e-9:
                    attempt["status"] = "FIT_FAILED"
                    return
                candidate["physical"], candidate["fit"] = physical, fit
            domains = make_translation_domains(candidate["physical"], xy_width=1,
                                                angle_width=width, max_poses=2000)
            fingerprint = domain_hash(domains)
            attempt.update(domain_sha256=fingerprint, domain_sizes=list(map(len, domains)), fit=candidate["fit"])
            if fingerprint in seen_domains:
                counts["duplicate_domains"] += 1
                attempt["status"] = "DUPLICATE_DOMAIN"
            else:
                seen_domains.add(fingerprint)
                remaining = min(budget, deadline - time.monotonic(),
                                available - (time.monotonic() - begin))
                if remaining <= .05:
                    attempt["status"] = "STAGE_BUDGET"
                else:
                    result = solve_phase(
                        domains, scored_once(candidate["physical"]),
                        cap + SCORE_GUARD - ACCEPTANCE_GUARD,
                        output / "exact" / f"attempt-{len(solver_records):04d}",
                        seconds=remaining, seed=seed + len(solver_records), local_only=True)
                    attempt["status"] = result["status"]
                    attempt["solver"] = result
                    if result.get("valid"):
                        returned_key = center_hash(np.asarray(result["raw_placements"]))
                        attempt["returned_center_known_before"] = returned_key in valid_centers
                        attempt["returned_center_qualified_before"] = returned_key in qualified_centers
                        register(np.asarray(result["raw_placements"]), "exact")
            if stage + 1 < len(ANGLE_LADDER):
                candidate["stage"] += 1
                pending.append(candidate)
        except (ValueError, np.linalg.LinAlgError) as error:
            attempt.update(status="REPAIR_ERROR", error=str(error))
        finally:
            attempt["seconds"] = time.monotonic() - begin
            solver_records.append(attempt)
            emit("exact", **attempt)
            attempt["seconds"] = time.monotonic() - begin
            exact_spent += attempt["seconds"]

    while time.monotonic() < deadline:
        batch = stream.next()
        if batch is None:
            counts["rejected_frames"] += 1
            continue
        frame, actual, coefficients, metadata = batch
        frame_records.append(metadata)
        counts["frames"] += 1
        for key in ("proposals", "branch_rejected", "duplicates", "ambiguous"):
            counts[key] += metadata[key]
        if not len(actual):
            continue
        areas, circles, _ = evaluate_batch(actual, centers=[0., 0.])
        counts["unique_evaluated"] += len(actual)
        overlap = np.maximum((areas - 1e-6) / frame.c.scale[None, :], 0.).max(axis=1)
        radius_loss = np.maximum(circles[:, 2] - cap, 0.)
        loss = np.maximum(overlap, radius_loss)
        eligible = np.flatnonzero((areas.max(axis=1) <= 1e-6 + 1e-11) & (circles[:, 2] <= cap + 1e-7))
        for index in eligible[np.argsort(circles[eligible, 2])]:
            if time.monotonic() >= deadline:
                break
            register(frame.raw(coefficients[index]), "native")
        if method == "hybrid":
            contenders = np.flatnonzero((overlap <= 30e-9) & (circles[:, 2] <= cap + 30e-9))
            for index in contenders[np.argsort(loss[contenders])][:2]:
                key = center_hash(actual[index])
                if key in seen_candidates or key in qualified_centers:
                    continue
                seen_candidates.add(key)
                serial = counts["candidate_nominations"]
                counts["candidate_nominations"] += 1
                pending.append({"geometry": actual[index].copy(), "loss": float(loss[index]),
                                "center_hash": key, "serial": serial, "stage": 0,
                                "frame": metadata["index"]})
            # Bounded queue; preserve wider pending jobs as well as best fresh
            # centers. Dropped proposals are documented by the count difference.
            if len(pending) > 64:
                counts["queue_dropped"] += len(pending) - 64
                pending[:] = sorted(pending, key=lambda c: (c["stage"] == 0, c["loss"]))[:64]
            repair_one()
        if time.monotonic() - last_report >= 10:
            write_json(output / "report.json", summary())
            emit("progress", best_score=best_score, first_qualifying=first_qualifying,
                 unique_valid_centers=len(valid_centers), qualifying_centers=len(qualified_centers),
                 counts=counts.copy(), exact_seconds=exact_spent)
            last_report = time.monotonic()
    result = summary()
    write_json(output / "report.json", result)
    emit("done", best_score=best_score, first_qualifying=first_qualifying,
         unique_valid_centers=len(valid_centers), qualifying_centers=len(qualified_centers),
         counts=counts, exact_seconds=exact_spent)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", choices=("native", "hybrid"), default="native")
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--seed", type=int, default=3111001)
    parser.add_argument("--exact-seconds", type=float, nargs=3, default=DEFAULT_EXACT_SECONDS,
                        metavar=("NARROW", "MEDIUM", "WIDE"))
    args = parser.parse_args()
    run(args.problem, args.output, method=args.method, seconds=args.seconds,
        seed=args.seed, exact_seconds=args.exact_seconds)


if __name__ == "__main__":
    main()
