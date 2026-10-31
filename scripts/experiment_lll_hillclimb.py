"""Nonlinear local search over the joint integer moves of an LLL checkpoint."""
from __future__ import annotations

import os
for variable in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np

from arena import check_solution
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.polish import AreaConstraints, checked, save


LEADER = 2.9485269238856113


def canonical_units(units):
    result = units.copy()
    theta = result[:, 2] * 1e-6
    result[:, 2] = np.rint(np.arctan2(np.sin(theta), np.cos(theta)) * 1e6).astype(np.int64)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("experiments/lll_basis.npz"))
    parser.add_argument("--output", type=Path, default=Path("experiments/lll_hillclimb_best.json"))
    parser.add_argument("--seconds", type=float, default=90)
    parser.add_argument("--seed", type=int, default=7102)
    parser.add_argument("--combinations", type=int, default=120)
    parser.add_argument("--stagnation", type=int, default=6)
    parser.add_argument("--fixed-center", action="store_true")
    parser.add_argument("--target", type=float)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    started = time.monotonic()
    evaluations = iterations = restarts = 0
    best_loss = float("inf")
    best_violation = float("inf")
    next_report = started + 10
    checkpoint_mtime = None
    current = None
    local_best = None
    constraint = None
    moves = None
    target = None
    origin = None
    stale = 0
    last_center = np.zeros(2)

    def evaluate(units, cutoff=None):
        nonlocal evaluations, last_center
        candidate = canonical_units(units)
        placements = candidate * 1e-6
        gaps, _ = constraint.evaluate(np.r_[placements.ravel(), target])
        if not args.fixed_center:
            area_negative = np.minimum(gaps[:105], 0)
            area_loss = float(area_negative @ area_negative)
            if cutoff is not None and area_loss >= cutoff:
                evaluations += 1
                return area_loss, float(-gaps[:105].min()), candidate
            cx, cy, _ = compute_mec_fast(placements, initial_center=last_center)
            last_center = np.array([cx, cy])
            centered = placements.copy()
            centered[:, :2] -= last_center
            gaps, _ = constraint.evaluate(np.r_[centered.ravel(), target])
        negative = np.minimum(gaps, 0)
        evaluations += 1
        return float(negative @ negative), float(-gaps.min()), candidate

    def record_best(units, loss, violation):
        nonlocal best_loss, best_violation
        if loss >= best_loss and violation >= best_violation:
            return
        best_loss = min(best_loss, loss)
        best_violation = min(best_violation, violation)
        candidate = units * 1e-6
        save(args.output.with_name(args.output.stem + "_nearest.json"), candidate)
        snapshot = args.output.with_name(args.output.stem + "_state.npz")
        temporary = snapshot.with_suffix(".tmp.npz")
        np.savez(temporary, placements=candidate, origin=origin, moves=moves,
                 target=target, loss=loss, violation=violation)
        temporary.replace(snapshot)
        print(json.dumps(dict(event="best", loss=loss, violation=violation,
                              best_violation=best_violation, evaluations=evaluations,
                              iterations=iterations, seconds=time.monotonic()-started)), flush=True)

    while time.monotonic() - started < args.seconds:
        mtime = args.checkpoint.stat().st_mtime_ns
        if checkpoint_mtime != mtime:
            with np.load(args.checkpoint) as state:
                current = canonical_units(np.rint(state["placements"] * 1e6).astype(np.int64))
                origin = state["origin"].copy()
                moves = np.rint(state["moves"]).astype(np.int64)
                target = float(state["target"]) if args.target is None else args.target
            constraint = AreaConstraints(origin)
            checkpoint_mtime = mtime
            loss, violation, current = evaluate(current)
            local_best = current.copy()
            local_loss = loss
            stale = 0
            restarts += 1
            print(json.dumps(dict(event="checkpoint", loss=loss, violation=violation,
                                  seconds=time.monotonic()-started)), flush=True)
            record_best(current, loss, violation)

        loss, violation, current = evaluate(current)
        candidate_loss, candidate_violation, candidate_units = loss, violation, current
        # Every basis direction is tried in both signs. Random sums reach
        # diagonals that a one-direction descent cannot cross.
        directions = list(moves) + list(-moves)
        ranked = []
        for direction in directions:
            trial_loss, trial_violation, trial_units = evaluate(current + direction, candidate_loss)
            ranked.append((trial_loss, direction))
            if trial_loss < candidate_loss:
                candidate_loss, candidate_violation, candidate_units = trial_loss, trial_violation, trial_units
        ranked.sort(key=lambda item: item[0])
        soft = [direction for _, direction in ranked[:18]]
        combinations = [a+b for a,b in itertools.combinations(soft[:12], 2)]
        if candidate_loss >= loss:
            combinations.extend(a+b+c for a,b,c in itertools.combinations(soft[:12], 3))
        for _ in range(args.combinations):
            count = int(rng.integers(2, 4))
            pool = soft if rng.random() < .8 else directions
            ids = rng.choice(len(pool), count, replace=False)
            combinations.append(np.sum([pool[index] for index in ids], axis=0))
        for direction in combinations:
            if not direction.any():
                continue
            trial_loss, trial_violation, trial_units = evaluate(current + direction, candidate_loss)
            if trial_loss < candidate_loss:
                candidate_loss, candidate_violation, candidate_units = trial_loss, trial_violation, trial_units
        iterations += 1
        if candidate_loss < loss * (1 - 1e-10) - 1e-28:
            current = candidate_units
            loss, violation = candidate_loss, candidate_violation
            stale = 0
        else:
            stale += 1
            # Restart around the best point in this frame after several kicks.
            if stale > args.stagnation:
                current = local_best.copy()
                stale = 0
                restarts += 1
            ids = rng.choice(len(moves), int(rng.integers(1, 5)), replace=False)
            signs = rng.choice([-1, 1], len(ids))
            current = canonical_units(current + np.sum(moves[ids] * signs[:, None, None], axis=0))
            loss, violation, current = evaluate(current)
        if loss < local_loss:
            local_loss, local_best = loss, current.copy()
        record_best(current, loss, violation)
        if violation < 1e-8:
            candidate = current * 1e-6
            validation = checked(candidate)
            if validation.valid and validation.score < LEADER:
                save(args.output, candidate)
                response = check_solution([dict(x=float(x), y=float(y), theta=float(theta))
                                           for x, y, theta in candidate])
                record = args.output.with_name(args.output.stem + "_api.json")
                record.write_text(json.dumps(response, indent=2) + "\n")
                print(json.dumps(dict(event="api", **response)), flush=True)
                if response["valid"] and response["score"] < LEADER:
                    return
        if time.monotonic() >= next_report:
            print(json.dumps(dict(event="progress", evaluations=evaluations,
                                  iterations=iterations, restarts=restarts,
                                  best_loss=best_loss, best_violation=best_violation,
                                  seconds=time.monotonic()-started)), flush=True)
            next_report = time.monotonic() + 10
    print(json.dumps(dict(event="done", evaluations=evaluations, iterations=iterations,
                          best_loss=best_loss, best_violation=best_violation,
                          seconds=time.monotonic()-started)), flush=True)


if __name__ == "__main__":
    main()
