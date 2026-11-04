"""A bounded production probe from the verified incumbent at a harder radius cap."""
from __future__ import annotations

import os
for _key in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from benchmark_recovery import DEFAULT_MANIFEST, digest, fit_frame, scored_once, verify_fixture, write_json
from experiment_mirror_family import FLATS, Manifold
from run_blind_pilot import ROOT, export_bundle, freeze_prior, grade_arm
from semicircle_packing.polish import checked, load


def run(output, candidate, leaderboard, *, seed=3111401):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    raw = load(candidate)
    validation = checked(raw)
    if not validation.valid:
        raise ValueError("Starting incumbent must pass the reference validator")
    board = json.loads(Path(leaderboard).read_text())
    board = board.get("response", board)
    leader = min(float(entry["score"]) for entry in board["entries"])
    manifest = json.loads(DEFAULT_MANIFEST.read_text())
    model = Manifold(verify_fixture(DEFAULT_MANIFEST.parent, manifest["seed"]))
    scored = scored_once(raw)
    branch = [round(float(scored[i, 2] - scored[j, 2]), 6) for i, j in FLATS]
    physical, fit = fit_frame(model.p, scored, model=model, branch=branch)
    if not fit["success"] or max(fit["section"]["violation"], fit["section"]["residual"]) > 1e-9:
        raise ValueError("Incumbent did not fit a valid continuous source frame")
    # The worker adds 1e-11 when grading. This leaves a strict 1e-11 margin
    # below the recorded leader; only a fresh arena check can establish a win.
    cap = leader - 2e-11
    problem = {"id": "first-target", "origin": physical.tolist(), "branch": branch, "score_cap": cap}
    prior = freeze_prior(output)
    bundle, _ = export_bundle(output, DEFAULT_MANIFEST)
    problem_path = bundle / "problems/first-target.json"
    write_json(problem_path, problem)
    write_json(output / "problem.json", problem)
    plan = {"scope": "Production probe using the known incumbent; not a blind benchmark or equal-budget method comparison",
            "candidate": str(candidate), "candidate_sha256": digest(candidate), "starting_score": validation.score,
            "leaderboard_sha256": digest(leaderboard), "recorded_leader": leader, "score_cap": cap,
            "effective_qualification_cap": cap + 1e-11, "seed": seed, "fit": fit,
            "budgets": {"native": 150, "hybrid": 90}, "exact_seconds_per_width": [2, 4, 10],
            "hypothesis": "Concentrate the shared native schedule around the newly verified packing; keep exact repair a minority of the fixed four-minute search budget"}
    write_json(output / "preregistered.json", plan)
    report = {"plan": plan, "arms": []}
    environment = dict(os.environ, PYTHONPATH=str(bundle / "src"),
                       VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    for method, seconds in plan["budgets"].items():
        folder = output / method
        command = [sys.executable, str(bundle / "scripts/blind_worker_guard.py"),
                   "--problem", str(problem_path), "--output", str(folder), "--original-repo", str(ROOT),
                   "--method", method, "--seconds", str(seconds), "--seed", str(seed),
                   "--exact-seconds", "2", "4", "10"]
        write_json(output / f"{method}-execution.json", {"command": command, "cwd": str(bundle)})
        print(json.dumps({"event": "probe_start", "method": method, "seconds": seconds,
                          "target": cap}), flush=True)
        before = time.monotonic()
        with (output / f"{method}.log").open("w") as log:
            result = subprocess.run(command, cwd=bundle, env=environment, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=seconds + 45)
        process_seconds = time.monotonic() - before
        if result.returncode:
            raise RuntimeError(f"Probe worker failed: {output / (method + '.log')}")
        before_grade = time.monotonic()
        grade = grade_arm(folder, prior, seconds, cap)
        grade.update(process_seconds=process_seconds, parent_grading_seconds=time.monotonic() - before_grade)
        report["arms"].append(grade)
        write_json(output / "report.json", report)
        print(json.dumps({"event": "probe_done", "method": method, "best_score": grade["best_score"],
                          "first_qualifying_seconds": grade["first_qualifying_seconds"]}), flush=True)
        if grade["best_score"] is not None and grade["best_score"] < leader:
            report["scope_note"] = "Stopped after a locally winning candidate; arena validation still required"
            break
    report["best_score"] = min((a["best_score"] for a in report["arms"] if a["best_score"] is not None), default=None)
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, default=ROOT / "solution.json")
    parser.add_argument("--leaderboard", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3111401)
    args = parser.parse_args()
    run(args.output, args.candidate, args.leaderboard, seed=args.seed)


if __name__ == "__main__":
    main()
