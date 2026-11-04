"""Freeze blind inputs, run paired workers sequentially, and grade independently."""
from __future__ import annotations

import os
for _key in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from benchmark_recovery import DEFAULT_MANIFEST, digest, verify_fixture, write_json
from semicircle_packing.polish import checked, load

ROOT = Path(__file__).resolve().parents[1]
GUARD = 1e-11


def center_keys(p):
    """Both labeled translation and unlabeled square-symmetry center identities."""
    xy = np.rint(np.asarray(p)[:, :2] * 1e6).astype(np.int64)
    labeled = hashlib.sha256((xy - xy[0]).astype("<i8").tobytes()).hexdigest()
    forms = []
    for swap in (False, True):
        axes = xy[:, ::-1] if swap else xy
        for sx in (-1, 1):
            for sy in (-1, 1):
                q = axes * [sx, sy]
                q = q[np.lexsort((q[:, 1], q[:, 0]))]
                forms.append((q - q[0]).astype("<i8").tobytes())
    return labeled, hashlib.sha256(min(forms)).hexdigest()


def freeze_prior(output):
    roots = [ROOT / "experiments/2026-11-04", ROOT / "tests/fixtures"]
    paths = list((ROOT / "experiments").glob("*.json"))
    for folder in roots:
        paths.extend(folder.rglob("*.json"))
    labeled, dihedral, sources = set(), set(), []
    for path in sorted(set(paths)):
        try:
            if path.stat().st_size > 32768:
                continue
            raw = path.read_bytes()
            data = json.loads(raw)
            if not isinstance(data, list) or len(data) != 15:
                continue
            p = np.array([[v[k] for k in ("x", "y", "theta")] for v in data], dtype=float)
            if p.shape != (15, 3) or not np.isfinite(p).all():
                continue
            a, b = center_keys(p)
            labeled.add(a)
            dihedral.add(b)
            sources.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(raw).hexdigest()})
        except (OSError, TypeError, ValueError, KeyError):
            continue
    record = {"definition": "XY integer microunits; labeled/common translation and unlabeled/D4/common translation",
              "scope": "Frozen observed archive, including invalid and rounded continuous states; not all possible prior geometry",
              "labeled": sorted(labeled), "dihedral": sorted(dihedral), "sources": sources}
    write_json(output / "prior-centers.json", record)
    return labeled, dihedral


def export_bundle(output, manifest_path):
    bundle = Path(tempfile.mkdtemp(prefix="packing-blind-", dir="/private/tmp"))
    sources = {}
    for folder in (ROOT / "scripts", ROOT / "src"):
        for source in sorted(folder.rglob("*")):
            if not source.is_file() or source.suffix not in (".py", ".cpp"):
                continue
            relative = source.relative_to(ROOT)
            destination = bundle / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            sources[str(relative)] = digest(source)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    problems = []
    for case in manifest["cases"]:
        if case["split"] != "heldout":
            continue
        origin = verify_fixture(manifest_path.parent, case["saved_origin"])
        problem = {"id": case["id"], "origin": origin.tolist(),
                   "branch": case["provenance"]["branch"], "score_cap": case["expected_score"]}
        path = bundle / "problems" / (case["id"] + ".json")
        write_json(path, problem)
        write_json(output / "problems" / path.name, problem)
        problems.append({"id": case["id"], "path": str(path), "sha256": digest(path),
                         "score_cap": problem["score_cap"], "origin_sha256": case["saved_origin"]["sha256"]})
    if len(problems) != 3:
        raise ValueError("This preregistered pilot requires exactly three held-out origins")
    write_json(output / "bundle.json", {"path": str(bundle), "source_hashes": sources,
                                        "manifest_sha256": digest(manifest_path), "problems": problems})
    return bundle, problems


def grade_arm(folder, prior, seconds, score_cap):
    worker = json.loads((folder / "report.json").read_text())
    if worker["score_cap"] != score_cap or worker["budget_seconds"] != seconds:
        raise AssertionError("Worker changed the independently supplied cap or time budget")
    cutoff = float(score_cap) + GUARD
    rows, labels, symmetries, qualified = [], set(), set(), []
    best = None
    for record in worker["valid_records"]:
        p = load(folder / record["file"])
        result = checked(p)
        if not result.valid or abs(result.score - record["score"]) > GUARD:
            raise AssertionError(f"Worker/reference mismatch: {folder / record['file']}")
        if record["seconds"] > seconds:
            raise AssertionError("Worker credited a late result")
        a, b = center_keys(p)
        if a != record["center_hash"]:
            raise AssertionError("Worker center hash mismatch")
        qualifies = result.score <= cutoff
        if qualifies != record["qualifying"]:
            raise AssertionError("Worker qualification cutoff mismatch")
        labels.add(a)
        symmetries.add(b)
        best = result.score if best is None else min(best, result.score)
        row = {**record, "reference_score": result.score, "dihedral_hash": b,
               "new_labeled_center": a not in prior[0], "new_dihedral_center": b not in prior[1]}
        rows.append(row)
        if qualifies:
            qualified.append(row)
    first = min((r["seconds"] for r in qualified), default=None)
    audit = json.loads((folder / "access-audit.json").read_text())
    if audit["blocked_reads"] or audit["network_attempts"]:
        raise AssertionError(f"Blind access audit failed: {folder}")
    return {
        "method": worker["method"], "case": worker["problem_id"], "best_score": best,
        "first_qualifying_seconds": first, "censored_seconds": first if first is not None else seconds,
        "distinct_valid_centers": len(labels), "distinct_dihedral_valid_centers": len(symmetries),
        "qualifying_centers": len({r["center_hash"] for r in qualified}),
        "new_qualifying_centers": len({r["center_hash"] for r in qualified if r["new_labeled_center"]}),
        "new_dihedral_qualifying_centers": len({r["dihedral_hash"] for r in qualified if r["new_dihedral_center"]}),
        "counts": worker["counts"], "exact_seconds": worker["exact_seconds"],
        "exact_budget_overrun": worker["exact_budget_overrun"],
        "stage_statuses": [{"angle_width": r["angle_width"], "status": r["status"], "seconds": r["seconds"]}
                           for r in worker["solver_records"]],
        "worker_seconds": worker["seconds"], "warmup_seconds": worker.get("warmup_seconds"),
        "access_audit": audit, "records": rows,
    }


def decide(pairs):
    native_hits = sum(p["native"]["first_qualifying_seconds"] is not None for p in pairs)
    hybrid_hits = sum(p["hybrid"]["first_qualifying_seconds"] is not None for p in pairs)
    latency_wins = sum(p["hybrid"]["censored_seconds"] < p["native"]["censored_seconds"] for p in pairs)
    native_time = sum(p["native"]["censored_seconds"] for p in pairs)
    hybrid_time = sum(p["hybrid"]["censored_seconds"] for p in pairs)
    regression = any(p["native"]["best_score"] is not None and
                     (p["hybrid"]["best_score"] is None or p["hybrid"]["best_score"] > p["native"]["best_score"] + GUARD)
                     for p in pairs)
    gain = 1 - hybrid_time / native_time if native_time > 0 else 0.
    go = hybrid_hits >= native_hits and latency_wins >= 2 and gain >= .2 and not regression
    return {"decision": "GO" if go else "NO_GO", "native_qualifying_cases": native_hits,
            "hybrid_qualifying_cases": hybrid_hits, "hybrid_latency_wins": latency_wins,
            "capped_latency_improvement_fraction": gain, "material_best_score_regression": regression,
            "scope": "A three-case pilot of this exact-repair funnel; not a general rejection of all exact or translation-union searches"}


def run(output, *, manifest_path=DEFAULT_MANIFEST, seconds=120, seed=3111201, exact_seconds=(2., 4., 10.)):
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Duration must be positive and finite")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    prior = freeze_prior(output)
    bundle, problems = export_bundle(output, manifest_path)
    plan = {
        "scope": "Three paired blind origin-to-score cases; shared native proposal schedule, hybrid exact repair",
        "command": sys.argv,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "shapely", "ortools", "fpylll")}},
        "seconds_per_arm": seconds, "base_seed": seed, "exact_seconds_per_width": list(exact_seconds),
        "score_guard": GUARD, "exact_fraction_limit": .25,
        "order": [[p["id"], method] for i, p in enumerate(problems)
                  for method in (("native", "hybrid") if i % 2 == 0 else ("hybrid", "native"))],
        "go_rule": "Hybrid qualifies on >= native cases, wins latency in >=2/3 pairs, improves aggregate censored latency >=20%, and has no per-case best-score regression >1e-11",
        "prior_centers_sha256": digest(output / "prior-centers.json"),
        "prior_labeled_centers": len(prior[0]), "prior_dihedral_centers": len(prior[1]),
        "attribution": "Exact repair may change the two local flat differences; native proposals preserve them",
    }
    write_json(output / "preregistered.json", plan)
    report = {"plan": plan, "arms": [], "pairs": []}
    environment = dict(os.environ, PYTHONPATH=str(bundle / "src"),
                       VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    for index, problem in enumerate(problems):
        pair = {"case": problem["id"]}
        methods = ("native", "hybrid") if index % 2 == 0 else ("hybrid", "native")
        for method in methods:
            folder = output / problem["id"] / method
            log = output / f"{problem['id']}-{method}.log"
            command = [sys.executable, str(bundle / "scripts/blind_worker_guard.py"),
                       "--problem", problem["path"], "--output", str(folder),
                       "--original-repo", str(ROOT), "--method", method,
                       "--seconds", str(seconds), "--seed", str(seed + index * 100),
                       "--exact-seconds", *map(str, exact_seconds)]
            print(json.dumps({"event": "arm_start", "case": problem["id"], "method": method,
                              "budget_seconds": seconds, "log": str(log)}), flush=True)
            before = time.monotonic()
            write_json(output / f"{problem['id']}-{method}-execution.json", {
                "command": command, "cwd": str(bundle), "problem_sha256": problem["sha256"],
                "threads": 1, "budget_seconds": seconds})
            with log.open("w") as stream:
                child = subprocess.run(command, cwd=bundle, env=environment, stdout=stream,
                                       stderr=subprocess.STDOUT, timeout=seconds + 45)
            process_seconds = time.monotonic() - before
            if child.returncode:
                raise RuntimeError(f"Worker failed ({child.returncode}); inspect {log}")
            before_grade = time.monotonic()
            grade = grade_arm(folder, prior, seconds, problem["score_cap"])
            grade["process_seconds"] = process_seconds
            grade["parent_grading_seconds"] = time.monotonic() - before_grade
            pair[method] = grade
            report["arms"].append(grade)
            write_json(output / "report.json", report)
            print(json.dumps({"event": "arm_done", **{k: grade[k] for k in (
                "case", "method", "best_score", "first_qualifying_seconds", "new_dihedral_qualifying_centers")}}), flush=True)
        a = json.loads((output / problem["id"] / "native/report.json").read_text())["frame_records"]
        b = json.loads((output / problem["id"] / "hybrid/report.json").read_text())["frame_records"]
        shared = min(len(a), len(b))
        if shared == 0 or any(x["index"] != y["index"] or x["proposal_hash"] != y["proposal_hash"]
                              for x, y in zip(a, b)):
            raise AssertionError("Native proposal prefixes diverged between paired arms")
        pair["verified_shared_frames"] = shared
        report["pairs"].append(pair)
        write_json(output / "report.json", report)
    report["decision"] = decide(report["pairs"])
    write_json(output / "report.json", report)
    print(json.dumps(report["decision"], indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds-per-arm", type=float, default=120)
    parser.add_argument("--seed", type=int, default=3111201)
    parser.add_argument("--exact-seconds", type=float, nargs=3, default=(2, 4, 10))
    args = parser.parse_args()
    run(args.output, seconds=args.seconds_per_arm, seed=args.seed, exact_seconds=tuple(args.exact_seconds))


if __name__ == "__main__":
    main()
