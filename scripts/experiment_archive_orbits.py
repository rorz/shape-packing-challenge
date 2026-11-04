"""Repair frozen archive angle realizations; retain a finite coverage ledger."""
from __future__ import annotations

import os
for _key in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

from experiment_angle_orbits import d4_matrices, make_angle_domains, scored_once, transform_scored
from lazy_phase_solver import ACCEPTANCE_GUARD, solve_phase
from semicircle_packing.polish import checked, load
from translation_union import domain_digest

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def pose_sets(domains):
    return [frozenset(map(tuple, np.rint(d * 1e6).astype(np.int64))) for d in domains]


def proof_covers(proved, proved_target, proposed, target):
    """Equal numerical policy required by the caller; larger caps are looser."""
    return proved_target >= target and len(proved) == len(proposed) and all(
        new <= old for old, new in zip(proved, proposed))


def load_prior(report_path, target):
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    if report["acceptance_guard"] != ACCEPTANCE_GUARD:
        raise ValueError("Prior proof uses a different acceptance policy")
    for name in ("lazy_phase_solver.py", "lazy_pair_oracle.py"):
        if report["source_hashes"][name] != digest(ROOT / "scripts" / name):
            raise ValueError(f"Prior proof's solver changed: {name}")
    proofs = []
    for serial, model in enumerate(report["models"]):
        result = model.get("result", {})
        if model["status"] != "INFEASIBLE" or not result.get("finite_exhausted"):
            continue
        if result["target"] < target:
            continue
        path = report_path.parent / f"model-{serial:03d}" / "domains.npz"
        with np.load(path) as archive:
            domains = [archive[f"p{i}"] for i in range(len(archive.files))]
        if domain_digest(domains) != model["domain_sha256"]:
            raise ValueError(f"Changed prior domain: {path}")
        proofs.append((pose_sets(domains), result["target"],
                       {"report": str(report_path), "model": serial,
                        "report_sha256": digest(report_path), "domain_sha256": model["domain_sha256"]}))
    return proofs


def run(census_path, output, *, target, seconds=300, per_model=2,
        widths=(3, 12), prior_reports=(), seed=3111601):
    if not 0 < seconds <= 300 or per_model <= 0 or not np.isfinite(target):
        raise ValueError("Use a positive budget at most300 seconds and finite target")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Use a new output directory")
    output.mkdir(parents=True)
    (output / "inputs").mkdir()
    (output / "source-snapshot").mkdir()
    hashes = {}
    for name in ("experiment_archive_orbits.py", "experiment_angle_orbits.py",
                 "lazy_phase_solver.py", "lazy_pair_oracle.py", "separable_support.py",
                 "experiment_support_bundle.py", "experiment_conflict_master.py"):
        path = ROOT / "scripts" / name
        hashes[name] = digest(path)
        shutil.copy2(path, output / "source-snapshot" / name)
    census_path = Path(census_path)
    census = json.loads(census_path.read_text())
    sources, seen_files = [], set()
    for row in census["under_50nm_class_shortlist"]:
        for saved in row["same_center_record_paths"]:
            path = ROOT / Path(saved).with_suffix(".json")
            sha = digest(path)
            if sha in seen_files:
                continue
            seen_files.add(sha)
            raw = load(path)
            if raw.shape != (15, 3):
                raise ValueError(f"Invalid input shape: {path}")
            frozen = output / "inputs" / f"source-{len(sources):03d}.json"
            shutil.copy2(path, frozen)
            sources.append((scored_once(raw), {"path": str(path.relative_to(ROOT)),
                "sha256": sha, "frozen_path": str(frozen.relative_to(output)),
                "class_deficit_billionths": row["normalized_max_geos_deficit_nm"]}))
    proofs = [proof for path in prior_reports for proof in load_prior(path, target)]
    report = {"scope": "Finite fixed-center angular repair, all saved angular realizations of13 shortlisted classes; no API/promotion",
              "target": target, "acceptance_guard": ACCEPTANCE_GUARD,
              "budget_seconds": seconds, "per_model_seconds": per_model, "angle_widths": list(widths),
              "seed": seed, "source_hashes": hashes, "census_sha256": digest(census_path),
              "inputs": [record for _, record in sources], "prior_proofs": len(proofs), "models": []}
    seen = {}
    scheduled = len(widths) * 8 * len(sources)
    started = time.monotonic()
    stop = False
    for width in widths:
        for orbit, matrix in d4_matrices():
            for index, (geometry, source) in enumerate(sources):
                remaining = seconds - (time.monotonic() - started)
                if remaining <= 0:
                    stop = True
                    break
                domains = make_angle_domains(transform_scored(geometry, matrix), width)
                sha = domain_digest(domains)
                sets = pose_sets(domains)
                row = {"input_index": index, "input_sha256": source["sha256"], "orbit": orbit,
                       "angle_width": width, "domain_sha256": sha, "domain_sizes": list(map(len, domains))}
                covered = next((evidence for prior, cap, evidence in proofs
                                if proof_covers(prior, cap, sets, target)), None)
                serial = len(report["models"])
                if covered:
                    row.update(status="COVERED_BY_PRIOR_PROOF", proof=covered)
                elif sha in seen:
                    row.update(status="DUPLICATE_DOMAIN", duplicate_of=seen[sha])
                else:
                    seen[sha] = serial
                    folder = output / f"model-{serial:04d}"
                    result = solve_phase(domains, np.asarray([d[0] for d in domains]), target,
                                         folder, seconds=min(per_model, remaining),
                                         seed=seed + serial, local_only=True)
                    row.update(status=result["status"], result=result)
                    if result.get("finite_exhausted"):
                        proofs.append((sets, target, {"report": str(output / "report.json"),
                                      "model": serial, "domain_sha256": sha}))
                    if result.get("valid"):
                        validation = checked(load(folder / "solution.json"))
                        if not validation.valid or validation.score > target + ACCEPTANCE_GUARD:
                            raise AssertionError("Independent validation failed")
                        shutil.copy2(folder / "solution.json", output / "best.json")
                        report["best"] = {"score": validation.score, "sha256": digest(output / "best.json"),
                                          "path": "best.json", "model": serial}
                        stop = True
                report["models"].append(row)
                report.update(seconds=time.monotonic() - started, attempted=len(report["models"]),
                              scheduled=scheduled, remaining=scheduled-len(report["models"]))
                write(output / "report.json", report)
                print(json.dumps({"model": serial, "input": index, "orbit": orbit,
                                  "width": width, "status": row["status"], "best": report.get("best")}), flush=True)
                if stop:
                    break
            if stop:
                break
        if stop:
            break
    report.update(seconds=time.monotonic()-started, status="VALID_CANDIDATE" if "best" in report else
                  "COMPLETE" if len(report["models"]) == scheduled else "BUDGET_EXHAUSTED")
    write(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path, default=ROOT / "docs/oblique-archive-census.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=float, default=2.9485269238806113)
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--per-model", type=float, default=2)
    parser.add_argument("--prior-report", type=Path, action="append", default=[])
    args = parser.parse_args()
    result = run(args.census, args.output, target=args.target, seconds=args.seconds,
                 per_model=args.per_model, prior_reports=args.prior_report)
    print(json.dumps({k: result[k] for k in ("status", "seconds", "attempted", "remaining")}, indent=2))


if __name__ == "__main__":
    main()
