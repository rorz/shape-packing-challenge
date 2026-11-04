"""Target-assisted control replay for translation unions and angular widths.

This checks representation and exact-model cost. The separate blind pilot must
not receive these fitted frames or the known target placements.
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

import numpy as np

from benchmark_recovery import (
    DEFAULT_MANIFEST, SCORE_GUARD, digest, fit_frame, membership, placements_json,
    same_centers, scored_once, verify_fixture, write_json,
)
from experiment_lazy_phases import make_domains
from experiment_mirror_family import FLATS, Manifold
from lazy_phase_solver import solve_phase
from translation_union import make_translation_domains


def run(manifest_path, output, *, seconds=15, case_ids=None):
    manifest_path, output = Path(manifest_path), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text())
    model = Manifold(verify_fixture(manifest_path.parent, manifest["seed"]))
    cases = [case for case in manifest["cases"] if case_ids is None or case["id"] in case_ids]
    if not cases:
        raise ValueError("No matching controls")
    started = time.perf_counter()
    report = {
        "scope": "Target-assisted replay only; not blind discovery",
        "manifest_sha256": digest(manifest_path),
        "source_hashes": {name: digest(Path(__file__).parent / name) for name in (
            "benchmark_translation_union.py", "translation_union.py", "benchmark_recovery.py",
            "experiment_lazy_phases.py", "lazy_phase_solver.py", "lazy_pair_oracle.py")},
        "seconds_per_model": seconds, "rows": [],
    }
    for case in cases:
        raw = verify_fixture(manifest_path.parent, case, "target")
        target = scored_once(raw)
        branch = [round(float(target[i, 2] - target[j, 2]), 6) for i, j in FLATS]
        physical, fit = fit_frame(model.p, target, model=model, branch=branch)
        if not fit["success"] or max(fit["section"]["violation"], fit["section"]["residual"]) > 1e-9:
            raise ValueError(f"Failed control fit: {case['id']}")
        write_json(output / case["id"] / "fitted-frame.json", placements_json(physical))
        base = scored_once(physical)
        for width in (1, 3, 12):
            for method in ("single-phase", "translation-union"):
                before = time.perf_counter()
                domains = (make_domains(base, 1, width) if method == "single-phase"
                           else make_translation_domains(physical, xy_width=1, angle_width=width))
                build_seconds = time.perf_counter() - before
                covered = membership(domains, target)
                if not covered["covered"]:
                    raise AssertionError(f"Control missing: {case['id']}/{method}/{width}: {covered}")
                folder = output / case["id"] / f"{method}-angle-{width}"
                result = solve_phase(domains, base, case["expected_score"] + SCORE_GUARD,
                                     folder, seconds=seconds, seed=3111101, local_only=True)
                row = {
                    "case": case["id"], "method": method, "angle_width": width,
                    "target_sha256": case["sha256"], "target_score": case["expected_score"],
                    "fit": fit, "covered": True, "domain_sizes": list(map(len, domains)),
                    "domain_sha256": hashlib.sha256(b"".join(
                        np.rint(d * 1e6).astype("<i8").tobytes() for d in domains)).hexdigest(),
                    "domain_build_seconds": build_seconds,
                    "status": result["status"], "valid": result.get("valid", False),
                    "score": result.get("score"), "solve_seconds": result["seconds"],
                    "oracle": result.get("oracle"), "counts": result.get("counts"),
                }
                if result.get("valid"):
                    row["same_target_centers"] = same_centers(np.asarray(result["scored_placements"]), target)
                report["rows"].append(row)
                report["seconds"] = time.perf_counter() - started
                write_json(output / "report.json", report)
                print(json.dumps({k: row[k] for k in (
                    "case", "method", "angle_width", "status", "score", "solve_seconds")}), flush=True)
    report["summary"] = {
        "models": len(report["rows"]), "covered": sum(row["covered"] for row in report["rows"]),
        "recovered": sum(row["valid"] for row in report["rows"]),
        "seconds": time.perf_counter() - started,
    }
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--seconds-per-model", type=float, default=15)
    args = parser.parse_args()
    report = run(args.manifest, args.output, seconds=args.seconds_per_model, case_ids=args.cases)
    print(json.dumps(report["summary"], indent=2))
    return int(report["summary"]["recovered"] != report["summary"]["models"])


if __name__ == "__main__":
    raise SystemExit(main())
