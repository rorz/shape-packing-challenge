"""Offline exact recombination across fitted physical deformation frames.

Unlike the old rigid phase pool, each source retains its fitted two physical
coordinates and flat-angle geometry before all sources share one rigid frame.
"""
from __future__ import annotations

import os
for _name in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from benchmark_recovery import fit_frame, placements_json, scored_once, transform
from experiment_mirror_family import FLATS, Manifold
import experiment_support_bundle as support_module
from experiment_conflict_master import cut_record
from lazy_phase_solver import ACCEPTANCE_GUARD, solve_phase
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.polish import checked, load
from translation_union import domain_digest, make_translation_domains

LEADER = 2.9485269238856113
TARGET = LEADER - 5e-12
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path); temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fit_sources(seed, raw_sources):
    model = Manifold(seed)
    records = []; fitted = []
    for raw in raw_sources:
        target = scored_once(raw)
        branch = [float(target[i, 2] - target[j, 2]) for i, j in FLATS]
        _, fit = fit_frame(seed, target, model=model, branch=branch)
        if not fit["success"] or max(fit["section"]["violation"], fit["section"]["residual"]) > 2e-9:
            raise ValueError("A source physical fit failed")
        physical, section = model.section(branch, fit["alpha"])
        fitted.append(physical)
        records.append({"fit": fit, "source_scored_flat_differences": branch, "section": section})
    common = records[0]["fit"]
    aligned = [transform(p, common["angle"], common["translation"]) for p in fitted]
    reference_distances = np.linalg.norm(aligned[0][:, None, :2] - aligned[0][None, :, :2], axis=2)
    for record, p in zip(records, aligned):
        record["aligned_flat_differences"] = [float(p[i, 2] - p[j, 2]) for i, j in FLATS]
        distances = np.linalg.norm(p[:, None, :2] - p[None, :, :2], axis=2)
        record["maximum_labeled_pair_distance_change_from_current"] = float(np.max(np.abs(distances - reference_distances)))
        record["aligned_placements"] = placements_json(p)
    return aligned, records, {"angle": common["angle"], "translation": common["translation"]}


def merge_domains(aligned_frames, incumbent_scored, *, xy_width=1, angle_width=1, max_poses=500):
    """Full finite union, with incumbent rows first and complete source labels."""
    if max_poses < 1:
        raise ValueError("max_poses must be positive")
    local = [make_translation_domains(p, xy_width=xy_width, angle_width=angle_width, max_poses=max_poses)
             for p in aligned_frames]
    domains = []; provenance = []; source_additions = np.zeros(len(local), dtype=int)
    for piece in range(15):
        first = tuple(map(int, np.rint(incumbent_scored[piece] * 1e6)))
        rows = {first: [-1]}
        for source_index, source in enumerate(local):
            for pose in source[piece]:
                key = tuple(map(int, np.rint(pose * 1e6)))
                if key not in rows:
                    if len(rows) >= max_poses:
                        raise ValueError(f"Piece {piece} exceeds max_poses={max_poses}; no domain was truncated")
                    rows[key] = []
                    source_additions[source_index] += 1
                if source_index not in rows[key]:
                    rows[key].append(source_index)
        domains.append(np.asarray(list(rows), dtype=float) / 1e6)
        provenance.append(list(rows.values()))
    return domains, provenance, {"local_domain_sizes": [list(map(len, d)) for d in local],
                                  "source_new_pose_totals_in_input_order": source_additions.tolist()}


def cut_digest(cut):
    return hashlib.sha256(str(cut["offset"]).encode() + b"".join(
        str(i).encode() + values.tobytes() for i, values in sorted(cut["coefficients"].items()))).hexdigest()


def all_frame_supports(domains, anchors):
    result = []; seen = set()
    original = support_module.support_bundle
    for index, anchor in enumerate(anchors):
        for cut in original(domains, anchor, compute_mec_fast(anchor)):
            digest = cut_digest(cut)
            if digest not in seen:
                seen.add(digest)
                result.append(dict(cut, physical_anchor_index=index))
    return result


@contextmanager
def use_supports(cuts):
    original = support_module.support_bundle
    support_module.support_bundle = lambda *_: cuts
    try:
        yield
    finally:
        support_module.support_bundle = original


def assignment_sources(result, provenance):
    if not result.get("valid"):
        return None
    sources = [provenance[i][value] for i, value in enumerate(result["assignment"])]
    common = set(sources[0])
    for row in sources[1:]:
        common &= set(row)
    return {"per_piece_sources": sources, "common_source_domain_indices": sorted(common),
            "requires_cross_source_domain_choices": not bool(common),
            "source_minus_one": "Explicit incumbent pose only"}


def run(origin_path, sources, output, *, seconds=180., control_seconds=30., seed=3111901,
        xy_width=1, angle_width=1, max_poses=500):
    if not 0 < seconds <= 180 or not 0 < control_seconds <= 30 or len(sources) < 2:
        raise ValueError("Use at least two sources, target budget <=180s and control budget <=30s")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be fresh")
    output.mkdir(parents=True, exist_ok=True)
    snapshots = output / "source-snapshot"; snapshots.mkdir()
    source_hashes = {}
    for name in ("experiment_physical_frame_pool.py", "benchmark_recovery.py", "experiment_mirror_family.py",
                 "translation_union.py", "lazy_phase_solver.py", "lazy_pair_oracle.py", "experiment_support_bundle.py",
                 "experiment_conflict_master.py", "separable_support.py", "conflict_area_band.py", "_conflict_area_band.cpp",
                 "interval_order_optimization.py", "interval_pair_relation.py", "_interval_order.cpp", "experiment_hillclimb.py"):
        path = ROOT / "scripts" / name
        shutil.copy2(path, snapshots / name); source_hashes[name] = sha(path)
    core = snapshots / "semicircle_packing"; core.mkdir()
    for path in sorted((ROOT / "src/semicircle_packing").iterdir()):
        if path.suffix in (".py", ".cpp"):
            shutil.copy2(path, core / path.name)
            source_hashes[f"semicircle_packing/{path.name}"] = sha(path)
    raw_sources = []
    inputs = []
    for index, path in enumerate(map(Path, sources)):
        raw = load(path); validation = checked(raw)
        if not validation.valid:
            raise ValueError(f"Expected reference-valid source: {path}")
        raw_sources.append(raw)
        shutil.copy2(path, output / f"input-{index:02d}.json")
        inputs.append({"index": index, "path": str(path), "sha256": sha(path),
                       "reference_valid": True, "reference_score": float(validation.score)})
    shutil.copy2(origin_path, output / "origin.json")
    begin = time.monotonic()
    aligned, fits, common_frame = fit_sources(load(origin_path), raw_sources)
    incumbent = scored_once(raw_sources[0])
    domains, provenance, build = merge_domains(aligned, incumbent, xy_width=xy_width,
        angle_width=angle_width, max_poses=max_poses)
    np.savez_compressed(output / "domains.npz", **{f"p{i}": d for i, d in enumerate(domains)})
    write_json(output / "pose-source-membership.json", provenance)
    cuts = all_frame_supports(domains, [incumbent, *aligned])
    write_json(output / "support-cuts.json", [cut_record(c) for c in cuts])
    report = {"inputs": inputs, "origin": {"path": str(origin_path), "sha256": sha(origin_path)},
              "dependency_versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "shapely", "ortools", "fpylll")},
              "source_hashes": source_hashes, "fits": fits, "common_rigid_frame": common_frame,
              "domain_sha256": domain_digest(domains), "domain_sizes": list(map(len, domains)),
              "domain_npz_sha256": sha(output / "domains.npz"),
              "support_cut_count": len(cuts), "support_cut_file_sha256": sha(output / "support-cuts.json"),
              "domain_build": build, "build_seconds": time.monotonic()-begin,
              "parameters": {"xy_width": xy_width, "angle_width": angle_width, "max_poses": max_poses,
                             "target": TARGET, "acceptance_guard": ACCEPTANCE_GUARD,
                             "target_seconds": seconds, "control_seconds": control_seconds, "seed": seed},
              "scope": "Finite independent per-piece union across fitted physical modes/branches in one common rigid frame; offline, no API/global promotion",
              "control_scope": "Actual incumbent poses explicitly included and hinted through index0 in the complete mixed domain; correctness control, not blind discovery",
              "old_pool_audit": {"path": "scripts/experiment_phase_pool.py", "sha256": sha(ROOT / "scripts/experiment_phase_pool.py"),
                                 "finding": "phase_domains rotates a single base about piece0; it does not call Manifold.section or vary physical deformation coordinates"},
              "command": [sys.executable, *sys.argv]}
    write_json(output / "report.json", report)
    print(json.dumps({"event": "prepared", "domain_sizes": report["domain_sizes"],
                      "domain_sha256": report["domain_sha256"], "support_cuts": len(cuts),
                      "build_seconds": report["build_seconds"]}), flush=True)
    with use_supports(cuts):
        control = solve_phase(domains, incumbent, inputs[0]["reference_score"]+1e-11,
            output / "control", seconds=control_seconds, seed=seed, control=True, local_only=True)
        report["control"] = control
        report["control_source_membership"] = assignment_sources(control, provenance)
        write_json(output / "report.json", report)
        print(json.dumps({"event": "control", "status": control["status"], "score": control.get("score"),
                          "seconds": control["seconds"]}), flush=True)
        if not control.get("valid"):
            report["status"] = "CONTROL_FAILED"
        elif control["score"] < LEADER:
            report["status"] = "LOCAL_FIRST_PLACE_CANDIDATE_IN_CONTROL"
            shutil.copy2(output / "control/solution.json", output / "candidate.json")
        else:
            target = solve_phase(domains, incumbent, TARGET, output / "target", seconds=seconds,
                                 seed=seed+1, local_only=True)
            report["target"] = target
            report["target_source_membership"] = assignment_sources(target, provenance)
            report["status"] = target["status"]
            if target.get("valid"):
                shutil.copy2(output / "target/solution.json", output / "candidate.json")
            print(json.dumps({"event": "target", "status": target["status"], "score": target.get("score"),
                              "seconds": target["seconds"]}), flush=True)
    report["seconds"] = time.monotonic()-begin
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", type=Path, default=Path("tests/fixtures/recovery/mirror-continuous.json"))
    parser.add_argument("--input", action="append", type=Path, required=True, help="First input must be current incumbent")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=180.)
    parser.add_argument("--control-seconds", type=float, default=30.)
    parser.add_argument("--seed", type=int, default=3111901)
    parser.add_argument("--max-poses", type=int, default=500)
    args = parser.parse_args()
    result = run(args.origin, args.input, args.output, seconds=args.seconds,
                 control_seconds=args.control_seconds, seed=args.seed, max_poses=args.max_poses)
    print(json.dumps({"status": result["status"], "seconds": result["seconds"]}, indent=2))


if __name__ == "__main__":
    main()
