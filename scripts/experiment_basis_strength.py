"""Matched physical targets for bounded BKZ-20/30/40 packing comparisons.

Basis construction runs in a killable subprocess. Ambient integer targets,
continuous geometry and grid phase are frozen before comparing reductions.
"""
from __future__ import annotations

import os
for _name in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fpylll import GSO, IntegerMatrix

from benchmark_recovery import fit_frame, scored_once, transform, write_json
from breakout_geometry import evaluate_batch
from experiment_batch_lattice import NativeBabai
from experiment_mirror_family import FLATS, Manifold
from experiment_tied_batch import Frame
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.geometry import canonicalize_theta
from semicircle_packing.polish import AreaConstraints, checked, load, polygon_areas, save

ROOT = Path(__file__).resolve().parents[1]
TARGET = 2.9485269238806113


def digest_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def set_grid_phase(frame, phase, radius):
    """Change the rounding anchor, keeping the model's enclosing center at zero."""
    old = frame.anchor_units.copy()
    anchor = np.rint(frame.p * 1e6).astype(np.int64)
    anchor[:, :2] = np.rint((frame.p[:, :2] + np.asarray(phase)) * 1e6).astype(np.int64)
    for (i, j), difference in zip(frame.pairs, frame.differences):
        anchor[i, 2] = anchor[j, 2] + difference
    _, jacobian = frame.c.evaluate(np.r_[frame.p.ravel(), radius])
    frame.b0 -= jacobian[frame.active, :45] @ (anchor - old).ravel()
    frame.anchor_units = anchor


def ambient_targets(frame, rng, count, settings):
    concentration = np.full(len(frame.weights), settings.concentration, dtype=float)
    concentration[np.flatnonzero(frame.active) >= 105] *= settings.boundary_bias
    desired = frame.budget * rng.dirichlet(concentration, size=count) / frame.weights[None, :] / 1e-6
    contact = (frame.weights[None, :] * (desired + frame.b0)) @ frame.projection.T
    wanted = rng.normal(0, settings.movement, (count, frame.dimension))
    return np.rint(np.c_[wanted * frame.identity, contact * frame.scale]).astype(np.int64)


def pose_key(units):
    values = np.asarray(units, dtype=np.int64).reshape(15, 3).copy()
    values[:, :2] -= values[0, :2].copy()
    return hashlib.sha256(values.astype("<i8").tobytes()).digest()


def build_worker(input_path, output_path, block):
    with np.load(input_path) as payload:
        p, phase = payload["physical"], payload["phase"]
        radius, target = float(payload["radius"]), float(payload["target"])
    if np.linalg.norm(compute_mec_fast(p)[:2]) > 1e-8:
        raise ValueError("Physical basis frame must be centered before applying its anchor phase")
    settings = SimpleNamespace(target=target, regularization=1e-5, block_size=block)
    started = time.monotonic()
    frame = Frame(p, radius, FLATS, settings)
    set_grid_phase(frame, phase, radius)
    reduced = np.asarray([list(row) for row in frame.M.B], dtype=np.int64)
    np.savez_compressed(output_path, reduced=reduced, moves=frame.moves, mu=frame.mu,
                        transform=frame.gso_transform, anchor=frame.anchor_units,
                        active=frame.active, weights=frame.weights, b0=frame.b0,
                        dimension=frame.dimension, differences=frame.differences,
                        build_seconds=time.monotonic() - started)


def prepare_frames(input_path, output, *, frames, proposals, seed):
    raw = load(input_path)
    check = checked(raw)
    if not check.valid:
        raise ValueError("Basis comparison needs a valid incumbent")
    p = scored_once(raw)
    model = Manifold(load(ROOT / "tests/fixtures/recovery/mirror-continuous.json"))
    branch = [round(float(p[i, 2] - p[j, 2]), 6) for i, j in FLATS]
    _, fit = fit_frame(model.p, p, model=model, branch=branch)
    if not fit["success"]:
        raise ValueError("Could not fit the current continuous family")
    rng = np.random.default_rng(seed)
    prepared = []
    for index in range(frames):
        if index == 0:
            delta, angle_delta = np.zeros(2), 0.
        elif index == 1:
            delta, angle_delta = np.array([.0002, 0.]), .00017
        elif index == 2:
            delta, angle_delta = np.array([0., -.0002]), -.00023
        else:
            delta, angle_delta = rng.normal(0., .0004, 2), float(rng.normal(0., .0004))
        physical, section = model.section(branch, np.asarray(fit["alpha"]) + delta)
        physical = transform(physical, fit["angle"] + angle_delta, [0., 0.])
        circle = np.asarray(compute_mec_fast(physical))
        physical[:, :2] -= circle[:2]
        if index == 0:
            offset = circle[:2] + np.asarray(fit["translation"])
            phase = offset - np.round(offset, 6)
        else:
            phase = rng.uniform(-.5e-6, .5e-6, 2)
        settings = SimpleNamespace(target=TARGET, regularization=1e-5, block_size=0,
                                   concentration=(10., 1., .3)[index % 3],
                                   boundary_bias=1., movement=(10., 50., 150.)[index % 3])
        if max(section["violation"], section["residual"]) > 1e-9:
            raise ValueError("Matched frame failed its continuous construction check")
        frame = Frame(physical, circle[2], FLATS, settings)
        set_grid_phase(frame, phase, circle[2])
        ambient = ambient_targets(frame, np.random.default_rng(seed + 1000 + index), proposals, settings)
        path = output / f"frame-{index:03d}.npz"
        np.savez_compressed(path, physical=physical, radius=circle[2], phase=phase, ambient=ambient,
                            target=TARGET, expected_anchor=frame.anchor_units, expected_b0=frame.b0,
                            expected_weights=frame.weights, expected_active=frame.active)
        prepared.append(dict(index=index, file=path.name, physical_sha256=digest_array(physical),
                             ambient_sha256=digest_array(ambient), phase=phase.tolist(),
                             radius=float(circle[2]), section=section,
                             concentration=settings.concentration, movement=settings.movement,
                             enclosing_center_after_shift=compute_mec_fast(physical)[:2]))
    return prepared, fit, check.score, AreaConstraints(p).scale


def evaluate_basis(payload_path, basis_path, folder, fixed_scale, validation_cap, *, audit_count=128):
    started = time.monotonic()
    with np.load(payload_path) as payload:
        ambient = payload["ambient"]
        for key, expected in (("anchor", "expected_anchor"), ("b0", "expected_b0"),
                              ("weights", "expected_weights"), ("active", "expected_active")):
            with np.load(basis_path) as basis:
                if not np.array_equal(basis[key], payload[expected]):
                    raise AssertionError(f"Basis arm changed the physical target mapping: {key}")
    with np.load(basis_path) as archive:
        basis = {key: archive[key].copy() for key in archive.files}
    decoder = NativeBabai()
    before = time.monotonic()
    actual, units, coefficients, ambiguous = decoder.decode(
        ambient @ basis["transform"], basis["mu"], basis["moves"], basis["anchor"])
    matrix = GSO.Mat(IntegerMatrix.from_matrix(basis["reduced"].tolist()), update=True)

    def raw_for(coefficient):
        delta = (coefficient @ basis["moves"]).reshape(15, 3)
        delta[:, :2] -= delta[0, :2].copy()
        return (basis["anchor"] + delta) * 1e-6

    for index in np.flatnonzero(ambiguous):
        coefficients[index] = matrix.babai(ambient[index].tolist())
        raw = raw_for(coefficients[index])
        actual[index] = scored_once(raw)
        units[index] = np.rint(actual[index].ravel() * 1e6).astype(np.int64)
    audit_count = min(audit_count, len(ambient))
    expected = np.asarray([matrix.babai(row.tolist()) for row in ambient[:audit_count]], dtype=np.int64)
    if not np.array_equal(expected, coefficients[:audit_count]):
        raise AssertionError("Native decoding disagrees with fpylll on this basis")
    decode_seconds = time.monotonic() - before
    permitted = np.ones(len(actual), dtype=bool)
    for (i, j), difference in zip(FLATS, basis["differences"]):
        permitted &= units[:, 3 * i + 2] - units[:, 3 * j + 2] == difference
    indices = np.flatnonzero(permitted)
    keys, unique_indices = set(), []
    for index in indices:
        key = pose_key(units[index])
        if key not in keys:
            keys.add(key)
            unique_indices.append(int(index))
    selected = np.asarray(unique_indices, dtype=int)
    report = dict(proposals=len(actual), branch_rejected=int((~permitted).sum()),
                  distinct_poses=len(selected), ambiguous_patches=int(ambiguous.sum()),
                  fpylll_audited=audit_count, decode_seconds=decode_seconds,
                  build_seconds=float(basis["build_seconds"]), valid_records=[])
    if not len(selected):
        report.update(seconds=time.monotonic() - started, status="NO_PERMITTED_POSES")
        return report, keys
    before = time.monotonic()
    areas, circles, _ = evaluate_batch(actual[selected], centers=[0., 0.])
    report["geometry_seconds"] = time.monotonic() - before
    deficits = np.maximum(((areas - 1e-6) / fixed_scale).max(axis=1), circles[:, 2] - TARGET)
    report["native_minimum_violation"] = float(deficits.min())
    report["native_violation_quantiles"] = np.quantile(deficits, [0., .01, .1, .5]).tolist()
    report["native_under_25_billionths"] = int((deficits <= 25e-9).sum())
    report["native_under_100_billionths"] = int((deficits <= 100e-9).sum())
    # Fixed pair normalization across every arm and frame; diagnostics cannot
    # improve merely because a new Jacobian changed the units of a violation.
    diagnostic = np.argsort(deficits)[:8]
    exact_merits = []
    for local in diagnostic:
        exact = polygon_areas(actual[selected[local]])
        exact_merits.append(float(max(((exact - 1e-6) / fixed_scale).max(), circles[local, 2] - TARGET)))
    report["geos_top8_minimum_violation"] = min(exact_merits)
    report["geos_diagnostics"] = len(exact_merits)
    eligible = np.flatnonzero((areas.max(axis=1) <= 1e-6 + 1e-11)
                              & (circles[:, 2] <= validation_cap + 2e-12))
    report["permissive_eligible"] = len(eligible)
    for local in eligible:
        raw = raw_for(coefficients[selected[local]])
        check = checked(raw)
        if check.valid:
            path = folder / f"valid-{len(report['valid_records']):04d}.json"
            save(path, raw)
            report["valid_records"].append(dict(score=float(check.score), file=path.name,
                                                raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    nearest = int(selected[int(np.argmin(deficits))])
    save(folder / "nearest.json", raw_for(coefficients[nearest]))
    np.savez_compressed(folder / "nearest-state.npz", raw=raw_for(coefficients[nearest]),
                        physical=actual[nearest], coefficients=coefficients[nearest],
                        ambient=ambient[nearest], moves=basis["moves"], anchor=basis["anchor"])
    report.update(seconds=time.monotonic() - started, status="COMPLETE")
    return report, keys


def run(input_path, output, *, frames=3, proposals=4096, build_timeout=8,
        seconds_per_arm=60, seed=3111901, blocks=(20, 30, 40)):
    output = Path(output)
    if output.exists():
        raise ValueError("Output directory must be new")
    output.mkdir(parents=True)
    snapshot = output / "source-snapshot"
    snapshot.mkdir()
    source_hashes = {}
    for name in ("experiment_basis_strength.py", "experiment_tied_batch.py", "experiment_batch_lattice.py",
                 "experiment_mirror_family.py", "benchmark_recovery.py", "_batch_lattice.cpp"):
        source = ROOT / "scripts" / name
        shutil.copy2(source, snapshot / name)
        source_hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    shutil.copy2(input_path, output / "incumbent.json")
    started = time.monotonic()
    NativeBabai()
    prepared, fit, incumbent, fixed_scale = prepare_frames(input_path, output, frames=frames,
                                                          proposals=proposals, seed=seed)
    preparation_seconds = time.monotonic() - started
    report = dict(scope="Matched ambient-target comparison of BKZ reduction strength; offline validation only",
                  input_sha256=hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
                  source_hashes=source_hashes, preparation_seconds=preparation_seconds, fit=fit,
                  frames=prepared, proposals_per_frame=proposals, target=TARGET,
                  incumbent=incumbent, validation_cap=incumbent + 100e-9,
                  fixed_pair_normalization=fixed_scale.tolist(), build_timeout=build_timeout,
                  seconds_per_arm=seconds_per_arm, seed=seed, arms=[])
    for block in blocks:
        arm_start = time.monotonic()
        arm = dict(block=block, frames=[], distinct_poses=0)
        all_keys = set()
        folder = output / f"bkz-{block}"
        folder.mkdir()
        for frame in prepared:
            remaining = seconds_per_arm - (time.monotonic() - arm_start)
            if remaining <= 0:
                break
            local = folder / f"frame-{frame['index']:03d}"
            local.mkdir()
            payload = output / frame["file"]
            basis = local / "basis.npz"
            command = [sys.executable, str(Path(__file__).resolve()), "--build-worker",
                       "--payload", str(payload.resolve()), "--basis-output", str(basis.resolve()),
                       "--block", str(block)]
            before = time.monotonic()
            try:
                result = subprocess.run(command, capture_output=True, text=True,
                                        timeout=min(build_timeout, remaining))
            except subprocess.TimeoutExpired:
                record = dict(index=frame["index"], status="BUILD_TIMEOUT",
                              process_seconds=time.monotonic() - before)
            else:
                if result.returncode:
                    record = dict(index=frame["index"], status="BUILD_ERROR", stderr=result.stderr[-3000:],
                                  process_seconds=time.monotonic() - before)
                else:
                    build_process_seconds = time.monotonic() - before
                    record, keys = evaluate_basis(payload, basis, local, fixed_scale, incumbent + 100e-9)
                    all_keys.update(keys)
                    record.update(index=frame["index"], build_process_seconds=build_process_seconds,
                                  ambient_sha256=frame["ambient_sha256"])
                    write_json(local / "report.json", record)
                    for valid in record["valid_records"]:
                        if valid["score"] < report.get("best_score", incumbent) - 2e-12:
                            report["best_score"] = valid["score"]
                            shutil.copy2(local / valid["file"], output / "candidate.json")
                            print(json.dumps({"event": "candidate", "score": valid["score"],
                                              "file": str(output / "candidate.json")}), flush=True)
            arm["frames"].append(record)
            arm.update(seconds=time.monotonic() - arm_start, distinct_poses=len(all_keys))
            write_json(folder / "report.json", arm)
            print(json.dumps({"event": "frame", "block": block, "frame": frame["index"],
                              "status": record["status"], "build_seconds": record.get("build_seconds"),
                              "geos_merit": record.get("geos_top8_minimum_violation"),
                              "distinct": record.get("distinct_poses"),
                              "valid": len(record.get("valid_records", []))}), flush=True)
        arm.update(seconds=time.monotonic() - arm_start, distinct_poses=len(all_keys))
        report["arms"].append(arm)
        report["seconds"] = time.monotonic() - started
        write_json(output / "report.json", report)
    completed_sets = [{r["index"] for r in arm["frames"] if r["status"] == "COMPLETE"} for arm in report["arms"]]
    report["matched_completed_frames"] = sorted(set.intersection(*completed_sets)) if completed_sets else []
    report.update(seconds=time.monotonic() - started, status="COMPLETE")
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "solution.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--proposals", type=int, default=4096)
    parser.add_argument("--build-timeout", type=float, default=8)
    parser.add_argument("--seconds-per-arm", type=float, default=60)
    parser.add_argument("--seed", type=int, default=3111901)
    parser.add_argument("--build-worker", action="store_true")
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--basis-output", type=Path)
    parser.add_argument("--block", type=int)
    args = parser.parse_args()
    if args.build_worker:
        build_worker(args.payload, args.basis_output, args.block)
        return
    if args.output is None:
        parser.error("--output is required")
    result = run(args.input, args.output, frames=args.frames, proposals=args.proposals,
                 build_timeout=args.build_timeout, seconds_per_arm=args.seconds_per_arm, seed=args.seed)
    print(json.dumps({"status": result["status"], "seconds": result["seconds"],
                      "matched_frames": result["matched_completed_frames"],
                      "best": result.get("best_score")}), flush=True)


if __name__ == "__main__":
    main()
