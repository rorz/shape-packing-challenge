"""Offline coverage and solver recovery benchmark for the mirror packing family.

The target supplies frame alignment and its flat-angle branch. This deliberately
optimistic test measures representation, not blind discovery. Its coordinates
are never appended to a domain or passed to the solver as hints. Box widths are
calibrated on the designated calibration cases only, then frozen for held-outs.
"""
from __future__ import annotations

import os

for _name in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from semicircle_packing.geometry import canonicalize_theta
from semicircle_packing.polish import checked, load

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "tests/fixtures/recovery/manifest.json"
MICRO = 1e-6
SCORE_GUARD = 1e-11


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def placements_json(p):
    return [dict(zip(("x", "y", "theta"), map(float, row))) for row in p]


def scored_once(raw):
    result = np.round(np.asarray(raw, dtype=float), 6)
    result[:, 2] = [canonicalize_theta(theta) for theta in result[:, 2]]
    return result


def wrap(theta):
    return np.arctan2(np.sin(theta), np.cos(theta))


def transform(p, angle, translation):
    co, sn = np.cos(angle), np.sin(angle)
    result = p.copy()
    result[:, :2] = p[:, :2] @ np.array([[co, sn], [-sn, co]]) + translation
    result[:, 2] = wrap(p[:, 2] + angle)
    return result


def rigid_guess(p, target):
    x = p[:, :2] - p[:, :2].mean(axis=0)
    y = target[:, :2] - target[:, :2].mean(axis=0)
    correlation = np.sum((y[:, 0] + 1j * y[:, 1]) * (x[:, 0] - 1j * x[:, 1]))
    correlation += np.sum(np.exp(1j * (target[:, 2] - p[:, 2])))
    angle = float(np.angle(correlation))
    translation = target[:, :2].mean(axis=0) - transform(p, angle, [0, 0])[:, :2].mean(axis=0)
    return angle, translation


def fit_frame(seed, target, *, model=None, branch=None):
    """Fit rigid parameters, optionally both nonlinear physical modes.

    A local least-squares alignment is not a minimum-width or global coverage
    certificate. Finite differences use fixed physical steps to stay above the
    contact-section solver's numerical tolerance.
    """
    started = time.perf_counter()
    flexible = model is not None
    if flexible:
        seed, _ = model.section(branch, [0, 0])
    angle0, translation0 = rigid_guess(seed, target)

    def evaluate(parameters):
        if flexible:
            p, section = model.section(branch, parameters[:2] * MICRO)
            rigid = parameters[2:]
        else:
            p, section, rigid = seed, None, parameters
        posed = transform(p, angle0 + rigid[0] * MICRO, translation0 + rigid[1:] * MICRO)
        return posed, section

    def residual(parameters):
        p, _ = evaluate(parameters)
        delta = p - target
        delta[:, 2] = wrap(delta[:, 2])
        return delta.ravel() / MICRO

    def jacobian(parameters):
        step = 0.05  # 5e-8 in physical coordinates, not scipy's near-zero step.
        columns = []
        for index in range(len(parameters)):
            offset = np.zeros_like(parameters)
            offset[index] = step
            columns.append((residual(parameters + offset) - residual(parameters - offset)) / (2 * step))
        return np.column_stack(columns)

    size = 5 if flexible else 3
    lower = np.full(size, -np.inf)
    upper = np.full(size, np.inf)
    if flexible:
        lower[:2], upper[:2] = -5000, 5000
    fit = least_squares(residual, np.zeros(size), jac=jacobian, bounds=(lower, upper),
                        max_nfev=80, ftol=1e-10, xtol=1e-10, gtol=1e-7)
    posed, section = evaluate(fit.x)
    delta = residual(fit.x).reshape(-1, 3)
    rigid = fit.x[2:] if flexible else fit.x
    metadata = {
        "method": "local nonlinear least squares; target-assisted alignment",
        "success": bool(fit.success), "message": str(fit.message), "evaluations": fit.nfev,
        "rms_grid_units": float(np.sqrt(np.mean(delta ** 2))),
        "max_xy_grid_units": float(np.max(np.abs(delta[:, :2]))),
        "max_angle_microradians": float(np.max(np.abs(delta[:, 2]))),
        "angle": float(angle0 + rigid[0] * MICRO),
        "translation": (translation0 + rigid[1:] * MICRO).tolist(),
        "alpha": (fit.x[:2] * MICRO).tolist() if flexible else None,
        "section": section, "seconds": time.perf_counter() - started,
    }
    return posed, metadata


def required_widths(base, target):
    """Integer offsets needed in this frame, including one-pass wrap behavior."""
    from experiment_lazy_phases import make_domains

    delta = np.rint(target * 1e6).astype(np.int64) - np.rint(base * 1e6).astype(np.int64)
    xy = int(np.max(np.abs(delta[:, :2])))
    # Away from the seam subtraction is exact; near it there is no integer
    # representation of a full 2*pi turn. Ask the production generator instead.
    estimate = int(np.ceil(np.max(np.abs(wrap(target[:, 2] - base[:, 2]))) * 1e6)) + 2
    if estimate > 20000:
        raise ValueError("Angle fit is too far away for the bounded recovery diagnostic")
    angles = make_domains(base, 0, estimate)
    widths = []
    for i, choices in enumerate(angles):
        values = set(np.rint(choices[:, 2] * 1e6).astype(np.int64))
        wanted = round(target[i, 2] * 1e6)
        if wanted not in values:
            raise ValueError(f"Unreachable scored angle on piece {i}; seam aliases need another generator")
        # At most 2*estimate+1 inexpensive scalar checks, independent of XYZ.
        width = next(w for w in range(estimate + 1) if any(
            round(canonicalize_theta(base[i, 2] + sign * w * MICRO) * 1e6) == wanted
            for sign in (-1, 1)))
        widths.append(width)
    return {"xy": xy, "angle": max(widths), "xy_offsets": delta[:, :2].tolist(),
            "angle_offsets_required": widths}


def membership(domains, target):
    assignment, missing = [], []
    units = np.rint(target * 1e6).astype(np.int64)
    for i, domain in enumerate(domains):
        rows = np.rint(domain * 1e6).astype(np.int64)
        matches = np.flatnonzero(np.all(rows == units[i], axis=1))
        assignment.append(int(matches[0]) if len(matches) else None)
        if not len(matches):
            missing.append(i)
    return {"covered": not missing, "covered_pieces": len(domains) - len(missing),
            "missing_pieces": missing, "witness_assignment": assignment}


def calibrated_profile(cases, floor):
    training = [case for case in cases if case["split"] == "calibration"]
    if not training:
        raise ValueError("At least one calibration case is required")
    return {key: max(floor[key], *(case["family"]["required_widths"][key] for case in training))
            for key in ("xy", "angle")}


def verify_fixture(directory, record, path_key="path"):
    path = directory / record[path_key]
    if digest(path) != record["sha256"]:
        raise ValueError(f"Fixture hash mismatch: {path}")
    p = load(path)
    if p.shape != (15, 3) or not np.all(np.isfinite(p)):
        raise ValueError(f"Expected 15 finite poses: {path}")
    return p


def run_benchmark(manifest_path, output, *, solve=True, seconds=15, xy_width=1,
                  angle_width=12, max_poses=2000):
    from experiment_lazy_phases import make_domains
    from experiment_mirror_family import FLATS, Manifold
    from lazy_phase_solver import solve_phase

    started = time.perf_counter()
    nominal_size = (2 * xy_width + 1) ** 2 * (2 * angle_width + 1)
    if min(xy_width, angle_width) < 0 or max_poses < 1 or not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Widths must be nonnegative, and pose/time limits positive and finite")
    if nominal_size > max_poses:
        raise ValueError(f"Current profile needs {nominal_size} poses per piece, exceeding --max-poses {max_poses}")
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output is not empty; choose a new directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported fixture schema")
    fixture_root = manifest_path.parent
    ids = [case["id"] for case in manifest["cases"]]
    if len(ids) != len(set(ids)) or any(Path(name).name != name for name in ids):
        raise ValueError("Case ids must be unique, single path components")
    seed = verify_fixture(fixture_root, manifest["seed"])
    model = Manifold(seed)
    current = {"xy": xy_width, "angle": angle_width}
    report = {
        "schema_version": 1,
        "scope": "Target-assisted local frame coverage and unhinted finite-domain recovery; not blind search or global exclusion",
        "manifest_sha256": digest(manifest_path),
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "shapely", "ortools", "fpylll")}},
        "parameters": {"current_profile": current, "solve": solve, "seconds_per_case": seconds,
                       "max_poses_per_piece": max_poses, "score_guard": SCORE_GUARD},
        "source_hashes": {name: digest(ROOT / "scripts" / name) for name in (
            "benchmark_recovery.py", "experiment_lazy_phases.py", "experiment_mirror_family.py",
            "lazy_phase_solver.py", "lazy_pair_oracle.py", "separable_support.py",
            "experiment_support_bundle.py", "_conflict_area_band.cpp")},
        "cases": [],
    }
    frames = {}
    # Complete all fits before calibrating; only the calibration split supplies widths.
    for case in manifest["cases"]:
        raw = verify_fixture(fixture_root, case, "target")
        validation = checked(raw)
        if not validation.valid or abs(validation.score - case["expected_score"]) > SCORE_GUARD:
            raise ValueError(f"Invalid fixture or score drift: {case['id']}")
        target = scored_once(raw)
        branch = [round(float(target[i, 2] - target[j, 2]), 6) for i, j in FLATS]
        entry = {"id": case["id"], "split": case["split"], "target_sha256": case["sha256"],
                 "reference_score": validation.score, "branch": branch}
        zero_mode_section, _ = model.section(branch, [0, 0])
        for kind in ("rigid", "family", "saved_origin"):
            if kind == "saved_origin":
                origin = verify_fixture(fixture_root, case["saved_origin"])
                physical, fit = fit_frame(origin, target)
                fit["origin_kind"] = case["saved_origin"]["kind"]
            else:
                physical, fit = fit_frame(zero_mode_section, target, model=model if kind == "family" else None, branch=branch)
            if not fit["success"]:
                raise ValueError(f"Frame fit did not converge: {case['id']}/{kind}")
            if kind == "family" and max(fit["section"]["violation"], fit["section"]["residual"]) > 1e-9:
                raise ValueError(f"Continuous family section failed its constraints: {case['id']}")
            base = scored_once(physical)
            widths = required_widths(base, target)
            entry[kind] = {"fit": fit, "required_widths": widths,
                           "current_profile": membership(make_domains(base, xy_width, angle_width), target)}
            if kind == "family":
                frames[case["id"]] = (base, target)
            write_json(output / case["id"] / f"{kind}-frame.json", placements_json(physical))
        report["cases"].append(entry)
        write_json(output / "report.json", report)
        print(json.dumps({"case": case["id"], "family_widths": entry["family"]["required_widths"],
                          "current_covered": entry["family"]["current_profile"]["covered"]}), flush=True)

    profile = calibrated_profile(report["cases"], current)
    report["calibrated_profile"] = profile
    # Freeze calibration before solver evaluation; held-out widths never enter it.
    write_json(output / "calibration.json", {"profile": profile, "calibration_ids": [
        c["id"] for c in report["cases"] if c["split"] == "calibration"],
        "rule": "maximum required integer width on calibration cases, floored by current profile"})
    for case in report["cases"]:
        base, target = frames[case["id"]]
        nominal_size = (2 * profile["xy"] + 1) ** 2 * (2 * profile["angle"] + 1)
        if nominal_size > max_poses:
            case["recovery"] = {"status": "DOMAIN_LIMIT", "nominal_poses_per_piece": nominal_size,
                                "limit": max_poses, "scope": "No domain was truncated or solved"}
            continue
        domains = make_domains(base, profile["xy"], profile["angle"])
        coverage = membership(domains, target)
        case["family"]["calibrated_profile"] = coverage
        domain_hash = hashlib.sha256(b"".join(np.rint(d * 1e6).astype("<i8").tobytes() for d in domains)).hexdigest()
        case["domain_sha256"] = domain_hash
        case["domain_sizes"] = list(map(len, domains))
        if not coverage["covered"]:
            case["recovery"] = {"status": "TARGET_NOT_IN_DOMAIN",
                                "scope": "This fitted frame omits the witness; not a family exclusion"}
        elif not solve:
            case["recovery"] = {"status": "NOT_RUN"}
        else:
            # Deliberately do not pass coverage['witness_assignment'] or target.
            result = solve_phase(domains, base, case["reference_score"] + SCORE_GUARD,
                                 output / case["id"] / "solver", seconds=seconds,
                                 seed=3110901, local_only=True)
            if result.get("valid"):
                recovered = np.asarray(result["scored_placements"])
                result["same_target_centers_mod_translation"] = same_centers(recovered, target)
                result["same_target_geometry"] = bool(np.array_equal(
                    np.rint(recovered * 1e6).astype(np.int64), np.rint(target * 1e6).astype(np.int64)))
            case["recovery"] = result
        write_json(output / "report.json", report)
        print(json.dumps({"case": case["id"], "coverage": coverage["covered"],
                          "status": case["recovery"]["status"]}), flush=True)

    report["summary"] = {
        "cases": len(report["cases"]),
        "current_covered": sum(c["family"]["current_profile"]["covered"] for c in report["cases"]),
        "calibrated_covered": sum(c["family"].get("calibrated_profile", {}).get("covered", False) for c in report["cases"]),
        "solver_recovered": sum(c["recovery"].get("valid", False) for c in report["cases"]),
        "seconds": time.perf_counter() - started,
    }
    write_json(output / "report.json", report)
    return report


def same_centers(a, b):
    a = np.rint(a[:, :2] * 1e6).astype(np.int64)
    b = np.rint(b[:, :2] * 1e6).astype(np.int64)
    return bool(np.array_equal(a - a[0], b - b[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True, help="new directory for immutable run evidence")
    parser.add_argument("--coverage-only", action="store_true", help="fit and audit domains, without invoking CP-SAT")
    parser.add_argument("--seconds-per-case", type=float, default=15)
    parser.add_argument("--xy-width", type=int, default=1)
    parser.add_argument("--angle-width", type=int, default=12)
    parser.add_argument("--max-poses", type=int, default=2000)
    parser.add_argument("--require-recovery", action="store_true", help="exit 1 unless every case is covered and solver-recovered")
    args = parser.parse_args()
    if min(args.xy_width, args.angle_width) < 0 or args.max_poses < 1 or args.seconds_per_case <= 0:
        parser.error("Widths must be nonnegative, and pose/time limits positive")
    if args.coverage_only and args.require_recovery:
        parser.error("--require-recovery needs the solver")
    report = run_benchmark(args.manifest, args.output, solve=not args.coverage_only,
                           seconds=args.seconds_per_case, xy_width=args.xy_width,
                           angle_width=args.angle_width, max_poses=args.max_poses)
    print(json.dumps(report["summary"], indent=2))
    if args.require_recovery and report["summary"]["solver_recovered"] != report["summary"]["cases"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
