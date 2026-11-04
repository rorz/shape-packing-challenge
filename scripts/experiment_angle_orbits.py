"""Offline fixed-center angular repair across all eight integer D4 transforms.

Quarter turns preserve the center lattice but not the microradian angle lattice.
Every transformed angle is rounded anew; a D4 center hash is not a proof that
these finite angular models are equivalent.
"""
from __future__ import annotations

import os
for _name in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
import operator
import shutil
import time
from pathlib import Path

import numpy as np

from lazy_phase_solver import ACCEPTANCE_GUARD, solve_phase
from semicircle_packing.geometry import canonicalize_theta
from semicircle_packing.polish import load
from translation_union import domain_digest

MICRO = 1e-6
ROOT = Path(__file__).resolve().parents[1]


def d4_matrices():
    """Named column-vector transforms; integer centers use rows @ matrix.T."""
    return [
        ("identity", np.array([[1, 0], [0, 1]], dtype=np.int64)),
        ("rotate_ccw_90", np.array([[0, -1], [1, 0]], dtype=np.int64)),
        ("rotate_180", np.array([[-1, 0], [0, -1]], dtype=np.int64)),
        ("rotate_cw_90", np.array([[0, 1], [-1, 0]], dtype=np.int64)),
        ("reflect_x_axis", np.array([[1, 0], [0, -1]], dtype=np.int64)),
        ("reflect_y_axis", np.array([[-1, 0], [0, 1]], dtype=np.int64)),
        ("reflect_y_equals_x", np.array([[0, 1], [1, 0]], dtype=np.int64)),
        ("reflect_y_equals_minus_x", np.array([[0, -1], [-1, 0]], dtype=np.int64)),
    ]


def scored_once(raw):
    p = np.round(np.asarray(raw, dtype=float), 6)
    if p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.all(np.isfinite(p)):
        raise ValueError("Expected finite (x, y, theta) rows")
    p[:, 2] = [canonicalize_theta(theta) for theta in p[:, 2]]
    return p


def transform_scored(scored, matrix, translation_units=(0, 0)):
    """Transform scored geometry without normalizing its angles a second time.

    Returns exact integer-grid centers and continuous transformed angles. The
    angle-domain generator is responsible for rounding those angles once.
    """
    p = np.asarray(scored, dtype=float)
    matrix = np.asarray(matrix)
    translation = np.asarray(translation_units)
    if p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.all(np.isfinite(p)):
        raise ValueError("Expected finite scored poses")
    if matrix.shape != (2, 2) or not np.array_equal(matrix, matrix.astype(np.int64)):
        raise ValueError("Expected an integer orthogonal 2x2 matrix")
    matrix = matrix.astype(np.int64)
    if not np.array_equal(matrix.T @ matrix, np.eye(2, dtype=np.int64)):
        raise ValueError("Expected an integer orthogonal 2x2 matrix")
    if translation.shape != (2,) or not np.array_equal(translation, translation.astype(np.int64)):
        raise ValueError("Translation must contain two integer micro-units")
    if np.max(np.abs(p[:, :2] * 1e6)) + np.max(np.abs(translation)) >= 2 ** 52:
        raise ValueError("Centers exceed the exact integer-micro range")
    units = np.rint(p[:, :2] * 1e6).astype(np.int64)
    if not np.allclose(units / 1e6, p[:, :2], rtol=0, atol=1e-12):
        raise ValueError("Input centers must already be on the six-decimal grid")
    centers = units @ matrix.T + translation.astype(np.int64)
    direction = np.column_stack([np.cos(p[:, 2]), np.sin(p[:, 2])]) @ matrix.T
    return np.column_stack([centers / 1e6, np.arctan2(direction[:, 1], direction[:, 0])])


def make_angle_domains(transformed, angle_width=3):
    """Fixed XY with raw rounded-angle offsets, plus reachable seam aliases.

    The ordinary neighborhood is round(ideal angle, 6) plus integer microradian
    offsets, canonicalized once. Scored ±3.141593 are additionally included when
    their circular distance from the ideal angle is within (width+0.5) micro;
    they require raw ±3pi submission aliases. This is a documented superset.
    """
    try:
        width = operator.index(angle_width)
    except TypeError as error:
        raise ValueError("angle_width must be an integer from 0 to 128") from error
    if not 0 <= width <= 128:
        raise ValueError("angle_width must be an integer from 0 to 128")
    p = np.asarray(transformed, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.all(np.isfinite(p)):
        raise ValueError("Expected finite transformed poses")
    domains = []
    for x, y, ideal in p:
        raw_base = float(np.round(ideal, 6))
        angles = [canonicalize_theta(raw_base)]
        for offset in range(-width, width + 1):
            angles.append(canonicalize_theta(float(np.round(raw_base + offset * MICRO, 6))))
        for alias in (-3.141593, 3.141593):
            delta = np.arctan2(np.sin(alias - ideal), np.cos(alias - ideal))
            if abs(delta) <= (width + 0.5) * MICRO + 2e-15:
                angles.append(alias)
        # Dict insertion order keeps the ordinary rounded transform at index0.
        unique = list(dict.fromkeys(round(theta * 1e6) for theta in angles))
        domain = np.column_stack([np.full(len(unique), x), np.full(len(unique), y),
                                  np.asarray(unique, dtype=float) / 1e6])
        domains.append(domain)
    return domains


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_orbits(inputs, target, output, *, seconds=60, per_model=2, angle_widths=(3,), seed=3110927):
    if not np.isfinite(target) or target <= 0:
        raise ValueError("target must be positive and finite")
    if not np.isfinite(seconds) or not 0 < seconds <= 60 or not np.isfinite(per_model) or per_model <= 0:
        raise ValueError("Use a total budget in (0,60] seconds and a positive per-model budget")
    widths = sorted(set(operator.index(w) for w in angle_widths))
    if not widths or min(widths) < 0 or max(widths) > 128 or not inputs:
        raise ValueError("Inputs and angle widths in [0,128] are required")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    (output / "inputs").mkdir()
    (output / "source-snapshot").mkdir()
    source_hashes = {}
    for name in ("experiment_angle_orbits.py", "lazy_phase_solver.py", "lazy_pair_oracle.py", "translation_union.py"):
        source = ROOT / "scripts" / name
        shutil.copy2(source, output / "source-snapshot" / name)
        source_hashes[name] = _sha(source)
    sources = []
    for index, path in enumerate(inputs):
        source = Path(path).resolve()
        raw = load(source)
        if raw.shape != (15, 3):
            raise ValueError(f"Expected 15 semicircles in {source}")
        copy = output / "inputs" / f"source-{index:03d}.json"
        shutil.copy2(source, copy)
        sources.append((scored_once(raw), {"path": str(source), "sha256": _sha(source),
                                          "frozen_path": str(copy.relative_to(output))}))
    report = {"scope": "Finite fixed-center D4 angular models; local reference validation only, no API or promotion",
              "target": target, "acceptance_guard": ACCEPTANCE_GUARD,
              "seconds_budget": seconds, "per_model_seconds": per_model,
              "angle_widths": widths, "seed": seed, "source_hashes": source_hashes,
              "inputs": [record for _, record in sources], "models": []}
    started = time.monotonic()
    seen = {}
    best_score = None
    expected = len(widths) * 8 * len(sources)
    done = False
    for width in widths:
        for orbit, matrix in d4_matrices():
            for index, (geometry, source) in enumerate(sources):
                remaining = seconds - (time.monotonic() - started)
                if remaining <= 0:
                    done = True
                    break
                transformed = transform_scored(geometry, matrix)
                domains = make_angle_domains(transformed, width)
                key = domain_digest(domains)
                record = {"input_index": index, "input_sha256": source["sha256"], "orbit": orbit,
                          "matrix": matrix.tolist(), "angle_width": width, "domain_sha256": key,
                          "domain_sizes": list(map(len, domains))}
                if key in seen:
                    record.update(status="DUPLICATE_DOMAIN", duplicate_of=seen[key])
                else:
                    serial = len(report["models"])
                    seen[key] = serial
                    base = np.asarray([domain[0] for domain in domains])
                    model_path = output / f"model-{serial:03d}"
                    result = solve_phase(domains, base, target, model_path,
                                         seconds=min(per_model, remaining), seed=seed + serial,
                                         local_only=True)
                    record.update(status=result["status"], result=result)
                    if result.get("valid") and (best_score is None or result["score"] < best_score):
                        best_score = float(result["score"])
                        shutil.copy2(model_path / "solution.json", output / "best.json")
                        report["best"] = {"score": best_score, "model": serial, "path": "best.json",
                                          "sha256": _sha(output / "best.json")}
                report["models"].append(record)
                report.update(seconds=time.monotonic() - started, attempted=len(report["models"]),
                              scheduled=expected, remaining=expected - len(report["models"]))
                _write(output / "report.json", report)
                print(json.dumps({"model": len(report["models"]) - 1, "input": index, "orbit": orbit,
                                  "width": width, "status": record["status"], "best_score": best_score}), flush=True)
            if done:
                break
        if done:
            break
    report.update(seconds=time.monotonic() - started, attempted=len(report["models"]), scheduled=expected,
                  remaining=expected - len(report["models"]),
                  status="COMPLETE" if len(report["models"]) == expected else "BUDGET_EXHAUSTED")
    _write(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--target", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--per-model", type=float, default=2)
    parser.add_argument("--angle-width", type=int, action="append", help="repeat for a ladder; default3")
    parser.add_argument("--seed", type=int, default=3110927)
    args = parser.parse_args()
    report = run_orbits(args.input, args.target, args.output, seconds=args.seconds,
                        per_model=args.per_model, angle_widths=args.angle_width or [3], seed=args.seed)
    print(json.dumps({key: report[key] for key in ("status", "seconds", "attempted", "remaining")}, indent=2))


if __name__ == "__main__":
    main()
