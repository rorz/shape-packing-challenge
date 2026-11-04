"""Offline simultaneous two-disk fission, with a bounded structural gate.

Fuse two detected disjoint opposed-flat pairs into disks, choose both diameter
partitions jointly, and release all four semicircles before recompression.
Only local reference-valid candidates are saved; no API or incumbent writes.
"""
from __future__ import annotations

import os
for _key in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import itertools
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment, minimize

from experiment_lll import flat_contact_pairs
from experiment_surgery import Geometry, write_full
from experiment_surgery_graphs import graph
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.polish import AreaConstraints, checked, load, save

LEADER = 2.9485269238856113
INCUMBENT = 2.948526950912552
ROOT = Path(__file__).resolve().parents[1]


class DeadlineReached(RuntimeError):
    pass


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise DeadlineReached("Structural gate deadline reached")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def detect_pairs(placements):
    """Find exactly two disjoint active opposed-flat pairs, independent of labels."""
    p = np.asarray(placements, dtype=float)
    if p.shape != (15, 3) or not np.isfinite(p).all():
        raise ValueError("Expected 15 finite semicircle placements")
    radius = compute_mec_fast(p)[2]
    constraints = AreaConstraints(p)
    pairs = flat_contact_pairs(p, constraints, constraints.fun(np.r_[p.ravel(), radius]))
    if len(pairs) != 2 or len({i for pair in pairs for i in pair}) != 4:
        raise ValueError(f"Expected two disjoint active opposed-flat pairs, found {pairs}")
    return pairs


def fuse_pairs(placements, pairs):
    p = np.asarray(placements, dtype=float)
    removed = [i for pair in pairs for i in pair]
    if len(pairs) != 2 or len(set(removed)) != 4:
        raise ValueError("Fusion needs two disjoint pairs")
    keep = [i for i in range(len(p)) if i not in removed]
    macro = np.vstack([p[keep], *[np.r_[p[list(pair), :2].mean(axis=0), 0.] for pair in pairs]])
    if macro.shape != (13, 3):
        raise ValueError("Fusion must produce eleven semicircles and two disks")
    return macro, keep


def release_disks(macro, cuts, staggers):
    """Release both disks together; zero stagger preserves their exact unions."""
    macro = np.asarray(macro, dtype=float)
    if macro.shape != (13, 3) or len(cuts) != 2 or len(staggers) != 2:
        raise ValueError("Expected a two-disk macro packing and two cuts/staggers")
    rows = list(macro[:11].copy())
    for disk, cut, stagger in zip(macro[11:], cuts, staggers):
        tangent = np.array([-np.sin(cut), np.cos(cut)])
        rows.extend([np.r_[disk[:2] + .5 * stagger * tangent, cut],
                     np.r_[disk[:2] - .5 * stagger * tangent, cut + np.pi]])
    return np.asarray(rows)


def cut_partitions(macro, radius, disk_index, maximum=8):
    """Contact-partition midpoints, then evenly spread supplementary diameters."""
    p = np.asarray(macro)
    center = p[disk_index, :2]
    bearings = []
    for j, row in enumerate(p):
        if j == disk_index:
            continue
        delta = center - row[:2]
        length = float(np.linalg.norm(delta))
        if length < 1e-12:
            continue
        normal = np.array([np.cos(row[2]), np.sin(row[2])])
        tangent = np.array([-normal[1], normal[0]])
        point = row[:2] + delta / length if j >= 11 or delta @ normal >= 0 else row[:2] + tangent * np.clip(delta @ tangent, -1., 1.)
        if np.linalg.norm(point - center) - 1 < 3e-5:
            bearings.append(float(np.arctan2(*(point - center)[::-1])))
    if radius - np.linalg.norm(center) - 1 < 3e-5:
        bearings.append(float(np.arctan2(center[1], center[0])))
    events = sorted(set(round((a - np.pi / 2) % np.pi, 10) for a in bearings))
    partitions = []
    for k, angle in enumerate(events):
        end = events[(k + 1) % len(events)] + (np.pi if k == len(events) - 1 else 0.)
        partitions.append((end - angle, float(((angle + end) / 2) % np.pi)))
    # If unusually many contact partitions occur, retain the widest explicitly.
    cuts = [a for _, a in sorted(partitions, reverse=True)[:maximum]]
    grid = list(np.linspace(0, np.pi, 32, endpoint=False))
    while len(cuts) < maximum:
        def distance(a):
            return min((abs(np.arctan2(np.sin(2 * (a - b)), np.cos(2 * (a - b)))) / 2 for b in cuts), default=np.pi)
        chosen = max(grid, key=distance)
        if distance(chosen) < 1e-10:
            break
        cuts.append(float(chosen))
        grid.remove(chosen)
    return sorted(cuts), {"contact_events": events, "partition_count": len(partitions),
                          "partitions_omitted": max(0, len(partitions) - maximum)}


def schedule(macros, seed):
    """Shuffle joint moves, interleaving macro starts so neither can starve."""
    rng = np.random.default_rng(seed)
    queues = []
    for macro in macros:
        moves = list(itertools.product(macro["cuts"][0], macro["cuts"][1], (0., .3, -.3), (0., .3, -.3)))
        rng.shuffle(moves)
        queues.append(moves)
    return [(index, move) for layer in range(max(map(len, queues), default=0))
            for index, queue in enumerate(queues) if layer < len(queue) for move in [queue[layer]]]


def congruent(a, b, tolerance=1e-5):
    """Rigid/reflection/permutation congruence using positions and normals."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return False
    x, y = a[:, :2] - a[:, :2].mean(axis=0), b[:, :2] - b[:, :2].mean(axis=0)
    da = np.sort(np.linalg.norm(x[:, None] - x[None, :], axis=2).ravel())
    db = np.sort(np.linalg.norm(y[:, None] - y[None, :], axis=2).ravel())
    if np.max(np.abs(da - db)) > 2 * tolerance:
        return False
    for reflection in (1., -1.):
        reflected = x * np.array([1., reflection])
        angles = a[:, 2] * reflection
        for target in b[:, 2]:
            rotation = target - angles[0]
            cs, sn = np.cos(rotation), np.sin(rotation)
            transformed = reflected @ np.array([[cs, sn], [-sn, cs]])
            pos = np.linalg.norm(transformed[:, None] - y[None, :], axis=2)
            angle = np.abs(np.arctan2(np.sin(angles[:, None] + rotation - b[None, :, 2]),
                                      np.cos(angles[:, None] + rotation - b[None, :, 2])))
            costs = np.maximum(pos, angle)
            rows, cols = linear_sum_assignment(costs)
            if np.max(costs[rows, cols]) <= tolerance:
                return True
    return False


def optimize_geometry(geometry, placements, radius, deadline, *, full_disks=(), iterations=160, area=False):
    p = np.asarray(placements, dtype=float).copy()
    z = np.r_[p.ravel(), radius]
    direction = np.zeros(len(z)); direction[-1] = 1.
    bounds = [(None, None)] * len(z)
    bounds[-1] = (1., 4.)
    for i in full_disks:
        z[3 * i + 2] = 0.
        bounds[3 * i + 2] = (0., 0.)
    check_deadline(deadline)
    result = minimize(lambda v: v[-1], z, jac=lambda v: direction,
                      constraints={"type": "ineq", "fun": geometry.fun, "jac": geometry.jac},
                      bounds=bounds, method="SLSQP", callback=lambda _: check_deadline(deadline),
                      options={"ftol": 2e-13 if area else 2e-11, "maxiter": iterations})
    minimum = float(geometry.fun(result.x).min())
    return result.x[:-1].reshape(-1, 3), float(result.x[-1]), minimum, {
        "iterations": int(result.nit), "success": bool(result.success), "message": str(result.message)}


def relax_geometry(geometry, placements, radius, deadline):
    def objective(v):
        check_deadline(deadline)
        gaps, jac = geometry.evaluate(np.r_[v, radius])
        negative = np.minimum(gaps, 0.)
        return .5 * float(negative @ negative), negative @ jac[:, :-1]
    result = minimize(objective, np.asarray(placements).ravel(), jac=True, method="L-BFGS-B",
                      options={"maxiter": 100, "ftol": 1e-14, "gtol": 2e-7, "maxls": 30})
    return result.x.reshape(15, 3), float(result.fun)


def native_probe(placements, radius, output, *, seconds=10., seed=3111501, target=LEADER):
    """Short generic45-coordinate probe, without arena calls or global files."""
    from experiment_batch_lattice import Frame, NativeBabai
    from breakout_geometry import evaluate_batch
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    p = np.asarray(placements).copy()
    cx, cy, radius = compute_mec_fast(p)
    p[:, :2] -= [cx, cy]
    decoder = NativeBabai(); decoder.reset()
    rng = np.random.default_rng(seed)
    args = SimpleNamespace(target=target + 3e-8, regularization=1e-5, flat_weight=.1,
                           block_size=20, concentration=1., boundary_bias=1., movement=50.)
    counts = {key: 0 for key in ("frames", "rejected_frames", "proposals", "reference_checks", "valid")}
    best = None; nearest = None
    started = time.monotonic(); deadline = started + seconds
    while time.monotonic() < deadline:
        try:
            frame = Frame(p, radius, float(rng.uniform(-np.pi, np.pi)), args)
        except ValueError:
            counts["rejected_frames"] += 1
            if counts["rejected_frames"] >= 16 and counts["frames"] == 0:
                break
            continue
        counts["frames"] += 1
        if frame.budget <= 0:
            continue
        ambient, _, gso = frame.targets(rng, 2048, args)
        decoded = decoder.decode(gso, frame.mu, frame.moves, frame.anchor_units)
        frame.patch(ambient, decoded)
        actual, units, coeff, _ = decoded
        keep, _ = decoder.unique(units)
        actual, coeff = actual[keep], coeff[keep]
        counts["proposals"] += len(actual)
        if not len(actual):
            continue
        areas, circles, _ = evaluate_batch(actual, centers=[0., 0.])
        losses = np.maximum(np.maximum((areas - 1e-6) / frame.constraint.scale, 0.).max(axis=1),
                            np.maximum(circles[:, 2] - target, 0.))
        index = int(np.argmin(losses))
        if nearest is None or float(losses[index]) < nearest:
            nearest = float(losses[index]); save(output / "nearest-raw.json", frame.raw(coeff[index]))
        eligible = np.flatnonzero((areas.max(axis=1) <= 1e-6 + 1e-11) & (circles[:, 2] < INCUMBENT + 1e-7))
        for index in eligible[np.argsort(circles[eligible, 2])][:4]:
            if time.monotonic() >= deadline:
                break
            raw = frame.raw(coeff[index]); validation = checked(raw); counts["reference_checks"] += 1
            if validation.valid:
                counts["valid"] += 1
                if best is None or validation.score < best:
                    best = float(validation.score); save(output / "best.json", raw)
                    print(json.dumps({"event": "probe_valid", "score": best, "path": str(output / "best.json")}), flush=True)
    report = {"seconds": time.monotonic() - started, "budget_seconds": seconds, "seed": seed,
              "counts": counts, "best_score": best, "best_deficit": nearest,
              "scope": "Offline generic45 native probe; reference validation only"}
    write_json(output / "report.json", report)
    return report


def run(inputs, output, *, seconds=180., seed=3111501, probe_seconds=10., max_probes=4):
    if not 1 <= len(inputs) <= 2 or not 0 < seconds <= 180 or not 0 <= probe_seconds <= 10:
        raise ValueError("Use one/two macro starts, a gate in (0,180] seconds, and probes in [0,10] seconds")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite run evidence")
    output.mkdir(parents=True, exist_ok=True)
    snapshots = output / "source-snapshot"; snapshots.mkdir()
    source_hashes = {}
    for name in ("experiment_joint_fission.py", "experiment_surgery.py", "experiment_surgery_graphs.py", "experiment_lll.py", "experiment_batch_lattice.py", "_batch_lattice.cpp", "breakout_geometry.py", "_breakout_geometry.cpp"):
        source = ROOT / "scripts" / name
        shutil.copy2(source, snapshots / name); source_hashes[name] = sha(source)
    problem_records = []
    parents = []
    for i, source in enumerate(map(Path, inputs)):
        shutil.copy2(source, output / f"input-{i}.json")
        p = load(source); pairs = detect_pairs(p); parents.append(p)
        problem_records.append({"path": str(source), "sha256": sha(source), "pairs": pairs})
    # Compile structural and native kernels before starting the bounded gate.
    unit = Geometry(np.ones(15))
    macro_geometry = Geometry(np.ones(13), np.r_[np.zeros(11, dtype=int), 1, 1])
    from experiment_batch_lattice import NativeBabai
    NativeBabai()
    started = time.monotonic(); deadline = started + seconds
    manifest = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "inputs": problem_records,
                "source_hashes": source_hashes, "seed": seed, "gate_seconds": seconds,
                "probe_seconds": probe_seconds, "max_probes": max_probes,
                "command": [sys.executable, *sys.argv], "macros": [], "seeds": [],
                "scope": "Finite simultaneous two-disk fission gate; offline, no API/global promotion",
                "novelty_scope": "Compared with both input geometries and accepted seeds from this run using rigid/reflection/permutation congruence; contact graph signatures are recorded, not used alone as equivalence."}
    counts = {key: 0 for key in ("trials", "strict_feasible", "near_competitive", "duplicate_strict", "area_refinements", "competitive", "duplicate_continuous", "new_seeds")}
    macros = []; strict_seen = []; continuous_seen = [p.copy() for p in parents]
    graph_seen = []
    events = output / "events.jsonl"
    def event(kind, **data):
        record = {"event": kind, "seconds": time.monotonic() - started, **data}
        with events.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        if kind != "trial" or counts["trials"] % 25 == 0:
            print(json.dumps(record, allow_nan=False), flush=True)
    try:
        strict_radii = []
        for i, (parent, problem) in enumerate(zip(parents, problem_records)):
            strict, strict_radius, gap, opt = optimize_geometry(unit, parent, 3., deadline)
            if gap < -2e-8:
                event("strict_parent_failed", input=i, radius=strict_radius, gap=gap, solver=opt)
                continue
            strict_seen.append(strict); strict_radii.append(strict_radius)
            graph_seen.append(graph(strict, strict_radius, unit))
            macro, kept = fuse_pairs(strict, problem["pairs"])
            macro, radius, gap, opt = optimize_geometry(macro_geometry, macro, 3.15, deadline, full_disks=(11, 12), iterations=220)
            record = {"input": i, "kept_labels": kept, "radius": radius, "minimum_gap": gap, "solver": opt}
            if gap < -2e-8:
                event("macro_failed", **record); continue
            cuts0, audit0 = cut_partitions(macro, radius, 11)
            cuts1, audit1 = cut_partitions(macro, radius, 12)
            write_full(output / f"macro-{i}.json", macro)
            record.update(cuts=[cuts0, cuts1], cut_audits=[audit0, audit1])
            manifest["macros"].append(record)
            macros.append({"placements": macro, "radius": radius, "cuts": [cuts0, cuts1], "input": i})
            event("macro_ready", **record)
        jobs = schedule(macros, seed)
        manifest["scheduled"] = len(jobs)
        strict_gate = max(strict_radii, default=0.) + 1e-4
        for macro_index, move in jobs:
            check_deadline(deadline)
            macro = macros[macro_index]; cuts, staggers = move[:2], move[2:]
            p = release_disks(macro["placements"], cuts, staggers)
            loss = 0.
            if any(staggers):
                p, loss = relax_geometry(unit, p, max(3.12, macro["radius"] + .03), deadline)
            p, radius, gap, opt = optimize_geometry(unit, p, max(3.12, macro["radius"]), deadline)
            counts["trials"] += 1
            record = {"trial": counts["trials"], "macro": macro_index, "cuts": cuts,
                      "staggers": staggers, "relax_loss": loss, "radius": radius,
                      "minimum_gap": gap, "solver": opt}
            if gap < -2e-8:
                record["status"] = "STRICT_INFEASIBLE"; event("trial", **record); continue
            counts["strict_feasible"] += 1
            if radius > strict_gate:
                record["status"] = "OUTSIDE_CONTINUOUS_GATE"; event("trial", **record); continue
            counts["near_competitive"] += 1
            if any(congruent(p, prior) for prior in strict_seen):
                counts["duplicate_strict"] += 1
                record["status"] = "KNOWN_STRICT_GEOMETRY"; event("trial", **record); continue
            strict_seen.append(p.copy())
            counts["area_refinements"] += 1
            constraint = AreaConstraints(p)
            q, area_radius, area_gap, area_opt = optimize_geometry(constraint, p, radius, deadline, iterations=220, area=True)
            record.update(area_radius=area_radius, area_minimum_gap=area_gap, area_solver=area_opt)
            if area_gap < -2e-9 or area_radius > LEADER - 8e-8:
                record["status"] = "NOT_COMPETITIVE"; event("trial", **record); continue
            counts["competitive"] += 1
            if any(congruent(q, prior) for prior in continuous_seen):
                counts["duplicate_continuous"] += 1
                record["status"] = "KNOWN_CONTINUOUS_GEOMETRY"; event("trial", **record); continue
            continuous_seen.append(q.copy())
            structural_graph = graph(p, radius, unit)
            same_graph = any(nx.is_isomorphic(structural_graph, prior, node_match=lambda a, b: a["boundary"] == b["boundary"]) for prior in graph_seen)
            graph_seen.append(structural_graph)
            index = counts["new_seeds"]; counts["new_seeds"] += 1
            path = output / f"continuous-seed-{index:03d}.json"; write_full(path, q)
            validation = checked(q)
            seed_record = {"index": index, "file": path.name, "sha256": sha(path),
                           "continuous_radius": area_radius, "minimum_gap": area_gap,
                           "same_boundary_colored_graph_as_prior": same_graph,
                           "graph_hash": nx.weisfeiler_lehman_graph_hash(structural_graph, node_attr="boundary"),
                           "rounded_valid": validation.valid, "rounded_score": validation.score,
                           "trial": counts["trials"]}
            if validation.valid:
                save(output / f"rounded-valid-{index:03d}.json", q)
            manifest["seeds"].append(seed_record)
            event("new_seed", **seed_record)
            write_json(output / "report.json", {**manifest, "counts": counts, "status": "RUNNING"})
            record["status"] = "NEW_COMPETITIVE_GEOMETRY"; event("trial", **record)
        manifest["status"] = "SCHEDULE_COMPLETE"
    except DeadlineReached:
        manifest["status"] = "GATE_DEADLINE"
    manifest["gate_elapsed_seconds"] = time.monotonic() - started
    manifest["counts"] = counts
    write_json(output / "report.json", manifest)
    event("gate_done", status=manifest["status"], counts=counts)
    # Probe only new competitive geometries; generic Frame handles dynamic labels.
    if probe_seconds:
        chosen = sorted(manifest["seeds"], key=lambda row: row["continuous_radius"])[:max_probes]
        for row in chosen:
            row["native_probe"] = native_probe(load(output / row["file"]), row["continuous_radius"], output / f"probe-{row['index']:03d}", seconds=probe_seconds, seed=seed + 1000 + row["index"])
            write_json(output / "report.json", manifest)
    manifest["total_seconds"] = time.monotonic() - started
    write_json(output / "report.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=180.)
    parser.add_argument("--seed", type=int, default=3111501)
    parser.add_argument("--probe-seconds", type=float, default=10.)
    parser.add_argument("--max-probes", type=int, default=4)
    args = parser.parse_args()
    report = run(args.input, args.output, seconds=args.seconds, seed=args.seed,
                 probe_seconds=args.probe_seconds, max_probes=args.max_probes)
    print(json.dumps({k: report[k] for k in ("status", "counts", "gate_elapsed_seconds", "total_seconds")}, indent=2))


if __name__ == "__main__":
    main()
