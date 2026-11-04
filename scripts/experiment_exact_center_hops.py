"""Offline exact coordinated center hops with translation-invariant novelty.

This is a private extension of the lazy phase algorithm. Shared production
solvers are not modified. Every reported hop passes independent reference
validation, and angular variants or common integer translations never count.
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
import time
from pathlib import Path

import numpy as np
from ortools.sat.python import cp_model

import experiment_conflict_master as master
from experiment_angle_orbits import d4_matrices, make_angle_domains, scored_once, transform_scored
from experiment_support_bundle import support_bundle
from interval_order_optimization import add_hamming_interval_pair
from lazy_pair_oracle import PairOracle
from lazy_phase_solver import ACCEPTANCE_GUARD, solve_phase, submission_for_geometry
from separable_support import Inserter, audit, factor, piece_values
from semicircle_packing.fast_mec import compute_mec_fast
from semicircle_packing.geometry import Semicircle
from semicircle_packing.polish import checked, load, save
from semicircle_packing.scoring import compute_mec
from translation_union import domain_digest, make_translation_domains

ROOT = Path(__file__).resolve().parents[1]
LEADER = 2.9485269238856113


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def center_units(poses):
    p = np.asarray(poses, dtype=float)
    if p.ndim != 2 or p.shape[1] < 2 or not len(p) or not np.all(np.isfinite(p)):
        raise ValueError("Expected nonempty finite pose rows")
    xy = np.rint(p[:, :2] * 1e6).astype(np.int64)
    if not np.allclose(xy / 1e6, p[:, :2], rtol=0, atol=1e-12):
        raise ValueError("Center novelty requires six-decimal coordinates")
    return xy - xy[0].copy()


def center_hash(poses):
    return hashlib.sha256(center_units(poses).astype("<i8").tobytes()).hexdigest()


def make_hop_domains(physical, xy_width=1, angle_width=1):
    """Union all XY phases, using the orbit helper's seam-safe angle choices."""
    xy_domains = make_translation_domains(physical, xy_width=xy_width,
                                           angle_width=0, max_poses=2000)
    angles = make_angle_domains(physical, angle_width)
    domains = []
    for xy_domain, angular in zip(xy_domains, angles):
        xy = np.unique(np.rint(xy_domain[:, :2] * 1e6).astype(np.int64), axis=0)
        theta = np.rint(angular[:, 2] * 1e6).astype(np.int64)
        if len(xy) * len(theta) > 2000:
            raise ValueError("Center-hop domain exceeds 2000 poses")
        first = tuple(np.rint(angular[0] * 1e6).astype(np.int64))
        rows = [first]
        rows.extend((int(x), int(y), int(t)) for (x, y), t in itertools.product(xy, theta)
                    if (int(x), int(y), int(t)) != first)
        domains.append(np.asarray(rows, dtype=float) / 1e6)
    return domains


def add_center_novelty(model, variables, domains, blocked):
    """Require a relative XY vector different from every blocked arrangement.

    Encodes both directions of each equality so a Boolean cannot manufacture
    novelty. Angles are absent from the constraint. Piece labels are retained;
    rotations/reflections remain separate search states.
    """
    units = [np.rint(np.asarray(d)[:, :2] * 1e6).astype(np.int64) for d in domains]
    coordinates = []
    for i, (variable, values) in enumerate(zip(variables, units)):
        row = []
        for axis in range(2):
            column = values[:, axis].tolist()
            coordinate = model.new_int_var(min(column), max(column), f"center_{i}_{axis}")
            model.add_element(variable, column, coordinate)
            row.append(coordinate)
        coordinates.append(row)
    unique = {}
    for poses in blocked:
        relative = center_units(poses)
        if len(relative) != len(domains):
            raise ValueError("Blocked center count differs from domain count")
        unique.setdefault(relative.astype("<i8").tobytes(), relative)
    for serial, relative in enumerate(unique.values()):
        differs = []
        for i in range(1, len(domains)):
            for axis in range(2):
                same = model.new_bool_var(f"same_{serial}_{i}_{axis}")
                expression = coordinates[i][axis] - coordinates[0][axis]
                model.add(expression == int(relative[i, axis])).only_enforce_if(same)
                model.add(expression != int(relative[i, axis])).only_enforce_if(same.Not())
                differs.append(same.Not())
        model.add_bool_or(differs)
    return len(unique)


def solve_hop(domains, base, target, blocked, output, *, seconds=5, seed=3111601):
    """Find a reference-valid new center arrangement in explicit finite domains."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + seconds
    np.savez_compressed(output / "domains.npz", **{f"p{i}": d for i, d in enumerate(domains)})
    result = {"valid": False, "target": target, "acceptance_guard": ACCEPTANCE_GUARD,
              "domain_sha256": domain_digest(domains), "domain_sizes": list(map(len, domains)),
              "scope": "Finite target model excluding listed labeled centers modulo integer translation"}
    try:
        oracle = PairOracle(domains, deadline=deadline)
    except TimeoutError:
        result.update(status="PREPARATION_DEADLINE", seconds=time.monotonic() - started)
        write_json(output / "summary.json", result)
        return result
    model = cp_model.CpModel()
    variables = [model.new_int_var(0, len(d) - 1, f"pose_{i}") for i, d in enumerate(domains)]
    for i, j, allowed in oracle.relaxed_tables():
        add_hamming_interval_pair(model, variables[i], variables[j], allowed, len(domains[i]), len(domains[j]))
    result["blocked_centers"] = add_center_novelty(model, variables, domains, blocked)
    circle = compute_mec_fast(base)
    cuts = support_bundle(domains, base, circle)
    write_json(output / "support-audit.json", audit(domains, cuts))
    inserter = Inserter(domains)
    radius = model.new_int_var(0, int(np.floor(target * 1e12)), "radius_lower_bound")
    for index, cut in enumerate(cuts):
        inserter(model, variables, radius, cut, index)
    model.minimize(radius)
    for variable in variables:
        model.add_hint(variable, 0)
    counts = dict(iterations=0, pair_nogoods=0, enclosure_cuts=0, enclosure_nogoods=0,
                  reference_checks=0)
    blocked_keys = {center_hash(p) for p in blocked}
    while time.monotonic() < deadline:
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max(.0001, deadline - time.monotonic())
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = seed + counts["iterations"]
        solver.parameters.stop_after_first_solution = True
        status = solver.solve(model)
        counts["iterations"] += 1
        if status == cp_model.INFEASIBLE:
            result.update(status="INFEASIBLE", finite_exhausted=True)
            break
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            result["status"] = solver.status_name(status)
            break
        assignment = [solver.value(v) for v in variables]
        bad = oracle.check(assignment)
        if bad:
            for conflict in bad:
                i, j = conflict["pair"]
                model.add_forbidden_assignments([variables[i], variables[j]], [conflict["values"]])
                counts["pair_nogoods"] += 1
        else:
            p = np.asarray([d[k] for d, k in zip(domains, assignment)])
            assert center_hash(p) not in blocked_keys, "CP returned a blocked center arrangement"
            circle = compute_mec_fast(p)
            raw, scored = submission_for_geometry(p)
            validation = checked(raw)
            counts["reference_checks"] += 1
            if validation.valid and validation.score <= target + ACCEPTANCE_GUARD:
                if time.monotonic() > deadline:
                    result.update(status="LATE_VALID", late_score=float(validation.score))
                    save(output / "late-solution.json", raw)
                else:
                    save(output / "solution.json", raw)
                    result.update(status="NEW_VALID_CENTER", valid=True, score=float(validation.score),
                                  center_hash=center_hash(scored), raw_placements=raw.tolist(),
                                  assignment=assignment, solution_path=str(output / "solution.json"))
                break
            cut = master.categorical_support_cut(domains, p, circle)
            if cut is not None:
                inserter(model, variables, radius, cut, f"learned_{counts['enclosure_cuts']}")
                counts["enclosure_cuts"] += 1
            factored = None if cut is None else factor(cut, inserter.prepared)
            rejects = False if factored is None else sum(
                int(piece_values(factored, inserter.prepared, i)[assignment[i]])
                for i in factored["pieces"]) > int(np.floor(target * 1e12)) * factored["ratio"] + factored["offset"]
            if not rejects:
                owners, _ = master.boundary_conflict(p, circle[:2], target)
                if owners is None:
                    reference = compute_mec([Semicircle(*v) for v in p])
                    if reference[2] <= target + ACCEPTANCE_GUARD:
                        result.update(status="NUMERIC_ENCLOSURE_BAND", reference_mec=reference)
                        break
                    owners = list(range(len(domains)))
                model.add_forbidden_assignments([variables[i] for i in owners],
                                                [[assignment[i] for i in owners]])
                counts["enclosure_nogoods"] += 1
        model.clear_hints()
        for variable, value in zip(variables, assignment):
            model.add_hint(variable, value)
    else:
        result["status"] = "DEADLINE"
    result.update(seconds=time.monotonic() - started, counts=counts, oracle=oracle.stats)
    write_json(output / "summary.json", result)
    return result


def run(input_path, output, *, seconds=120, per_model=5, seed=3111601,
        relay_billionths=(50, 100), angle_widths=(1, 3), tighten_seconds=30):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    source = Path(input_path)
    raw = load(source)
    validation = checked(raw)
    if not validation.valid:
        raise ValueError("Hop seed must be reference-valid")
    scored = scored_once(raw)
    shutil.copy2(source, output / "seed.json")
    snapshot = output / "source-snapshot"
    snapshot.mkdir()
    hashes = {}
    for name in ("experiment_exact_center_hops.py", "experiment_angle_orbits.py", "translation_union.py",
                 "lazy_phase_solver.py", "lazy_pair_oracle.py", "experiment_support_bundle.py",
                 "separable_support.py", "experiment_conflict_master.py"):
        path = ROOT / "scripts" / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copy2(path, snapshot / name)
    report = dict(input=str(source), input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  initial_score=float(validation.score), source_hashes=hashes, seed=seed,
                  seconds_budget=seconds, per_model_seconds=per_model,
                  relay_billionths=list(relay_billionths), angle_widths=list(angle_widths),
                  novelty="Labeled centers modulo common integer translation; all D4 orientations retained",
                  models=[], valid_centers=[], scope="Offline finite center-hop gate; no API or promotion")
    started = time.monotonic()
    deadline = started + seconds
    found = None
    for width, relay, (orbit, matrix) in itertools.product(angle_widths, relay_billionths, d4_matrices()):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or found is not None:
            break
        physical = transform_scored(scored, matrix)
        domains = make_hop_domains(physical, xy_width=1, angle_width=width)
        base = np.asarray([d[0] for d in domains])
        serial = len(report["models"])
        target = validation.score + relay * 1e-9
        result = solve_hop(domains, base, target, [base], output / f"model-{serial:03d}",
                           seconds=min(per_model, remaining), seed=seed + serial)
        record = dict(orbit=orbit, angle_width=width, relay_billionths=relay, result=result)
        report["models"].append(record)
        print(json.dumps({"event": "model", "serial": serial, "orbit": orbit, "width": width,
                          "relay": relay, "status": result["status"], "seconds": result["seconds"]}), flush=True)
        if result.get("valid"):
            found = np.asarray(result["raw_placements"])
            independent = checked(found)
            assert independent.valid and independent.score <= target + ACCEPTANCE_GUARD
            assert center_hash(found) != center_hash(base)
            shutil.copy2(result["solution_path"], output / "hop.json")
            report["valid_centers"].append(dict(score=float(independent.score),
                                               center_hash=center_hash(found), file="hop.json"))
            print(json.dumps({"event": "new_center", "score": independent.score,
                              "file": str(output / "hop.json")}), flush=True)
        report["gate_seconds"] = time.monotonic() - started
        write_json(output / "report.json", report)
    report["gate_seconds"] = time.monotonic() - started
    report["gate_passed"] = found is not None
    if found is not None and tighten_seconds > 0:
        # Tightening has a separate small budget; it is not credited to the hop gate.
        physical = transform_scored(scored_once(found), np.eye(2, dtype=int))
        domains = make_hop_domains(physical, xy_width=1, angle_width=3)
        base = np.asarray([d[0] for d in domains])
        report["tightening"] = solve_phase(domains, base, LEADER - 5e-12,
                                           output / "tightening", seconds=tighten_seconds,
                                           seed=seed + 1000, local_only=True)
        if report["tightening"].get("valid"):
            shutil.copy2(output / "tightening" / "solution.json", output / "candidate.json")
    report["seconds"] = time.monotonic() - started
    report["status"] = "GATE_PASSED" if found is not None else "NO_NEW_CENTER"
    write_json(output / "report.json", report)
    return report


def run_graph(incumbent_path, seed_paths, output, *, seconds=360, per_model=4,
              tighten_seconds=5, beam_size=16, seed=3111701):
    """Traverse validated centers under barriers anchored to one frozen incumbent.

    Finite-domain infeasibility is cached only for that domain and cap; the
    blocked set grows monotonically. Positive/unknown domains are not treated
    as exclusions. Queue breadth is a search heuristic, not a coverage claim.
    """
    if seconds <= 0 or per_model <= 0 or tighten_seconds <= 0 or beam_size < 1:
        raise ValueError("Budgets and beam size must be positive")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    incumbent_raw = load(incumbent_path)
    incumbent_check = checked(incumbent_raw)
    if not incumbent_check.valid:
        raise ValueError("Global incumbent must be valid")
    fixed_score = float(incumbent_check.score)
    shutil.copy2(incumbent_path, output / "incumbent.json")
    source_names = ("experiment_exact_center_hops.py", "experiment_angle_orbits.py", "translation_union.py",
                    "lazy_phase_solver.py", "lazy_pair_oracle.py", "experiment_support_bundle.py",
                    "separable_support.py", "experiment_conflict_master.py")
    snapshot = output / "source-snapshot"
    snapshot.mkdir()
    source_hashes = {}
    for name in source_names:
        source = ROOT / "scripts" / name
        shutil.copy2(source, snapshot / name)
        source_hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    schedule = [(width, relay, name, matrix) for width, relay, (name, matrix)
                in itertools.product((1, 3), (50, 100), d4_matrices())]
    nodes = []
    blocked = []
    visited = set()
    queue = []
    done_jobs = set()
    exhausted = []
    tightened = set()
    report = dict(scope="Bounded exact valid-center graph; no API or global promotion",
                  fixed_incumbent_score=fixed_score, relay_caps=[fixed_score + v * 1e-9 for v in (50, 100)],
                  seconds_budget=seconds, per_model_seconds=per_model,
                  tightening_seconds=tighten_seconds, beam_size=beam_size, seed=seed,
                  source_hashes=source_hashes, nodes=nodes, models=[], tightening=[],
                  queue_evictions=0, domain_exclusions_reused=0,
                  novelty="Labeled XY relative to piece zero; D4 orientations kept distinct")
    started = time.monotonic()
    deadline = started + seconds

    def checkpoint():
        report.update(seconds=time.monotonic() - started, queued=len(queue),
                      distinct_valid_centers=len(nodes), blocked_center_count=len(visited))
        write_json(output / "report.json", report)

    def emit(event, **data):
        record = dict(event=event, seconds=time.monotonic() - started, **data)
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    def enqueue(raw, source, score):
        geometry = scored_once(raw)
        key = center_hash(geometry)
        if key in visited:
            return False
        visited.add(key)
        blocked.append(geometry)
        index = len(nodes)
        path = output / f"node-{index:03d}.json"
        save(path, raw)
        nodes.append(dict(index=index, file=path.name, score=float(score), center_hash=key,
                          pose_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source=source))
        queue.append(dict(index=index, raw=np.asarray(raw).copy(), score=float(score), next_job=0,
                          urgent=source != "frozen_global_incumbent", expansions=0))
        if len(queue) > beam_size:
            # Retain a recent exploration slot as well as the low-radius beam.
            protected = queue[-1]
            worst = max((v for v in queue if v is not protected), key=lambda v: (v["score"], -v["index"]))
            queue[:] = [value for value in queue if value is not worst]
            report["queue_evictions"] += 1
        emit("valid_center", index=index, score=float(score), source=source, file=str(path))
        if score < report.get("best_score", float("inf")):
            report["best_score"] = float(score)
            shutil.copy2(path, output / "best.json")
        return True

    enqueue(incumbent_raw, "frozen_global_incumbent", fixed_score)
    for path in seed_paths:
        raw = load(path)
        validation = checked(raw)
        if not validation.valid or validation.score > fixed_score + 100e-9 + ACCEPTANCE_GUARD:
            raise ValueError(f"Seed {path} is invalid or above the fixed global barrier")
        enqueue(raw, str(path), validation.score)
    serial = 0
    winner = False
    while queue and time.monotonic() < deadline and not winner:
        # Newly accepted nodes get their first-place tightening immediately.
        # Otherwise use three best-first expansions and one least-visited turn.
        urgent = [v for v in queue if v["urgent"]]
        node = min(urgent, key=lambda v: v["score"]) if urgent else min(
            queue, key=(lambda v: (v["expansions"], v["index"])) if serial % 4 == 3
            else (lambda v: (v["score"], v["next_job"], v["index"])))
        node["urgent"] = False
        node["expansions"] += 1
        if node["next_job"] >= len(schedule):
            queue[:] = [value for value in queue if value is not node]
            continue
        width, relay, orbit, matrix = schedule[node["next_job"]]
        job_key = (node["index"], node["next_job"])
        node["next_job"] += 1
        if job_key in done_jobs:
            continue
        done_jobs.add(job_key)
        physical = transform_scored(scored_once(node["raw"]), matrix)
        # Every orientation gets a first-place attempt before its center-hop
        # exclusion. Thus omitting angle-only transitions cannot hide a win.
        tight_domains = make_hop_domains(physical, xy_width=1, angle_width=max(3, width))
        tight_key = domain_digest(tight_domains)
        if tight_key not in tightened and deadline > time.monotonic():
            tightened.add(tight_key)
            path = output / f"tight-{len(report['tightening']):03d}"
            base = np.asarray([d[0] for d in tight_domains])
            result = solve_phase(tight_domains, base, LEADER - 5e-12, path,
                                 seconds=min(tighten_seconds, deadline - time.monotonic()),
                                 seed=seed + 10000 + serial, local_only=True)
            report["tightening"].append(dict(node=node["index"], orbit=orbit,
                                             domain_sha256=tight_key, result=result))
            if result.get("valid"):
                raw = load(path / "solution.json")
                check = checked(raw)
                if check.valid and check.score < fixed_score:
                    save(output / "candidate.json", raw)
                    report["candidate"] = dict(score=float(check.score), file="candidate.json")
                    emit("candidate", score=float(check.score), file=str(output / "candidate.json"))
                    winner = check.score < LEADER
                    checkpoint()
                    if winner:
                        break
        if deadline <= time.monotonic():
            break
        domains = make_hop_domains(physical, xy_width=1, angle_width=width)
        base = np.asarray([d[0] for d in domains])
        key = domain_digest(domains)
        local_blocked = blocked + [base]
        blocked_keys = frozenset(center_hash(p) for p in local_blocked)
        # A looser proven cap excludes a tighter cap on these same domains,
        # only if every center excluded in the old proof is still excluded.
        if any(old_key == key and proven_relay >= relay and old_blocked <= blocked_keys
               for old_key, proven_relay, old_blocked in exhausted):
            report["domain_exclusions_reused"] += 1
            continue
        path = output / f"hop-{serial:03d}"
        result = solve_hop(domains, base, fixed_score + relay * 1e-9, local_blocked, path,
                           seconds=min(per_model, deadline - time.monotonic()), seed=seed + serial)
        report["models"].append(dict(node=node["index"], orbit=orbit, width=width,
                                     relay_billionths=relay, result=result))
        emit("model", serial=serial, node=node["index"], orbit=orbit, width=width,
             relay=relay, status=result["status"], model_seconds=result["seconds"])
        if result.get("finite_exhausted"):
            exhausted.append((key, relay, blocked_keys))
        if result.get("valid"):
            raw = np.asarray(result["raw_placements"])
            independent = checked(raw)
            assert independent.valid and independent.score <= fixed_score + relay * 1e-9 + ACCEPTANCE_GUARD
            enqueue(raw, dict(parent=node["index"], model=serial), independent.score)
            # Revisit this positive domain with the new center blocked, allowing
            # discovery of another neighbor without mistaking a SAT result for
            # complete domain coverage. The queue still gives new nodes turns.
            node["next_job"] -= 1
            done_jobs.remove(job_key)
        serial += 1
        checkpoint()
    report["status"] = "LOCAL_FIRST_PLACE_CANDIDATE" if winner else "BUDGET_OR_QUEUE_FINISHED"
    checkpoint()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("solution.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--per-model", type=float, default=5)
    parser.add_argument("--seed", type=int, default=3111601)
    parser.add_argument("--tighten-seconds", type=float, default=30)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--seed-input", type=Path, action="append", default=[])
    parser.add_argument("--beam-size", type=int, default=16)
    args = parser.parse_args()
    if args.graph:
        result = run_graph(args.input, args.seed_input, args.output, seconds=args.seconds,
                           per_model=args.per_model, seed=args.seed, beam_size=args.beam_size,
                           tighten_seconds=args.tighten_seconds)
    else:
        result = run(args.input, args.output, seconds=args.seconds, per_model=args.per_model,
                     seed=args.seed, tighten_seconds=args.tighten_seconds)
    print(json.dumps({key: result[key] for key in ("status", "gate_seconds", "seconds") if key in result}), flush=True)


if __name__ == "__main__":
    main()
