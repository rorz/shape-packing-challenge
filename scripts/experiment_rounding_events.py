"""Offline production comparison of adjacent rounding events and fresh frames.

The event locator brackets a change in the actual finite domain and bisects it.
It is a heuristic: a nonlinear ray may leave and re-enter a domain between
probes. Endpoint-domain unions contain both sampled sides, not every intervening
continuous frame. Every accepted submission still needs the reference validator.
"""
from __future__ import annotations

import os
for _key in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import heapq
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmark_recovery import fit_frame, scored_once, transform, write_json
from blind_packing_worker import center_hash, warmup
from breakout_geometry import evaluate_batch
from experiment_mirror_family import FLATS, Manifold
from experiment_batch_lattice import NativeBabai
from experiment_tied_batch import Frame
from lazy_phase_solver import solve_phase, submission_for_geometry
from semicircle_packing.polish import checked, load, save
from semicircle_packing.geometry import Semicircle, semicircle_polygon
from semicircle_packing.scoring import compute_mec
from translation_union import domain_digest, make_translation_domains

ROOT = Path(__file__).resolve().parents[1]
LEADER = 2.9485269238856113
DEFAULT_INPUTS = ["solution.json", "tests/fixtures/recovery/random-d-origin.json",
                  "tests/fixtures/recovery/neighbor-g-origin.json"]


def merge_domains(left, right, max_poses=2000):
    """Contain both endpoint pose sets; never truncate or quotient orientations."""
    if len(left) != len(right):
        raise ValueError("Endpoint domains have different piece counts")
    merged = []
    for a, b in zip(left, right):
        units = np.unique(np.rint(np.vstack([a, b]) * 1e6).astype(np.int64), axis=0)
        if len(units) > max_poses:
            raise ValueError("Endpoint union exceeds max_poses")
        # Keep the right endpoint's ordinary rounded pose as the solver hint.
        base = np.rint(b[0] * 1e6).astype(np.int64)
        units = np.vstack([base, units[np.any(units != base, axis=1)]])
        merged.append(units.astype(float) / 1e6)
    return merged


def locate_transition(evaluate, start, direction, *, initial=.05, maximum=8.,
                      iterations=12, deadline=float("inf")):
    """Approximately locate a changed domain along a parameter ray.

    Parameters are in micro-units. This is not a proof of the first transition:
    exponential bracketing can skip a leave/re-enter event. A return value always
    has different endpoint hashes and records the actual bracket and probes.
    """
    start = np.asarray(start, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if start.shape != direction.shape or not np.isfinite(start).all() or not np.isfinite(direction).all():
        raise ValueError("Finite same-shaped ray parameters required")
    if not np.any(direction) or not 0 < initial <= maximum or iterations < 0:
        raise ValueError("Invalid event-locator limits")
    before = evaluate(start)
    if before is None:
        return None
    origin_hash = before["hash"]
    low, high, evaluations = 0., initial, 1
    after = None
    while time.monotonic() < deadline:
        after = evaluate(start + high * direction)
        evaluations += 1
        if after is None:
            return None
        if after["hash"] != origin_hash:
            break
        low, before = high, after
        if high >= maximum:
            return None
        high = min(maximum, 2 * high)
    else:
        return None
    for _ in range(iterations):
        if time.monotonic() >= deadline:
            break
        middle = (low + high) * .5
        candidate = evaluate(start + middle * direction)
        evaluations += 1
        if candidate is None:
            return None
        if candidate["hash"] == origin_hash:
            low, before = middle, candidate
        else:
            high, after = middle, candidate
    assert before["hash"] != after["hash"]
    return {"before": before, "after": after, "low": low, "high": high,
            "direction": direction.tolist(), "evaluations": evaluations,
            "scope": "sampled transition bracket, not complete event coverage"}


class Origin:
    def __init__(self, path, seed, angle_width=1, raw_input=False):
        self.path = Path(path)
        self.raw = load(path)
        if self.raw.shape != (15, 3) or not np.isfinite(self.raw).all():
            raise ValueError("Expected 15 finite poses")
        self.model = Manifold(seed)
        target = scored_once(self.raw) if raw_input else self.raw
        self.branch = np.round([target[i, 2] - target[j, 2] for i, j in FLATS], 6)
        _, self.fit = fit_frame(self.model.p, target, model=self.model,
                                branch=self.branch.tolist())
        self.anchor = np.r_[np.asarray(self.fit["alpha"]) * 1e6,
                            self.fit["angle"] * 1e6]
        self.translation = np.asarray(self.fit["translation"])
        self.angle_width = angle_width
        self.cache = {}

    def evaluate(self, parameters):
        key = tuple(map(float, parameters))
        if key in self.cache:
            return self.cache[key]
        section, meta = self.model.section(self.branch.tolist(), np.asarray(parameters[:2]) * 1e-6)
        if max(meta["violation"], meta["residual"]) > 2e-9:
            self.cache[key] = None
            return None
        physical = transform(section, float(parameters[2]) * 1e-6, self.translation)
        domains = make_translation_domains(physical, 1, self.angle_width)
        record = {"parameters": list(key), "physical": physical,
                  "domains": domains, "hash": domain_digest(domains), "section": meta}
        if len(self.cache) >= 2048:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = record
        return record

    def describe(self):
        return {"path": str(self.path), "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                "branch": self.branch.tolist(), "fit": self.fit,
                "anchor_micro_parameters": self.anchor.tolist()}


def frontier_score(physical, scale, target):
    scored = scored_once(physical)
    areas, circles, status = evaluate_batch(scored[None], centers=[0., 0.])
    overlap = float(np.max(np.maximum(areas[0] - 1e-6, 0) / scale))
    radius = float(circles[0, 2])
    return {"loss": max(overlap, radius - target), "overlap_deficit": overlap,
            "radius": radius, "native_bad_pairs": int((areas[0] > 1e-6 + 1e-11).sum()),
            "native_mec_status": int(status[0])}


def pop_balanced(queue, preferred_origin):
    """Use the requested origin whenever it has work, then its best priority."""
    eligible = [(row[0], row[1], index) for index, row in enumerate(queue)
                if row[2]["origin"] == preferred_origin]
    if not eligible:
        return heapq.heappop(queue)
    index = min(eligible)[2]
    answer = queue[index]
    final = queue.pop()
    if index < len(queue):
        queue[index] = final
        heapq.heapify(queue)
    return answer


def eligible_native_indices(areas, circles, target):
    """Do not hide feasible candidates behind a lower-deficit invalid winner."""
    allowed = (np.max(areas, axis=1) <= 1e-6 + 1e-11) & (circles[:, 2] <= target + 1e-7)
    indices = np.flatnonzero(allowed)
    return indices[np.argsort(circles[indices, 2], kind="stable")]


def native_preview(state, origin, target, seed, decoder):
    """Proposal quality near this frame, explicitly not a domain lower bound."""
    rounded = scored_once(state["physical"])
    raw, _ = submission_for_geometry(rounded)
    metric = frontier_score(state["physical"], origin.model.c.scale, target)
    result = {"raw": raw, "scored": rounded, "metric": metric, "source": "rounded_frame",
              "inside_domain": True, "proposals": 1,
              "eligible": [raw] if metric["native_bad_pairs"] == 0 and metric["radius"] <= target + 1e-7 else []}
    settings = SimpleNamespace(target=target, regularization=1e-5, block_size=20,
                               concentration=1., boundary_bias=1., movement=10.)
    try:
        frame = Frame(state["physical"], state["section"]["radius"], FLATS, settings)
        rng = np.random.default_rng(int(state["hash"][:12], 16) ^ seed)
        actual, units, coefficients, _, _ = frame.sample(rng, 256, settings, decoder)
    except (ValueError, np.linalg.LinAlgError):
        return result
    areas, circles, status = evaluate_batch(actual, centers=[0., 0.])
    result["eligible"].extend(frame.raw(coefficients[i])
                              for i in eligible_native_indices(areas, circles, target))
    overlap = np.max(np.maximum(areas - 1e-6, 0) / frame.c.scale, axis=1)
    loss = np.maximum(overlap, circles[:, 2] - target)
    best = int(np.argmin(loss))
    result["proposals"] += len(actual)
    if loss[best] < metric["loss"]:
        candidate = actual[best]
        inside = all(np.any(np.all(np.rint(domain * 1e6).astype(np.int64) ==
                                   np.rint(row * 1e6).astype(np.int64), axis=1))
                     for domain, row in zip(state["domains"], candidate))
        result.update(raw=frame.raw(coefficients[best]), scored=candidate,
                      source="native_preview", inside_domain=bool(inside),
                      metric={"loss": float(loss[best]), "overlap_deficit": float(overlap[best]),
                              "radius": float(circles[best, 2]),
                              "native_bad_pairs": int((areas[best] > 1e-6 + 1e-11).sum()),
                              "native_mec_status": int(status[best])})
    return result


def exact_frontier(raw, scale):
    """Save the real full-submission frontier rather than a rounded-frame proxy."""
    scored = scored_once(raw)
    shapes = [Semicircle(*row) for row in scored]
    polygons = [semicircle_polygon(shape) for shape in shapes]
    bad, deficits, pair = [], [], 0
    for i in range(15):
        for j in range(i + 1, 15):
            area = polygons[i].intersection(polygons[j]).area
            deficits.append(max(area - 1e-6, 0.) / scale[pair])
            if area > 1e-6:
                bad.append({"pair": [i, j], "area": float(area)})
            pair += 1
    return {"reference_mec": list(compute_mec(shapes)), "exact_bad_pairs": bad,
            "maximum_exact_normalized_overlap_deficit": float(max(deficits))}


def source_snapshot(output, inputs):
    folder = output / "source-snapshot"
    folder.mkdir()
    hashes = {}
    names = ["experiment_rounding_events.py", "benchmark_recovery.py", "blind_packing_worker.py",
             "experiment_mirror_family.py", "translation_union.py", "lazy_phase_solver.py",
             "lazy_pair_oracle.py", "separable_support.py", "experiment_support_bundle.py",
             "breakout_geometry.py", "_breakout_geometry.cpp", "experiment_tied_batch.py",
             "experiment_batch_lattice.py", "_batch_lattice.cpp"]
    for name in names:
        source = ROOT / "scripts" / name
        shutil.copy2(source, folder / name)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    frozen = output / "inputs"
    frozen.mkdir()
    for index, path in enumerate(inputs):
        shutil.copy2(path, frozen / f"origin-{index}.json")
    return hashes


def gate(origins, output, seconds=60):
    output.mkdir()
    started = time.monotonic()
    deadline = started + seconds
    records, hashes = [], set()
    control = None
    first = origins[0].evaluate(origins[0].anchor)
    baseline = checked(origins[0].raw)
    if first is not None and baseline.valid:
        control = solve_phase(first["domains"], np.asarray([d[0] for d in first["domains"]]),
                              float(baseline.score) + 1e-11, output / "control",
                              seconds=min(10., max(.01, deadline - time.monotonic())),
                              local_only=True, seed=3112400)
    for index, origin in enumerate(origins):
        for axis in range(3):
            for sign in [-1, 1]:
                if time.monotonic() >= deadline:
                    break
                direction = np.zeros(3)
                direction[axis] = sign
                event = locate_transition(origin.evaluate, origin.anchor, direction, deadline=deadline)
                if event is None:
                    records.append({"origin": index, "axis": axis, "sign": sign, "status": "NO_BRACKET"})
                    continue
                domains = merge_domains(event["before"]["domains"], event["after"]["domains"])
                key = domain_digest(domains)
                hashes.add(key)
                records.append({"origin": index, "axis": axis, "sign": sign,
                                "status": "BRACKETED", "low": event["low"], "high": event["high"],
                                "domain_sha256": key, "domain_sizes": list(map(len, domains)),
                                "evaluations": event["evaluations"]})
    result = {"seconds": time.monotonic() - started, "budget_seconds": seconds,
              "control": control, "transitions": records, "distinct_domains": len(hashes),
              "go": bool(control and control.get("valid") and len(hashes) >= 9 and time.monotonic() <= deadline),
              "criteria": "Reference-valid incumbent recovered; at least nine distinct endpoint-union domains in 60 seconds"}
    write_json(output / "report.json", result)
    print(json.dumps({"event": "gate", "go": result["go"], "distinct_domains": len(hashes),
                      "seconds": result["seconds"]}), flush=True)
    return result


def run_arm(origins, output, *, method, seconds=180, max_models=200, per_model=1.5,
            target=LEADER - 5e-12, seed=3112401):
    if method not in ("event", "ordinary"):
        raise ValueError("Unknown method")
    output.mkdir()
    started = time.monotonic()
    deadline = started + seconds
    rng = np.random.default_rng(seed)
    queue, queued_cells, expanded_cells, tested_domains = [], set(), set(), set()
    records, valid, frontier, serial = [], [], None, 0
    best = None
    solver_spent = 0.
    decoder = NativeBabai()
    seen_raw = set()
    frontier_serial = 0

    def enqueue(index, state, domains=None, bracket=None, choice="adjacent_event"):
        nonlocal serial
        if state is None:
            return
        marker = (index, state["hash"])
        if marker in queued_cells:
            return
        queued_cells.add(marker)
        preview = native_preview(state, origins[index], target, seed, decoder)
        score = preview["metric"]
        item = {"origin": index, "state": state,
                "domains": domains if domains is not None else state["domains"],
                "bracket": bracket, "frontier": score, "preview": preview,
                "choice": choice}
        heapq.heappush(queue, (score["loss"], serial, item))
        serial += 1
        for raw in preview["eligible"]:
            register(raw, index, len(records), "queued_native_preview")

    def register(raw, origin_index, model_number, source):
        nonlocal best
        raw_key = hashlib.sha256(np.rint(raw * 1e6).astype("<i8").tobytes()).hexdigest()
        if raw_key in seen_raw:
            return
        seen_raw.add(raw_key)
        check = checked(raw)
        if not check.valid:
            return
        record = {"score": float(check.score), "seconds": time.monotonic() - started,
                  "origin": origin_index, "model": model_number, "source": source,
                  "center_hash": center_hash(raw), "late": time.monotonic() > deadline}
        path = output / f"valid-{len(valid):04d}.json"
        save(path, raw)
        record["file"] = path.name
        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        valid.append(record)
        if best is None or check.score < best:
            best = float(check.score)
            save(output / "best.json", raw)
            print(json.dumps({"event": "valid", "method": method, **record}), flush=True)

    def report():
        return {"method": method, "seed": seed, "budget_seconds": seconds,
                "seconds": time.monotonic() - started, "target": target,
                "max_models": max_models, "per_model_seconds": per_model,
                "models": records, "distinct_domains": len(tested_domains),
                "queued_cells": len(queued_cells), "expanded_cells": len(expanded_cells),
                "solver_seconds": solver_spent, "best_valid_score": best,
                "valid_records": valid, "frontier": frontier,
                "policy": "Round-robin origins; event arm uses 90% adjacent cells and 10% fresh restarts; 256 native previews per frame",
                "scope": "Production local search with known starts; approximate event traversal, not blind evaluation or complete coverage"}

    for index, origin in enumerate(origins):
        enqueue(index, origin.evaluate(origin.anchor), choice="initial_origin")

    while time.monotonic() < deadline and len(records) < max_models:
        random_step = (method == "ordinary" and len(records) >= len(origins)) or (method == "event" and len(records) % 10 == 9)
        if random_step:
            index = len(records) % len(origins)
            origin = origins[index]
            parameters = origin.anchor + rng.normal(0, 400., 3)
            if len(records) % 4 == 0:
                parameters[2] = rng.uniform(-np.pi, np.pi) * 1e6
            state = origin.evaluate(parameters)
            if state is None:
                continue
            preview = native_preview(state, origin, target, seed, decoder)
            item = {"origin": index, "state": state, "domains": state["domains"],
                    "bracket": None, "frontier": preview["metric"], "preview": preview,
                    "choice": "random_restart" if method == "event" else "ordinary_random"}
        else:
            if not queue:
                break
            _, _, item = pop_balanced(queue, len(records) % len(origins))
        index, state, domains = item["origin"], item["state"], item["domains"]
        item["preview"]["inside_domain"] = all(
            np.any(np.all(np.rint(domain * 1e6).astype(np.int64) ==
                          np.rint(row * 1e6).astype(np.int64), axis=1))
            for domain, row in zip(domains, item["preview"]["scored"]))
        key = domain_digest(domains)
        if frontier is None or item["frontier"]["loss"] < frontier["loss"] - 2e-11:
            raw = item["preview"]["raw"]
            exact = exact_frontier(raw, origins[index].model.c.scale)
            frontier = {**item["frontier"], "origin": index, "parameters": state["parameters"],
                        "seconds": time.monotonic() - started, "source": item["preview"]["source"],
                        "inside_domain": item["preview"]["inside_domain"], "exact": exact}
            save(output / "frontier-raw.json", raw)
            saved_frontier = output / f"frontier-{frontier_serial:04d}.json"
            save(saved_frontier, raw)
            frontier_serial += 1
            frontier["file"] = saved_frontier.name
            frontier["sha256"] = hashlib.sha256(saved_frontier.read_bytes()).hexdigest()
        if key not in tested_domains and time.monotonic() < deadline:
            tested_domains.add(key)
            number = len(records)
            model_path = output / f"model-{number:03d}"
            base = np.asarray([d[0] for d in domains])
            before = time.monotonic()
            result = solve_phase(domains, base, target, model_path,
                                 seconds=min(per_model, max(.01, deadline - time.monotonic())),
                                 seed=seed + number, local_only=True)
            solver_spent += time.monotonic() - before
            record = {"number": number, "origin": index, "parameters": state["parameters"],
                      "domain_sha256": key, "domain_sizes": list(map(len, domains)),
                      "frontier": item["frontier"], "bracket": item["bracket"], "result": result,
                      "choice": item["choice"], "preview_source": item["preview"]["source"],
                      "preview_inside_domain": item["preview"]["inside_domain"]}
            records.append(record)
            if result.get("valid"):
                register(load(model_path / "solution.json"), index, number, "target_solver")
            # Non-target pair-feasible leaves can still improve the incumbent.
            events = model_path / "events.jsonl"
            if events.exists():
                for line in events.read_text().splitlines():
                    leaf = json.loads(line)
                    if leaf["event"] != "exact_pair_feasible":
                        continue
                    raw, _ = submission_for_geometry(np.asarray([d[v] for d, v in zip(domains, leaf["assignment"])]))
                    register(raw, index, number, "pair_feasible_leaf")
            for raw in item["preview"]["eligible"]:
                register(raw, index, number, "native_preview_batch")
            write_json(output / "report.json", report())
            if number % 10 == 0:
                print(json.dumps({"event": "progress", "method": method, "models": len(records),
                                  "seconds": time.monotonic() - started, "best": best,
                                  "frontier": frontier}), flush=True)
            if best is not None and best < LEADER:
                break
        marker = (index, state["hash"])
        if method == "event" and marker not in expanded_cells:
            expanded_cells.add(marker)
            for axis in range(3):
                for sign in [-1, 1]:
                    if time.monotonic() >= deadline:
                        break
                    direction = np.zeros(3)
                    direction[axis] = sign
                    event = locate_transition(origins[index].evaluate, state["parameters"], direction, deadline=deadline)
                    if event is None:
                        continue
                    merged = merge_domains(event["before"]["domains"], event["after"]["domains"])
                    bracket = {k: event[k] for k in ("low", "high", "direction", "evaluations", "scope")}
                    bracket.update(before_hash=event["before"]["hash"], after_hash=event["after"]["hash"])
                    enqueue(index, event["after"], merged, bracket)
    final = report()
    final["status"] = "MODEL_CAP" if len(records) >= max_models else "BUDGET_OR_QUEUE_END"
    write_json(output / "report.json", final)
    print(json.dumps({"event": "done", "method": method, "models": len(records),
                      "seconds": final["seconds"], "best": best, "frontier": frontier}), flush=True)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--max-models", type=int, default=200)
    parser.add_argument("--per-model", type=float, default=1.5)
    parser.add_argument("--angle-width", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3112401)
    parser.add_argument("--target", type=float, default=LEADER - 5e-12)
    parser.add_argument("--gate-only", action="store_true")
    args = parser.parse_args()
    if args.seconds <= 0 or args.max_models <= 0 or args.per_model <= 0:
        parser.error("Positive time and model limits required")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = args.input or DEFAULT_INPUTS
    hashes = source_snapshot(args.output, inputs)
    started = time.monotonic()
    seed = load(ROOT / "tests/fixtures/recovery/mirror-continuous.json")
    warmup(seed)
    origins = [Origin(path, seed, args.angle_width, raw_input=index == 0)
               for index, path in enumerate(inputs)]
    write_json(args.output / "setup.json", {"origins": [o.describe() for o in origins],
               "source_sha256": hashes, "setup_seconds": time.monotonic() - started,
               "parameters": vars(args) | {"output": str(args.output)},
               "scope": "Known production starts; no API, submission or global promotion"})
    construction = gate(origins, args.output / "gate")
    if not construction["go"] or args.gate_only:
        return
    comparison = {"gate": construction, "arms": {}}
    for method in ("event", "ordinary"):
        # Gate caches do not give either timed arm a construction advantage.
        for origin in origins:
            origin.cache.clear()
        result = run_arm(origins, args.output / method, method=method,
                         seconds=args.seconds, max_models=args.max_models,
                         per_model=args.per_model, target=args.target, seed=args.seed)
        comparison["arms"][method] = result
        write_json(args.output / "comparison.json", comparison)
        if result["best_valid_score"] is not None and result["best_valid_score"] < LEADER:
            break


if __name__ == "__main__":
    main()
