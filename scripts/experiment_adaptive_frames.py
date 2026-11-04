"""Offline common-random-number search over physical, rotation and grid phases.

Each frame decodes the same seeded native proposal batch. Geometry is normalized
to a fixed integer translation and scored with one frozen pair-scale vector, so
the same scored packing has the same numerical fitness in every source frame.
The adaptive arm changes outer frame parameters; it does not train slack targets.
"""
from __future__ import annotations

import os
for _name in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmark_recovery import scored_once, transform, write_json
from blind_packing_worker import warmup
from breakout_geometry import evaluate_batch
from experiment_batch_lattice import NativeBabai
from experiment_mirror_family import FLATS
from experiment_rounding_events import Origin, eligible_native_indices, exact_frontier
from experiment_tied_batch import Frame
from semicircle_packing.polish import AreaConstraints, checked, load, save

ROOT = Path(__file__).resolve().parents[1]
LEADER = 2.9485269238856113


def normalized_geometry(units):
    """Exact integer common-translation gauge; all angles remain in the key."""
    result = np.asarray(units, dtype=np.int64).reshape(-1, 15, 3).copy()
    result[:, :, :2] -= result[:, :1, :2].copy()
    return result, result.astype(float) / 1e6


def pose_key(units, centers_only=False):
    p = np.asarray(units, dtype=np.int64).reshape(15, 3)
    data = p[:, :2] if centers_only else p
    return hashlib.sha256(np.asarray(data, dtype="<i8").tobytes()).hexdigest()


def fitness_values(areas, circles, scales, target):
    deficits = np.maximum(np.asarray(areas) - 1e-6, 0.) / np.asarray(scales)
    radius = np.maximum(np.asarray(circles)[:, 2] - target, 0.)
    return np.maximum(deficits.max(axis=1), radius), deficits.sum(axis=1) + radius


def wrap_phases(parameters):
    result = np.asarray(parameters, dtype=float).copy()
    result[:3] = np.clip(result[:3], -3., 3.)
    result[3:] = (result[3:] + .5) % 1. - .5
    return result


def diversity_weights(population):
    """Repeated centers lose parent-sampling weight, never candidate eligibility."""
    counts = {}
    for item in population:
        key = item.get("center_key")
        counts[key] = counts.get(key, 0) + 1
    weights = np.array([1. / counts[item.get("center_key")] for item in population])
    return weights / weights.sum()


class Evaluator:
    def __init__(self, origins, *, target, batch, seed, scales):
        self.origins = origins
        self.target, self.batch, self.seed = target, batch, seed
        self.scales = np.asarray(scales).copy()
        self.decoder = NativeBabai()
        self.pose_cache = {}
        self.evaluations = self.proposals = self.rejected = 0

    def evaluate(self, origin_index, parameters):
        self.evaluations += 1
        origin = self.origins[origin_index]
        parameters = wrap_phases(parameters)
        physical_parameters = origin.anchor + np.r_[parameters[:2] * 400., parameters[2] * 400.]
        section, meta = origin.model.section(origin.branch.tolist(), physical_parameters[:2] * 1e-6)
        if max(meta["violation"], meta["residual"]) > 2e-9:
            self.rejected += 1
            return None
        # Manifold.section fixes the enclosing center at zero. Do not add the
        # fitted common translation before Frame's zero-centered constraints.
        physical = transform(section, physical_parameters[2] * 1e-6, [0., 0.])
        phase = ((origin.translation * 1e6 + parameters[3:] + .5) % 1.) - .5
        settings = SimpleNamespace(target=self.target, regularization=1e-5,
                                   block_size=20, concentration=1., boundary_bias=1., movement=10.)
        try:
            frame = Frame(physical, meta["radius"], FLATS, settings)
        except (ValueError, np.linalg.LinAlgError):
            self.rejected += 1
            return None
        if frame.budget <= 0:
            self.rejected += 1
            return None
        anchor = frame.anchor_units.copy()
        changed = np.rint(physical * 1e6).astype(np.int64)
        changed[:, :2] = np.rint(physical[:, :2] * 1e6 + phase).astype(np.int64)
        for (i, j), difference in zip(FLATS, frame.differences):
            changed[i, 2] = changed[j, 2] + difference
        _, jacobian = frame.c.evaluate(np.r_[physical.ravel(), meta["radius"]])
        frame.b0 -= jacobian[frame.active, :45] @ (changed - anchor).ravel()
        frame.anchor_units = changed
        # Restarting exactly this RNG is deliberate common-random-number coupling.
        actual, units, coefficients, _, _ = frame.sample(
            np.random.default_rng(self.seed), self.batch, settings, self.decoder)
        units, actual = normalized_geometry(units)
        self.proposals += len(actual)
        keys = [pose_key(row) for row in units]
        fresh = list(dict.fromkeys(i for i, key in enumerate(keys) if key not in self.pose_cache))
        if fresh:
            areas, circles, _ = evaluate_batch(actual[fresh], centers=[0., 0.])
            primary, secondary = fitness_values(areas, circles, self.scales, self.target)
            for offset, index in enumerate(fresh):
                self.pose_cache[keys[index]] = {"primary": float(primary[offset]),
                    "secondary": float(secondary[offset]), "radius": float(circles[offset, 2]),
                    "overlap": float(np.max(np.maximum(areas[offset] - 1e-6, 0.) / self.scales)),
                    "eligible": bool(areas[offset].max() <= 1e-6 + 1e-11 and circles[offset, 2] <= self.target + 1e-7)}
        metrics = [self.pose_cache[key] for key in keys]
        best_index = min(range(len(metrics)), key=lambda i: (metrics[i]["primary"], metrics[i]["secondary"]))

        def raw_at(index):
            raw = frame.raw(coefficients[index])
            raw[:, :2] = units[index, :, :2].astype(float) / 1e6
            return raw

        eligible = []
        seen = set()
        for index, metric in enumerate(metrics):
            if metric["eligible"] and keys[index] not in seen:
                seen.add(keys[index])
                eligible.append((keys[index], raw_at(index), metric))
        best = metrics[best_index]
        return {"origin": origin_index, "parameters": parameters.tolist(),
                "physical_parameters": physical_parameters.tolist(), "phase": phase.tolist(),
                "fitness": [best["primary"], best["secondary"]],
                "radius": best["radius"], "overlap": best["overlap"],
                "pose_key": keys[best_index], "center_key": pose_key(units[best_index], True),
                "raw": raw_at(best_index), "eligible": eligible,
                "continuous_radius": meta["radius"], "new_poses": len(fresh)}


def run_arm(origins, output, *, method, seconds=240, population_size=12,
            batch=512, seed=3112501, target=LEADER-5e-12, scales):
    output.mkdir()
    evaluator = Evaluator(origins, target=target, batch=batch, seed=seed, scales=scales)
    rng = np.random.default_rng(seed + 100)
    initial_rng = np.random.default_rng(seed + 101)
    initial = [[np.zeros(5)] + [wrap_phases(np.r_[initial_rng.normal(0, .8, 3),
                            initial_rng.uniform(-.5, .5, 2)]) for _ in range(population_size-1)]
               for _ in origins]
    populations = [[] for _ in origins]
    cursor = [0] * len(origins)
    valid_seen, valid, pareto, frame_records = set(), [], [], []
    best = None
    started = time.monotonic()
    deadline = started + seconds
    evaluations = 0

    def register(key, raw, candidate, provenance):
        nonlocal best
        if key in valid_seen:
            return
        valid_seen.add(key)
        validation = checked(raw)
        if not validation.valid:
            return
        name = f"valid-{len(valid):04d}.json"
        save(output / name, raw)
        record = {"score": float(validation.score), "file": name, "pose_key": key,
                  "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest(),
                  "seconds": time.monotonic()-started, "late": time.monotonic()>deadline,
                  "frame": provenance, "native": candidate}
        valid.append(record)
        if best is None or validation.score < best:
            best = float(validation.score)
            save(output / "best.json", raw)
            print(json.dumps({"event": "valid", "method": method, **record}), flush=True)

    def checkpoint():
        result = {"method": method, "seconds": time.monotonic()-started, "budget_seconds": seconds,
                  "seed": seed, "batch": batch, "population_size_per_origin": population_size,
                  "target": target, "best_valid_score": best, "valid_records": valid,
                  "pareto": pareto, "frames": frame_records,
                  "evaluations": evaluator.evaluations, "proposals": evaluator.proposals,
                  "rejected_frames": evaluator.rejected, "unique_pose_fitnesses": len(evaluator.pose_cache),
                  "scope": "Production outer-frame comparison; fixed random batches and fixed geometric scales; no API or global promotion"}
        write_json(output / "report.json", result)
        return result

    while time.monotonic() < deadline:
        index = evaluations % len(origins)
        population = populations[index]
        slot = cursor[index] % population_size
        if len(population) < population_size:
            parameters = initial[index][len(population)]
            choice = "shared_initial"
        elif method == "random":
            parameters = wrap_phases(np.r_[rng.normal(0, .8, 3), rng.uniform(-.5, .5, 2)])
            choice = "fresh_random"
        else:
            probabilities = diversity_weights(population)
            selected = rng.choice(len(population), size=3, replace=False, p=probabilities)
            a, b, c = (np.asarray(population[k]["parameters"]) for k in selected)
            donor = wrap_phases(a + .7 * (b - c))
            mask = rng.random(5) < .8
            mask[int(rng.integers(0, 5))] = True
            parameters = wrap_phases(np.where(mask, donor, population[slot]["parameters"]))
            choice = "differential_evolution"
        proposal = evaluator.evaluate(index, parameters)
        evaluations += 1
        if proposal is None:
            # Do not get stuck retrying the same rejected initialization.
            if len(population) < population_size:
                initial[index][len(population)] = wrap_phases(np.r_[rng.normal(0, .8, 3), rng.uniform(-.5, .5, 2)])
            continue
        metadata = {k: v for k, v in proposal.items() if k not in ("raw", "eligible")}
        metadata.update(number=len(frame_records), choice=choice, seconds=time.monotonic()-started)
        frame_records.append(metadata)
        for key, raw, candidate in proposal["eligible"]:
            register(key, raw, candidate, metadata["number"])
        if best is not None and best < LEADER:
            break
        item = {k: proposal[k] for k in ("parameters", "fitness", "pose_key", "center_key")}
        if len(population) < population_size:
            population.append(item)
        elif method == "adaptive" and tuple(item["fitness"]) <= tuple(population[slot]["fitness"]):
            population[slot] = item
        cursor[index] += 1
        # Reference-check the full native Pareto witness, not its source frame.
        native_radius, native_overlap = proposal["radius"], proposal["overlap"]
        dominated = any(p["radius"] < native_radius - 1e-10 and p["overlap"] < native_overlap - 1e-10 for p in pareto)
        already = any(p["pose_key"] == proposal["pose_key"] for p in pareto)
        if not dominated and not already and native_radius < target + 1e-7 and native_overlap < 1e-7:
            exact = exact_frontier(proposal["raw"], scales)
            radius = exact["reference_mec"][2]
            overlap = exact["maximum_exact_normalized_overlap_deficit"]
            if not any(p["radius"] <= radius and p["overlap"] <= overlap for p in pareto):
                pareto[:] = [p for p in pareto if not (radius <= p["radius"] and overlap <= p["overlap"])]
                name = f"frontier-{metadata['number']:05d}.json"
                save(output / name, proposal["raw"])
                pareto.append({"radius": radius, "overlap": overlap, "exact": exact,
                               "pose_key": proposal["pose_key"], "file": name,
                               "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest(),
                               "frame": metadata["number"], "seconds": time.monotonic()-started})
                if len(pareto) > 128:
                    pareto.sort(key=lambda p: max(p["overlap"], p["radius"]-target))
                    del pareto[128:]
        if len(frame_records) % 100 == 0:
            result = checkpoint()
            print(json.dumps({"event": "progress", "method": method, "seconds": result["seconds"],
                              "frames": len(frame_records), "best": best, "pareto": len(pareto),
                              "unique_poses": result["unique_pose_fitnesses"]}), flush=True)
    result = checkpoint()
    print(json.dumps({"event": "done", "method": method, "frames": len(frame_records),
                      "seconds": result["seconds"], "best": best, "pareto": len(pareto)}), flush=True)
    return result


def incumbent_control(origin, scales, output, seed):
    validation = checked(origin.raw)
    if not validation.valid:
        raise ValueError("First production input must be a valid incumbent")
    started = time.monotonic()
    evaluator = Evaluator([origin], target=float(validation.score)+1e-11,
                          batch=1024, seed=seed, scales=scales)
    proposal = evaluator.evaluate(0, np.zeros(5))
    accepted = []
    if proposal is not None:
        for key, raw, _ in proposal["eligible"]:
            result = checked(raw)
            if result.valid:
                accepted.append(float(result.score))
                if result.score <= validation.score+1e-11:
                    save(output/'control-recovered.json', raw)
    control = {"seconds": time.monotonic()-started, "expected": float(validation.score),
               "batch": 1024, "seed": seed, "eligible": 0 if proposal is None else len(proposal["eligible"]),
               "valid_scores": accepted, "pass": bool(accepted and min(accepted) <= validation.score+1e-11)}
    write_json(output/'control.json', control)
    if not control["pass"]:
        raise RuntimeError("Native incumbent recovery control failed; do not launch optimizer")
    print(json.dumps({"event": "control", **control}), flush=True)
    return control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=240)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--seed", type=int, default=3112501)
    args = parser.parse_args()
    if args.seconds <= 0 or args.batch <= 0 or args.population < 4:
        parser.error("Positive budgets and population >=4 required")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "inputs").mkdir()
    (args.output / "source-snapshot").mkdir()
    hashes = {}
    for path in [Path(__file__), ROOT/'scripts/experiment_rounding_events.py', ROOT/'scripts/experiment_tied_batch.py',
                 ROOT/'scripts/experiment_mirror_family.py', ROOT/'scripts/experiment_batch_lattice.py',
                 ROOT/'scripts/_batch_lattice.cpp', ROOT/'scripts/_breakout_geometry.cpp']:
        shutil.copy2(path, args.output/'source-snapshot'/path.name)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    seed = load(ROOT/'tests/fixtures/recovery/mirror-continuous.json')
    warmup(seed)
    frozen_inputs = []
    for index, path in enumerate(args.input):
        frozen = args.output/'inputs'/f'origin-{index}.json'
        shutil.copy2(path, frozen)
        frozen_inputs.append(frozen)
    origins = [Origin(path, seed, raw_input=True) for path in frozen_inputs]
    scales = AreaConstraints(scored_once(origins[0].raw)).scale.copy()
    write_json(args.output/'setup.json', {"origins": [origin.describe() for origin in origins],
               "fixed_pair_scales": scales.tolist(), "source_sha256": hashes,
               "parameters": vars(args) | {"output": str(args.output)},
               "notes": ["Native Frame is built at zero enclosure center; fitted translation is retained only as a fractional grid phase.",
                         "Random seed resets for every batch; pose fitness uses a fixed integer gauge and fixed scales."]})
    incumbent_control(origins[0], scales, args.output, args.seed)
    summary = {}
    for method in ["adaptive", "random"]:
        summary[method] = run_arm(origins, args.output/method, method=method,
                seconds=args.seconds, population_size=args.population, batch=args.batch,
                seed=args.seed, scales=scales)
        write_json(args.output/'comparison.json', summary)
        if summary[method]["best_valid_score"] is not None and summary[method]["best_valid_score"] < LEADER:
            break


if __name__ == "__main__":
    main()
