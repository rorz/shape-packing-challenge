"""Find a target score on the submission grid across rigid coordinate frames."""
from __future__ import annotations
import os
for key in ('VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse
import json
from pathlib import Path
import time
import numpy as np
from lattice_polish import flat_contact_cuts, solve
from rounding_search import area_model, write_solution
from semicircle_packing.polish import checked, load, polish
from semicircle_packing.geometry import canonicalize_theta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='solution.json')
    parser.add_argument('--output', type=Path, default=Path('experiments/feasibility_best.json'))
    parser.add_argument('--target', type=float, default=2.9485269238)
    parser.add_argument('--seconds', type=float, default=90)
    parser.add_argument('--solve-seconds', type=float, default=1)
    parser.add_argument('--steps', type=int, default=15)
    parser.add_argument('--seed', type=int, default=91)
    parser.add_argument('--flat-cuts', action='store_true')
    args = parser.parse_args()
    initial = polish(load(args.input))
    original = initial.x[:-1].reshape(-1,3)
    rng = np.random.default_rng(args.seed)
    started = time.monotonic()
    phase = 0
    while time.monotonic() - started < args.seconds:
        angle = rng.uniform(-np.pi, np.pi)
        cs, sn = np.cos(angle), np.sin(angle)
        rotation = np.array([[cs, -sn], [sn, cs]])
        base = original.copy()
        base[:, :2] = base[:, :2] @ rotation.T + rng.uniform(-.5e-6,.5e-6,2)
        base[:, 2] += angle
        base[:, 2] = [canonicalize_theta(t) for t in base[:, 2]]
        base = np.round(base, 6)
        cuts = flat_contact_cuts(base,args.steps,1e-12) if args.flat_cuts else []
        for attempt in range(8):
            remaining = args.seconds - (time.monotonic() - started)
            if remaining <= 0:
                return
            candidate, info = solve(base, args.steps, min(remaining,args.solve_seconds),
                                    margin=1e-12, cuts=cuts, target=args.target)
            if candidate is None:
                break
            print(json.dumps(dict(phase=phase, attempt=attempt,
                seconds=time.monotonic()-started, **info)), flush=True)
            if info['valid'] and info['score'] < args.target:
                validation = checked(candidate)
                if validation.valid and validation.score < args.target:
                    write_solution(args.output, candidate)
                    return
            _, areas, jac = area_model(candidate)
            active = areas > 1e-6-1e-12
            if not active.any():
                break
            cuts.append((candidate, areas[active], jac[active]))
        phase += 1
        if phase % 10 == 0:
            print(json.dumps(dict(phase=phase, seconds=time.monotonic()-started)), flush=True)


if __name__ == '__main__':
    main()
