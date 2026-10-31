#!/usr/bin/env python3
"""Fast packing experiments. Run `python search.py --help` for recipes."""
from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys

# Accelerate's default thread pool makes these tiny optimization problems slow.
for variable in ('VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(variable, '1')

ROOT = Path(__file__).resolve().parent
RECIPES = {
    'polish': ROOT / 'src/semicircle_packing/polish.py',
    'lattice': ROOT / 'scripts/lattice_polish.py',
    'rounding': ROOT / 'scripts/rounding_search.py',
    'basins': ROOT / 'scripts/experiment_basins.py',
    'compression': ROOT / 'scripts/experiment_compression.py',
    'symmetry': ROOT / 'scripts/experiment_symmetry.py',
    'continuation': ROOT / 'scripts/experiment_continuation.py',
    'feasibility': ROOT / 'scripts/experiment_frames.py',
    'lll': ROOT / 'scripts/experiment_lll.py',
    'chase': ROOT / 'scripts/experiment_lll.py',
    'arena': ROOT / 'scripts/arena.py',
}


def benchmark():
    import json
    import time
    import numpy as np
    from semicircle_packing.exact_area import intersection_areas_and_jacobian, polygon_intersection_areas_and_jacobian
    from semicircle_packing.fast_mec import compute_mec_fast
    from semicircle_packing.fast_geometry import FastGeometry
    from semicircle_packing.polish import checked, load, polish

    p = load('solution.json')
    geometry = FastGeometry()
    z = np.r_[p.ravel(), 3.]
    # Build native kernels before timing; no optimizer or server in this loop.
    geometry.evaluate(z)
    intersection_areas_and_jacobian(p)
    timings = {}
    def sat():
        z[0] += 1e-12
        return geometry.evaluate(z)

    for name, function, iterations in [
        ('sat_and_jacobian_us', sat, 10000),
        ('areas_and_jacobian_us', lambda: intersection_areas_and_jacobian(p), 10000),
        ('polygon_areas_and_jacobian_us', lambda: polygon_intersection_areas_and_jacobian(p), 10000),
        ('enclosing_circle_us',lambda: compute_mec_fast(p),100),
        ('reference_validation_us', lambda: checked(p), 3),
    ]:
        start = time.perf_counter()
        for _ in range(iterations):
            function()
        timings[name] = 1e6*(time.perf_counter()-start)/iterations
    start = time.perf_counter()
    result = polish(p)
    timings['continuous_polish_seconds'] = time.perf_counter()-start
    timings['continuous_radius'] = float(result.x[-1])
    timings['continuous_feasible'] = result.max_violation < 1e-9
    timings['rounded_valid'] = checked(result.x[:-1].reshape(-1, 3)).valid
    print(json.dumps(timings, indent=2))


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else '--help'
    if command in ('-h', '--help', 'help'):
        print(__doc__)
        print('''
  uv run python search.py benchmark
  uv run python search.py polish solution.json
  uv run python search.py lattice --input solution.json --steps 5 --seconds 10 --rounds 5
  uv run python search.py basins --seconds 60 --seed 11
  uv run --extra lattice python search.py chase --seconds 60 --seed 992
  uv run python search.py arena leaderboard
  uv run python search.py arena check CANDIDATE.json --save experiments/check.json

Additional recipes: lll, feasibility, rounding, compression, symmetry, continuation.
Pass --help after a recipe for its parameters. Search output is experimental;
use run.py and arena check to validate any candidate before promotion.
''')
        return
    os.chdir(ROOT)
    if command == 'benchmark':
        benchmark()
        return
    if command not in RECIPES:
        raise SystemExit(f'Unknown recipe: {command}. Use --help.')
    sys.path.insert(0, str(ROOT / 'scripts'))
    preset = []
    if command == 'chase':
        preset = ['--input', 'experiments/branch-3.141616-3.141592.json',
                  '--keep-seed', '--movement', '50', '--block-size', '20',
                  '--target', '2.94852698', '--goal', '2.9485269238',
                  '--concentration', '10', '--boundary-bias', '5', '--seconds', '60',
                  '--seed', '500305', '--incumbent', 'solution.json',
                  '--checkpoint-prefix', 'chase',
                  '--output', 'experiments/challenger.json']
    sys.argv = [str(RECIPES[command]), *preset, *sys.argv[2:]]
    if command == 'polish':
        runpy.run_module('semicircle_packing.polish', run_name='__main__')
    else:
        runpy.run_path(str(RECIPES[command]), run_name='__main__')


if __name__ == '__main__':
    main()
