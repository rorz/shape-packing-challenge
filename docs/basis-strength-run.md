# Matched BKZ basis strength comparison

The deadline campaign audit found no saved JSON `block_size` 30/40 or Markdown
BKZ-30/40 experiments. Existing native runs use block 20. This experiment changes
the lattice reduction strength while preserving each physical target exactly.

The current packing is fitted to the two-mode continuous family. Three initial
frames use that fit, a +0.0002 first-mode/+0.00017-radian perturbation, and a
−0.0002 second-mode/−0.00023-radian perturbation. Each physical packing is
recentered at its actual enclosing center before model construction. Subgrid
translation enters through the anchor and matching `b0` update, never by moving
the physical constraint origin. The worker rejects an uncentered frame.

Each frame's full integer ambient-target matrix is generated once and saved.
BKZ blocks 20, 30 and 40 decode those same targets. Both physical arrays and
ambient matrices are hashed. Every basis is constructed in a separate process
with an eight-second hard timeout, including imports and setup. Each reduction
retains the existing two-loop limit. Fixed pair normalization from the saved
incumbent is used for merit comparisons across every frame and arm.

The first 128 decodes per basis are checked against fpylll; ambiguous native
decodes are patched by fpylll. Whole scored poses are deduplicated only modulo
common integer translation. Every candidate passing the permissive native area
screen and radius cap (incumbent +100 billionths) receives reference validation;
the eight best native violations per frame receive independent GEOS diagnostics.
No API or global promotion is performed.

```bash
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/experiment_basis_strength.py \
  --input solution.json \
  --output experiments/pantheon-pilot/deadline-basis-smoke-v1 \
  --frames 3 --proposals 4096 --build-timeout 8 --seconds-per-arm 60 --seed 3111901
```

Three tests passed in 0.50 seconds: identical ambient targets under different
reductions, correct translation-only pose identity, and rejection of an
uncentered physical basis frame. Timings are reported separately for preparation,
basis construction, child-process overhead, decoding, geometry and total
evaluation (including validation).

## Smoke result: stop this schedule

The complete three-frame comparison took **11.86 seconds**. BKZ blocks 20,
30 and 40 produced **bit-identical reduced bases and move matrices on every
frame**. Consequently the native distinct-pose counts, GEOS merits and valid
results matched. Each basis's 128 native/fpylll decode comparisons passed.
Basis construction itself cost only 0.011–0.015 seconds per frame; no build timed
out. No better solution was found.

This fails the viability gate for a longer BKZ-only campaign, so none was run.
It does not establish that every packing frame or different reduction strategy
would produce the same basis. [The compact result](basis-strength-results.json)
contains source, physical-target and ambient-target hashes, matched merits,
timings and the exact reduced-basis identity audit. Full arrays remain in the
ignored immutable run directory.
