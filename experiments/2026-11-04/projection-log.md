# Residual-feedback projection experiment

## Hypothesis — 2026-11-04

Independent targets forget how each nearest lattice point failed. Test a
Douglas–Rachford iteration between the integer contact embedding and the
halfspaces for feasible contact clearances. Retain the residual vector between
iterations; a sequence can therefore cross local nearest-point boundaries.

Implementation: `scripts/experiment_difference_map.py`. Integer translations
are removed by fixing piece 0's x/y, leaving 43 integer variables. The enclosing
center remains continuous and is projected analytically. Babai approximates the
lattice projection; contact clipping is exact for the local linear model.
Nonlinear polygon geometry and reference validation decide acceptance. No claim
of convergence or global exclusion is made.

First run uses seed 110411, beta 0.8, 1000 iterations per random frame, movement
100 microunits, BKZ block 20, 120 seconds. Full command, source hashes, stdout,
duration, and exit status are recorded by `research_run.py` in `runs/`.

## Subsequent bounded runs and model changes

- First fixed-tangent run: 829,055 lattice projections, 797,900 geometry checks,
  830 frames in 120 seconds. Best normalized violation 26.1979e-9; no valid
  improvement. This is progress in proposal diversity, not a winning packing.
- New first-flat branches: 27.35 and 32.35 microradians, beta 1.2 and 0.7,
  respectively; 180 seconds each, seeds 110412/110413. Use the exact command
  records in `runs/` for all other flags.
- Nonlinear offset correction: at every decoded lattice point, replace each
  linear contact offset with the actual gap minus its linear contribution.
  This follows curvature while retaining the lattice basis; seed 110414,
  beta 0.8, 180 seconds. Added as `--nonlinear`, not a silent behavior change.

Independent review checked center elimination and projection algebra numerically:
center rank 2, orthogonality error 7.1e-17, projector idempotence error 4.5e-16.

## Curved-family lattice search

Hypothesis: the previous 50-microunit integer neighborhoods underexplore the two
physical soft motions left after fixing both opposed-flat angles and global
rotation. Existing old Newton-family work sampled rounding there but did not
perform reduced-lattice decoding around each exact continuous section.

`experiment_family_lattice.py` extracts those two directions, fixes their
generalized displacement, and re-solves all continuous nonlinear constraints.
Then it invokes the known LLL decoder around each distinct feasible section.
This combines a changed search region with the proven integer decoder.

Initial campaign: 12 sections on radii 0.0005/0.001 in the orthonormal soft basis,
20 seconds per section, seed 110420, target 2.948526945. Full precision seeds,
child commands, child logs, and actual continuous violations are retained in
`family/`; parent command and source hashes in `runs/`.

## Recording correction

Review found hashes alone do not reproduce concurrently edited sources.
`research_run.py` now preserves content-addressed source snapshots, records
installed package versions and platform, hashes standard default input files,
and reports files changed during the run. Earlier run logs remain intact and
their recorded hashes must not be mistaken for preserved historical source.

## External research checked

Searched for published 15-semicircle solutions. The challenge's
[official rules](https://www.optimizationarena.com/packing/about) confirm six
decimal places and overlap-area tolerance. A public
[March 2026 account](https://buttondown.com/TillUntil/archive/the-gap-between-close-and-right/)
describes much earlier annealing results, not today's leading coordinates.
It supplies no improvement to transplant. Standard circle-in-semicircle tables
solve a different problem. These are excluded as solution sources.
