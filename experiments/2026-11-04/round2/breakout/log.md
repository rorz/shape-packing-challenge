# Adaptive-contact search and native scorer

## Hypothesis

Fixed squared-violation energy can trap search by repeatedly sacrificing the
same contact. Increase prices on currently violated constraints at local traps,
retain physical angular moves, use reduced integer lattice moves jointly across
pieces, and require the original validator for any purported improvement.

## Scorer construction and verification

- Added `_breakout_geometry.cpp` and `breakout_geometry.py`. Areas use the
  existing chord-corrected C++ kernel. MEC enumeration solves tangent pairs and
  triples of endpoint points and exposed unit-arc disks. A center inside the
  support-center convex hull has positive support stress. A candidate that
  contains all actual semicircles is optimal for that local support problem;
  missing support configurations fall back to the independent convex solver.
- Seed **110450**: 1,000 placements with Gaussian coordinate/angle deviations
  of 1e-4 from `solution.json`; compare every tenth point plus the incumbent
  against the independent MEC. Result: 21.26 microseconds per complete area+MEC
  candidate, maximum radius difference 1.55e-14 across 101 comparisons, zero
  native-area difference, zero fallbacks. `geometry-benchmark.json`.
- Added four regression tests: mixed endpoint/arc supports, deliberately empty
  support selection and fallback, rigid invariance, arbitrary 1/2/4-piece
  instances. `pytest tests/test_breakout_geometry.py -q`: all pass.
- The first narrow support window was unsuitable for larger lattice moves.
  Seed **110458**, 1,000 deviations with standard deviation .002: a .001 window
  needed 874 fallbacks and 646 microseconds per candidate; .01 needed none and
  20.22 microseconds. Maximum difference over 50 independent reference radii:
  2.07e-13 for the wider window. `wide-window-benchmark.json`. The default is now
  .01. `python search.py benchmark` includes this reproducible batch benchmark.

## Bounded search runs

All exact command lines, source snapshots, stdout and exit codes are in
`../runs/ledger.jsonl` and its linked files. None below produced a reference-valid
improvement. The reported `best_score` is initialized from the saved incumbent;
it does not mean the run found that score anew. `_nearest.json` files are
diagnostics, not submissions.

| Label / seed | Seconds | Evaluations | Accepted distinct states | Distinct center configurations |
| --- | ---: | ---: | ---: | ---: |
| breakout-smoke / 110451 | 12 | 505,528 | 787 | not recorded |
| breakout-incumbent / 110452 | 180 | 4,866,469 | 1,881 | not recorded |
| breakout-three-conflicts / 110453 | 180 | 4,196,521 | 1,522 | not recorded |
| breakout-quantized / 110455 | 180 | 4,885,769 | 3,389 | 167 |
| breakout-angular-conflict / 110456 | 180 | 4,109,184 | 1,691 | 44 |
| breakout-mincost-frame / 110457 | 180 | 6,528,432 | 3,429 | 175 |
| breakout-reflected-patch / 110462 | 180 | 7,105,922 | 2,748 | 166 |

The first two long runs spent many iterations revisiting states. The numerical
descent threshold was below possible MEC fallback noise. Later runs quantized
the **search energy only** to 1e-12 coordinate units, retained full precision for
validation, enlarged the tabu memory, and periodically raised prices on revisited
center configurations. The correction improved observability and trap handling;
it did not find a better packing. A root cause of every repeated state was not
formally established, so do not attribute all cycling to floating-point noise.

The rotated angular-conflict start reduced its maximum remaining violation to
40.304e-9; the min-cost start retained 40.164e-9. Both remain invalid. The known
incumbent remained the best energy in its own frame. Millions of correlated
proposals do **not** establish broad geometric coverage: the last three runs
visited only 167, 44 and 175 distinct center configurations respectively.

The later reflected-patch run uses a genuinely different equal-radius contact
topology from the surgery campaign. Its starting snapshot has R=2.9485269000239547
but five overlaps, maximum normalized excess19.991e-9. After180 seconds no
reference-valid improvement appeared; it remains invalid. Opposed flat pairs
are now detected from geometry in both the LLL search and this basis builder.
The source snapshot is `new-geometry-snapshot.json`, taken before search so
concurrent LLL checkpoint updates cannot change this run's input. The wider
native support window needed zero MEC fallbacks in all7,105,922 evaluations.

## Inference

The speed improvement is real and reusable. The adaptive-price objective is not
enough to solve the integer bottleneck in these three frames. Repeating these
same settings is not justified. A new move generator, a larger structured
neighborhood, or direct angular-conflict constraints is required.

The new `plot_contact_slack.py` generated `../contact-slack.png` and `.svg`.
The image was visually inspected. It shows the current circle supports 3/5/7,
near-active endpoints of piece14, and pair slacks normalized by area gradients.
These normalized slacks are diagnostics, not exact geometric clearances.
