# Integer centers, then angular 2-SAT

## Hypothesis and derivation

The earlier beams spent many entries on cheap angular moves. Eliminate those
integer dimensions during candidate generation. Fix the global integer
translation by holding piece 0's center, leaving 28 integer center coordinates.
Treat the 15 angles and two enclosing-center coordinates as continuous variables
in the embedding's orthogonal projection. The code is
`scripts/experiment_position_projection.py`.

For fixed centers, give each piece its floor and ceil decimal angle. Every
incompatible pair assignment is one binary clause; containment gives unary
clauses. This is a 30-node implication graph, solved by Tarjan SCC in
`scripts/angular_rounder.py`. It removes the need to enumerate 32,768 assignments.
All 30 semicircle variants are evaluated together by the polygon-area kernel.
Final reference validation remains compulsory because the area kernel estimates
the server polygons.

The rounder matched brute force on 350 randomized binary systems, recovered the
known valid incumbent exactly, and rejected radius 2. Baseline cost including
MEC is about 0.54 milliseconds per placement.

## Bounded experiments

Complete commands, versions, source snapshots, and output are under `runs/`.

1. `integer-centers-2sat`: seed 110430, target 2.948526950, 180 seconds,
   continuous angular projection. Reached a 5.76e-9 normalized continuous
   violation within 3.2 seconds. This does **not** establish angular-grid
   feasibility; only the rounder followed by reference validation can do that.
2. `integer-centers-angular-feedback`: seed 110431, target 2.948526966,
   300 seconds, angular weight 0.001. Snap the reflected angular coordinates
   onto the angle grid before residual feedback. This splits integer projections
   between center decoding and scalar angle rounding.
3. `integer-centers-strong-angular`: seed 110432, target 2.948526960,
   300 seconds, angular weight 0.01, beta 0.5. Test whether stronger angular
   feedback favors center configurations whose angles can actually be rounded.

Both the center projection and angle-grid projection are nonconvex heuristics.
Failure of these bounded runs is not an impossibility certificate.

## Diagnostics that changed the formulation

The initial three runs ended with **zero** satisfying adjacent-angle assignments
despite 1,855,773 position proposals and 900,299 angular checks. Some continuous
scores were below the leader, but those are not submission candidates.

Wider fixed-center angular repair found specific discrete obstructions:

- Piece 14 has two nearly active diameter endpoints. On `best_continuous.json`,
  its continuous angle yields R=2.94852692147954, while the best nearby integer
  angle forces R≥2.94852697785667 even after optimizing the enclosing center.
- On `angular-feedback_continuous.json`, pair (12,13) needs θ13≥-3.053402,
  whereas pair (2,13) needs θ13≤-3.053403. A ±50-microradian domain cannot
  satisfy both. Reference polygon checks confirm the nearest excluded choices.

These are local grid obstructions, not failures caused by too narrow a two-value
rounding choice. Their derivation and exact commands are in
`angular-repair-log.md` and `scoring-audit.md`.

The position search now accepts `--integer-boundary-angle` and
`--integer-angle-indices`: retain problematic angles in the joint integer
lattice, while eliminating the remaining angles continuously. Bounded cases:

- Seed 110433: piece 14, 300 seconds, target .526960.
- Seed 110434: piece 14 plus reflected angular feedback, weight 0.0001,
  300 seconds, target .526960.
- Seed 110435: pieces 2,13,14, 180 seconds, target .526960.
- Seed 110436: pieces 0,1,2,3,4,11,12,13,14, 300 seconds, target .526966;
  these are the columns with contact-Jacobian norm >0.5. Norms are recorded in
  `positions/angular-sensitivity.json`.
- Seed 110437: only pieces 13,14, 180 seconds, target .526960.

Review also found that caching UNSAT by centers and angle cells alone can miss a
different enclosing-circle center. The deduplication key now includes that
center at 1e-13 resolution; the exact reference validator still decides success.

## Least-violating angular completion

The next projection variant completes each decoded center configuration with
the adjacent-angle assignment minimizing summed squared actual contact and
containment violations. Native Gray-code enumeration checks all 32,768 binary
assignments cheaply. Feeding this completion back into the projection gives
the center update a geometric repair target even when 2-SAT is unsatisfiable.
Flag: `--mincost-projection`. This is a heuristic projection change; it makes
no convergence claim. Independent cost-table verification is required before
running the integrated experiment.

The cost solver passed 13 tests, including independent exhaustive comparisons
at ordinary and tiny energy scales, infeasible cases, the saved valid packing,
and n=1 handling. Native enumeration takes 0.523 ms; full cost construction and
fixed-center enumeration 0.713 ms; automatic center refinement 2.23 ms median.
Evidence: `angular-min-cost-benchmark.json`, `tests/test_angular_min_cost.py`.

Integrated runs: seeds 110439/110440, 300 seconds each, target .526966;
110439 keeps angles13/14 in the integer master, 110440 lets all angles float
until energy completion. Seed110441 starts at the actual valid packing with
fixed orientation for180 seconds: it recovered that exact valid score after
three iterations, a useful end-to-end positive control. Seed110442 applies
stronger compression at the leader target and widens moves to1000microunits;
valid intermediate improvements can be retained even above the search target.

The initial incumbent and long-move runs diverged from their local model and
eventually failed the angular solver's grid-input guard. Their nonzero exit
statuses and tracebacks are retained. Integer reconstruction is now explicit,
and a trust-region restart rejects decoded steps larger than 20 times the stated
movement. Future runs terminate after 20 divergent restarts. The already-running
bounded replay still used the earlier restart-only version and spent its budget
repeating the same four-step divergence; this is a failed method configuration,
not additional productive search. Its log is retained without overwriting the
initial failures.

## Implementation correction affecting one batch

When adding arbitrary integer angles, the first generalized model omitted their
contribution to the two paired-flat embedding rows. This affected the37-variable
sensitivity-based batch110436, which contains angle1. It does not affect batches
whose added angles are only2,13,14 (their flat-row coefficients are zero).
Corrected the rows and reran as110438 for180 seconds. The first batch is retained
as an implementation failure, not evidence against the37-variable strategy.

## Final outcome

Every bounded run finished. No new reference-valid improvement was produced.
The two 300-second minimum-cost projection campaigns made 125,527 and124,240
proposals respectively; their best remaining violations were36.23nm and35.49nm.
The incumbent-centered positive control recovered the original valid score,
but its subsequent divergent search did not improve it. All failures and source
changes are visible in the run records. Final verification:78 tests pass;
the production solution is unchanged and remotely revalidated.
