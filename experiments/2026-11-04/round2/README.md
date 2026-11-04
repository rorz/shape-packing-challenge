# Adversarial follow-up: 2026-11-04

## Result and the important correction

**No new API-valid improvement.** The saved radius remains
**2.948526966725367**, against first place **2.9485269238856113**.
`api-final.json` confirms the saved file is valid; it was not changed or submitted.
The full suite passes **98 tests** (`tests-final.txt`).
All bounded runs have finished. `solution.json` has no diff.

The strongest discovery is a different, equal-radius geometry: reflect a
three-piece patch while the other twelve remain in place to within 1.1e-6.
The continuous radius is **2.948526549416158**. Its contact graph is
non-isomorphic to the original, and rigid/permutation/reflection invariants rule
out a renamed copy. The old basin archive **deduplicated by radius alone**, so
its repeated score did not prove repeated geometry. That filter is now replaced
with pose congruence checks, and every new geometry is saved independently of
whether it improves the score. See `topology/patch-audit.json`,
`topology/equal-radius-graph-audit.json`, and `patch-reflection.png`.

Two 180-second lattice runs tested this new geometry: **332,121** and **326,005**
proposals. Flat pairs are now detected geometrically rather than fixed to piece
indices. One diagnostic reached radius **2.9485269000239547**, below first, but
has **five overlaps** and is **invalid**. A further 180-second adaptive repair
checked **7,105,922** proposals without fixing them. These results support
retaining the new seed, not promoting it or claiming a win.

## What was missing

1. **Geometry-aware diversity.** Equal radii can hide a local mirror flip and a
   distinct contact graph. Score-only deduplication destroyed useful starts.
2. **Known-valid rejection controls.** The angular rounder rejected four families
   containing the incumbent because it fixed the wrong enclosing center. An old
   hill-climb screen also overstated a valid packing's radius. Both are repaired
   and regression-tested. The position-search caller was separately changed to
   invoke free-center rounding; its corrected bounded replay made 9,505 proposals
   and 9,371 angular checks, found no assignment, and ended on its 20-restart guard.
3. **Physical neutral moves.** A valid angle change with the same radius is not
   a redundant translation. The new search found 505 valid configurations across
   two relative-center states, including a valid worse-radius relay. Its bounded
   expansion only found a return edge. See `neutral/research.md` and
   `neutral/final-summary.json`.
4. **Honest search coverage.** Millions of correlated proposals are not millions
   of independently explored packings. The last three original-frame breakout
   runs visited only 167, 44 and 175 center configurations. `breakout/log.md`.
5. **Stable projection operators.** The old four-step divergence is reproducible
   nonlinear feedback outside a real trust region. Minimum-cost angular repair
   is not an orthogonal projection. Failed feedback is not an infeasibility
   certificate. `audit/log.md` and `audit/projection-trace.json`.

## Why the leader probably wins; what remains uncertain

The public API gives scores, not the leader's coordinates. A better integer-grid
realization of one of these competitive geometries is the strongest working
hypothesis. The gap is just 4.28398e-8; one coordinate step is 1e-6, about 23.3
times larger, so coordinated changes matter. Conditional on the same continuous
minimum, the leader pays about 10% less rounding overhead. This is an inference,
not knowledge of their algorithm or a global optimality claim.

The leader's older public writeup describes large-neighborhood search plus
geometric polishing at R=2.960586, long before the current score. It supports
trying structural moves but does not reveal the present solution:
[Velocity, March 31](https://buttondown.com/TillUntil/archive/velocity/).

## Strategies actually exercised

- **Structural moves:** 5,600 removal/regrowth, disk-fission and unequal-cleavage
  trials. The first family alone reached 1,338 distinct feasible unlabelled
  contact graphs. Isomorphism and rigid-invariance audits separate real diversity
  from piece renaming. The new mirror patch came from this work.
- **Hill climbing:** exact-overlap single/pair/triple checks, valid neutral walks,
  conditional center-plus-angle moves, a bounded worse-radius relay, and adaptive
  contact prices. None improved the saved score in these runs.
- **Angular geometry:** enumerate the endpoint-sensitive angular choices first
  (piece14 in the original arrangement, piece12 in the new one), solve the
  remaining pair compatibility, and refit the enclosing center. The generic
  helper is `scripts/angular_boundary_centers.py`; its completeness flag applies
  only to the stated finite angular family and numerical tolerance.
- **Grid symmetries:** all six nontrivial quarter-turn/reflection cases fail
  area-only angular consistency even over ±50 microradians. The original and its
  plain reflection remain valid but are equivalent. Do not repeat this test.
- **Finite patch-flip inventory:** all 295 connected subsets of sizes1–4 crossed
  with 45 candidate diameter axes, **13,275 proposals**, finished in 1.16 seconds.
  It recovered the known three-piece flip (positive control) and a baseline
  equivalent, with no additional noncongruent seed. Tiny repairs stayed within
  the fixed +1e-9 radius budget and passed full continuous-coordinate GEOS checks.
  These are continuous seeds, not valid rounded submissions.
  `topology/inventory/summary.json` records the precise finite scope.

## Fast reruns

The final benchmark is **19.2 microseconds** per batch candidate for all overlap
estimates plus the enclosing circle, about 52,000 candidates/second. Unsupported
MEC support configurations fall back to the convex solver; promoted outputs
still require the original polygon validator and API. `benchmark-final.json`.

```sh
.venv/bin/python search.py benchmark
.venv/bin/python search.py slack-plot
.venv/bin/python scripts/plot_patch_reflection.py
.venv/bin/python scripts/research_run.py \
  --directory experiments/2026-11-04/round2/runs \
  --label NEW-NAME --hypothesis 'State the new reason for this run' --timeout 150 -- \
  .venv/bin/python search.py breakout --seconds 120 --seed NEW-SEED --output experiments/NEW.json
```

Use a new configuration for a stated reason. The ledger rejects exact successful
repeats and stores source/input hashes, source snapshots, commands and outputs.
`summary.json` indexes the root run records. Agent-owned records are linked from
the audit, neutral and topology reports.

This is a new campaign after the first 30 recorded runs failed to improve the
saved score. It must not silently repeat their assumptions or configurations.

Starting saved score: **2.948526966725367**. Public leader at campaign start:
**2.9485269238856113**. The exact difference is **4.28397557428184e-8**.
The leader's coordinates are unavailable; a shared topology is an inference,
not a known fact. An invalid lower-radius file is not an improvement.

## Concurrent lines of attack

- `audit/`: cold audit and known-answer/metamorphic controls. The old angular
  routine can report fixed-center UNSAT for a family containing our saved valid
  packing. An old hill-climbing screen is not a sound lower bound after motion.
  Repairs require regression tests, not just a note in this log.
- `neutral/`: retain valid equal-radius angular moves and recombine them to
  create a bridge to an improving center move. Cheap physical angles are not
  redundant rigid translations.
- `topology/`: explicitly remove/reinsert pieces, grow shrunken pieces, and
  repartition paired semicircles through a full disk. Record contact-graph
  diversity rather than equating restarts with independent basin coverage.
- `breakout/`: nonlinear discrete local search with adaptive prices on violated
  contacts. At a local trap, increase the price of the blocked constraints rather
  than repeatedly optimizing the same fixed squared-violation energy.

## Native batch scorer

`scripts/_breakout_geometry.cpp` evaluates all chord-corrected overlap areas and
enumerates near-active pair/triple enclosing-circle supports. Positive support
stress plus containment of every actual semicircle supplies a local optimality
certificate. Missing support sets or arc branches fall back to the existing
convex MEC solver. This is a proposal accelerator, never the final validator.

Initial benchmark: 1,000 independently perturbed placements, 21.26 microseconds
per candidate for overlap plus MEC, 101 independent reference MEC checks with
maximum radius difference 1.55e-14, no fallback; all areas match the existing
native area implementation exactly. Seed 110450; `breakout/geometry-benchmark.json`.
Initial compilation took 1.22 seconds and is cached by source hash in `/tmp`.

Root search runs use `../..`'s `scripts/research_run.py`, with the ledger and
source snapshots in this directory. Agent logs record their exact commands.
The contact/slack figure is reproducible with `scripts/plot_contact_slack.py`.
