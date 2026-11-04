# Cold audit, second pass — 2026-11-04

Purpose: challenge the algorithms and the interpretation of negative results,
not accumulate more repetitions of the previous campaigns. Nanounit figures
below mean `1e-9` of the puzzle's unit radius. No new packing was promoted and
`solution.json` was not edited by this audit.

## 1. Live evidence and what it does not establish

Refreshed the exact public API:

```sh
python scripts/arena.py leaderboard --save /tmp/round2-leaderboard.json
```

Copied the response to `leaderboard.json` here.

| Rank | Author | Server radius |
| --- | --- | --- |
| 1 | @cameronspirals | 2.9485269238856113 |
| 2 | @fordgt1111 | 2.948526931599013 |
| 3 | @rorzio | 2.948526966725367 |

The leader is 42.8397557 nanounits below our radius and 7.7134019 nanounits
below second. These scalar scores do not identify the leader's contact graph,
angular branch, support features, or algorithm. Similar scores suggest a useful
research hypothesis, not proof of the same topology or basin.

Historical primary sources, checked directly:

- [Velocity, March 31, 2026](https://buttondown.com/TillUntil/archive/velocity/)
  reports an earlier first-place packing at **2.960586**, using large-neighborhood
  search and GJK polishing, with 30 contacts and a topology labelled `2-11-2`.
  This is a dated self-report about a substantially worse radius, not evidence
  that the present leader used the same method or configuration.
- [Till's about page](https://www.tilluntil.com/about) links Cameron Armstrong
  to `@cameronspirals`, establishing the connection to that account.
- [The Gap Between Close and Right, March 28, 2026](https://buttondown.com/TillUntil/archive/the-gap-between-close-and-right/)
  describes failures of annealing, approximate overlap checks, and assumptions
  about pairing semicircles into disks. Its density-bound assertion was not
  independently verified here and is not used as a mathematical bound.

The live API, not those historical scores, determines today's target.

## 2. A real false-negative in free-center angular rounding

Frozen the API-verified incumbent in
`tests/fixtures/packing_valid_20261104.json` so later promotions cannot silently
change these regressions.

Metamorphic construction: keep all positions unchanged and increase only
theta14 by a fraction of one microradian. The two adjacent angle choices still
include the exact original valid angle. Test at the incumbent radius plus 1 nm.

Before the correction:

| Added theta14 fraction | Original free-center helper | At known valid center |
| --- | --- | --- |
| 0.1, 0.2 | SAT | SAT |
| 0.4, 0.6, 0.8, 0.9 | UNSAT | SAT |

Thus the helper had rejected provably feasible binary families. It was testing
the MEC center of the *continuous* proposal and silently treating that one
center as the angular family's only center. Data are retained in
`incumbent-angular-recovery.json`.

Initial fix: on default free-center failure, try the true MEC center returned
by a minimum-cost angular completion, then solve the same SAT tables again.
Explicit `center=...` retains fixed-center semantics. Free-center failures are
labelled `no_assignment_at_tested_centers`, not global UNSAT.

An independent positive-control probe shifted every angle into a random
positive fractional cell, keeping the known valid all-floor assignment in the
domain. Seed **202611043**, 48 cases: **48 recovered**, and the original
validator accepted every returned packing. Data:
`random-known-family-recovery.json`.

The root agent then found a separate caller problem: the position projection
still passed an explicit center, bypassing the free-center recovery. Its old
fixed-center failures therefore do not establish angular-family infeasibility.
The root owns that caller correction; this audit owns the helper correction.

## 3. Generic boundary-angle center coverage

Added `scripts/angular_boundary_centers.py`; `audit2_boundary_centers.py` is a
compatibility import for the first experimental name. API:

```python
records, info = boundary_centers(p, angle_choices, target=R,
                                max_combinations=256)
```

The helper detects endpoint-sensitive shapes from geometry, not a hardcoded
piece number. It bounds the shape family's Hausdorff perturbation by delta.
Supporting-point geometry bounds the new MEC-center displacement by
`2*sqrt(R*delta + delta**2)`. Shapes guaranteed radial across that center ball,
or guaranteed strictly interior, need no orientation enumeration for the MEC.
The remaining endpoint choices are enumerated, and each center is checked for
containment of all unenumerated variants and shared active support witnesses.

`round_angles` now tries these bounded center families before the minimum-cost
fallback. Typical current-basin calls enumerate only the two piece-14 choices.
The default cap is 64 combinations; truncation is explicit. `complete_geometry`
and `native_family_covered` concern only the finite angular family, the stated
numerical tolerance, and the native estimated area tables. They are not global
packing impossibility certificates.

Three independent oracle tests cover generic endpoint detection, permutation
invariance, and a 17-angle domain with explicit truncation. The helper was
provided to the separate valid-neutral-network worker for wider angle CSPs.

## 4. The old hill-climb MEC rejection was mathematically unsound

The old screen multiplied exact distances from the *old* center by dual weights
computed from the *old* boundary normals. Once the geometry changes, those
normals are not gradients of the new distances. Their weighted sum at the old
center is not a lower bound on the new free-center MEC.

Counterexample: translate the unchanged valid packing by `(0.001,-0.002)`.

| Quantity | Radius |
| --- | --- |
| True translated MEC | 2.9485269667253684 |
| Old claimed lower bound | 2.9485280151839564 |
| Corrected support lower bound | 2.9485269667243665 |

The old bound overstates an equivalent packing by **1048.46 nanounits**. This
translation is not itself one of the old one/two-coordinate moves, so the
counterexample does not prove that an old campaign missed a leader. It does
invalidate the claimed general bound. Data: `hill-bound-counterexample.json`.

Corrected `experiment_hillclimb.py` to use fixed-direction support functions.
For nonnegative weights with sum one and balanced directions,
`R >= sum(weight * support(direction))`. A residual correction bounds numerical
direction imbalance using a bound on the possible MEC-center displacement.
Two-point-supported cases are handled. A small arithmetic guard is explicit.

The old tangent/static circle-to-polygon rejection remains available only under
`--heuristic-areas`. The default now uses cached exact GEOS polygons for pairs
incident to changed pieces. Initial validity is checked, since skipping
unchanged pairs requires a valid incumbent. Logs now distinguish area checks
from valid-area survivors; the old `tested 0` label meant zero survivors, not
zero checks.

## 5. One justified corrected hill-climb replay

This replay was specifically authorized because the old rejection was wrong.
It does not repeat an unchanged campaign.

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/research_run.py \
  --directory experiments/2026-11-04/round2/audit/runs \
  --label corrected-hill-60 \
  --hypothesis 'Audit found the old fixed-center-distance dual rejection unsound; replay once with fixed-direction support lower bounds and cached exact polygon rejection, not static circle corrections.' \
  --timeout 100 -- .venv/bin/python scripts/experiment_hillclimb.py \
  --input tests/fixtures/packing_valid_20261104.json \
  --output experiments/2026-11-04/round2/audit/hill-best.json \
  --seconds 60 --seed 202611042 --triples 100000
```

Result: clean exit after 60.51 wall seconds. Two complete passes over the 4140
single/pair neighborhood each had 408 potentially improving radius survivors;
all failed exact polygon overlap checks. Five full random triple batches and
one deadline-truncated batch also had no valid-area survivor. The printed
eligible count for those batches is not the exact count processed before the
final deadline. No candidate was promoted or sent to the API.

This supports a finite local statement at the explicit improvement threshold
`1e-11`. It says nothing about coordinated larger moves, neutral paths,
different contact graphs, or every triple assignment.

## 6. Projection divergence: reproduced mechanism, not an infeasibility proof

Added `audit2_projection_trace.py`, which instruments one bounded frame rather
than conducting another search. The first diagnostic uses seed **202611046**,
eight steps, frame zero, and integer theta14. Neither continuous nor minimum-cost
completion diverged in eight steps. Therefore divergence after exactly four
steps is not universal.

Then instrumented the actual archived failure configuration: seed **110442**,
incumbent input, target `2.9485269238`, movement 1000, beta 1.1, integer angles
13 and 14, minimum-cost completion. Only one frame was run, stopping at its
existing guard.

| Decoded step | Maximum integer displacement | Actual violation | Nonlinear intercept drift |
| --- | --- | --- | --- |
| 0 | 6 | 4.85e-7 | 0.00171 |
| 1 | 839 | 8.76e-7 | 0.247 |
| 2 | 2245 | 5.44e-6 | 2.03 |
| 3 | 8073 | 1.33e-4 | 35.19 |
| 4 | 155310 | guard rejection | — |

The **155310** step exactly matches the archived four-step failure. The fixed
Jacobian plus rapidly changing nonlinear intercept produces growing residual
feedback outside a true local trust region. The `movement` parameter clips a
reflected embedding coordinate; it is not a bound on decoded proposals. The
separate guard only rejects displacements above 20 times that parameter.

Also demonstrated a precise operator mismatch. A continuous theta14 at +0.9
microunits is closer to the upper grid value in the center-quotient Euclidean
metric. The minimum-violation solver chooses the lower, valid value, which is
**nine times farther** in that metric. This is correct for its violation
objective, but it is not the orthogonal lattice projection in the nominal DR
split. A continuous-angle point already belongs to that original lattice-plus-
continuous-subspace set, so its orthogonal projection would leave it unchanged.
Mixing this new completion operator into DR has no inherited contraction or
convergence guarantee. This does not prove that the operator change alone
caused every failure, nor identify a matrix indexing/sign bug.

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/research_run.py \
  --directory experiments/2026-11-04/round2/audit/runs \
  --label projection-logged-failure-diagnostic \
  --hypothesis 'Instrument only one frame of the exact previously logged divergent configuration (seed110442); stop after its guard, not a campaign replay.' \
  --timeout 30 -- .venv/bin/python scripts/audit2_projection_trace.py
```

Exact commands, source hashes, traces, and stdout are in `projection-trace.json`
and the run records.

## 7. Remaining evidence limits and validation

- The projection models hardcode flat pairs `(1,5)` and `(6,10)`, sometimes
  endpoint slot 14, and tight relative-angle clipping. Those are physical
  branch restrictions, not merely translation/rotation gauge removal.
- The new boundary-center oracle detects endpoint pieces geometrically and is
  permutation-tested; the old global search models are not generally invariant
  to arbitrary relabelling without remapping their contact assumptions.
- No analysis of today's scalar leaderboard establishes that the leader belongs
  to our preferred contact family. The older public history is not current
  solution disclosure.
- No new periodic-angle experiment, private-endpoint probe, or long campaign
  was conducted. All diagnostic and replay runs have finished.

Targeted regression command:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/test_audit2_regressions.py tests/test_audit2_boundary_centers.py \
  tests/test_angular_min_cost.py -q
```

The tests cover the known feasible angular families, explicit fixed-center
semantics, the dual-bound counterexample, 48 independent grid perturbations,
two-point support, exact cached overlap agreement with the original validator,
generic boundary detection, permutation invariance, and angular enumeration.
The root agent owns the final whole-suite run and any public solution promotion.

Final targeted run after boundary-oracle integration: **25 tests passed**.
Captured stdout, source snapshots, environment versions, and exit status are in
the `audit-regressions-final` record under `runs/`. All audit code and tests are
now frozen for the root agent's whole-suite check; no owned process remains.

Outcome: **no new verified score improvement in this audit**. The important
result is removal of two real rejection errors and narrower, defensible claims
about what the failed searches actually tested.
