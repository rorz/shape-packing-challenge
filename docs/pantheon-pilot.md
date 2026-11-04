# Blind comparison of native sampling and selective exact repair

This continues the [recovery benchmark](recovery-benchmark.md). The user
authorized the bounded comparison after three independent strategy reviews.
No public submission is part of this experiment.

## Control replay

`benchmark_translation_union.py` replays the four frozen, target-assisted
controls at angle widths ±1, ±3 and ±12, with either one translation phase or
a union covering every fractional XY translation. All 24 models recovered a
valid packing at the corresponding recorded score.

| Neighborhood | Angle width | Poses per piece on these controls | Median solve time |
|---|---:|---:|---:|
| Single phase | ±1 | 27 | 0.481 s |
| Single phase | ±3 | 63 | 0.611 s |
| Single phase | ±12 | 225 | 1.986 s |
| All translation phases | ±1 | 48 | 0.817 s |
| All translation phases | ±3 | 112 | 1.426 s |
| All translation phases | ±12 | 400 | 6.323 s |

These are observed control timings, not promised search speedups. The narrow
union is about 2.4 times faster than the previous ±12 single-phase model by
these medians. Unioning phases increases cost at a fixed angular width; its
benefit is broader coverage. Four controls do not establish that narrow angles
always suffice.

[`translation-union-results.json`](translation-union-results.json) retains
compact results and hashes. Full diagnostics and matching source snapshots are
local under `experiments/pantheon-pilot/control-replay/`, which is ignored.

The generator bounds the production rounding operations monotonically, with
outward floating-point guards. Exact-grid ties can need five values on an axis
instead of four. It preserves the exceptional raw ±3π angle aliases, checks
domain-size limits before allocation, and never truncates a domain. Tests check
containment against production neighborhoods at random phases, endpoints and
rounding ties.

## Frozen blind pilot

Run with:

```bash
uv run --extra lattice --extra constraint python scripts/run_blind_pilot.py \
  --output experiments/pantheon-pilot/run-001 --seconds-per-arm 120 \
  --seed 3111201 --exact-seconds 2 4 10
```

The runner exports source and three problem JSON files to a temporary bundle.
Each problem contains only its saved continuous origin, flat-angle branch and
unshifted score cap. Target placements, historical fitted parameters, fixtures,
research outputs and the incumbent are absent from that bundle. A Python audit
hook blocks original-workspace data reads and network connections; API functions
are independently disabled. This is an audited code path, not an OS security
sandbox. Allowed reads and blocked attempts are recorded.

Both arms use the same seeded native lattice proposal stream: two physical
modes with σ=0.0004, mostly rotations near the supplied origin with a fixed
global tail, 2,048 proposals per frame, and one numerical thread. Native
compilation is warmed before timing. Their shared proposal prefixes are hashed
and compared after each pair. This is a native-only control using a shared
proposal schedule, not a byte-for-byte replay of an older campaign.

The hybrid fits only its own fresh near-frontier proposals, then uses the
translation-union exact solver. It skips centers it has already qualified,
deduplicates complete domains, and reserves at most 25% of its accrued time
for fitting, domain building and solving. Angular stages receive 2, 4 and 10
seconds respectively, chosen from the control timings before the race. Wider
stages have a scheduling allowance so fresh narrow jobs cannot starve them.
An infeasible narrow model never excludes a wider one.

The six two-minute arms run sequentially, alternating which method goes first
across the three origins. A parent independently revalidates every saved raw
submission and qualifies it only by the supplied cap plus 1e-11. The solver's
additional 2e-12 guard is removed from its target to keep that cutoff identical.
Validation completed after the deadline receives no credit. Exact-solver
deadline overruns and incomplete solves are reported explicitly.

Novelty is graded against an archive frozen before either arm: both labeled XY
centers modulo common integer translation and unlabeled centers modulo square
rotations/reflections and translation. This records novelty against observed
history, not a claim that a packing has never existed. Exact repair may change
the local flat differences; native proposals preserve them. This is therefore
a comparison of complete methods, not an isolated causal estimate of the
translation union alone.

The preregistered GO rule requires all of:

- At least as many qualifying cases as the native control.
- Faster first qualification in at least two of three cases.
- At least 20% lower aggregate time to qualification, censoring failures at 120 seconds.
- No per-case best-score regression larger than 1e-11.

A failed gate rejects this particular repair policy for a larger campaign; it
does not rule out all exact methods or all translation-union searches. Three
cases are a pilot, not a statistical significance claim.

## Verification before the race

The full suite passed 208 tests. A separate 12-second hybrid smoke, with a
different seed from the official pilot, exercised a real 48-pose exact model.
Its audit recorded only the supplied problem and NumPy package metadata reads,
with zero blocked accesses or network attempts. Source and parameters are frozen
for the official six arms; subsequent changes must use a new run directory.

## Pilot result

All three cases qualified in both methods. The native proposal prefixes matched
in every pair, all saved results passed independent grading, and the access
audits recorded no blocked reads or network attempts.

| Origin | Native first qualification | Hybrid first qualification | Best radius, both methods |
|---|---:|---:|---:|
| Random-D | 14.783 s | 9.742 s | 2.948526986788315 |
| Neighbor-G | 0.377 s | 0.375 s | 2.948526950912552 |
| Fresh-baseline | 0.207 s | 0.208 s | 2.9485270133102732 |

The preregistered formula returns **GO**, with a 32.8% reduction in aggregate
capped qualification time. Practical interpretation is more cautious: Random-D
shows a real five-second benefit from exact repair, while the other two first
qualifications happen in the initial native work and their millisecond
differences are noise. Best scores are identical in each pair. The improvement
below was generated by native sampling in both methods, not by exact repair.

[`blind-pilot-results.json`](blind-pilot-results.json) preserves the compact
record. The original `process_seconds` field included independent parent grading;
the compact artifact correctly labels it `execution_and_parent_grading_seconds`.
Future runner versions record these durations separately. Search budgets remain
120 seconds per arm; total wall time also includes compilation/imports and
independent validation of 869 saved submissions.

## Verified improvement and its cause

The pilot produced **2.948526950912552**, independently accepted by the arena API
and the reference validator. It improves the preceding incumbent by
**1.2040946906211047e-8**, closing **30.82%** of the remaining gap to the recorded
leader. It would still place third, with **2.7026940685459522e-8** left to first.
`solution.json` and `experiments/best.json` now contain it. No public submission
was made. The [promotion record](../experiments/pantheon-pilot/promotion.json)
retains the old score, candidate hash, source and fresh leaderboard evidence.

The geometry audit found an older **invalid** witness with the same centers
after a clockwise quarter-turn and integer translation. Merely rotating and
rounding its angles remains invalid. The new candidate additionally changes
pieces 4 and 14 by +1 microradian, and piece 6 by −1 microradian. This is a new
valid angular realization of a previously observed center pattern. A quarter-turn
preserves the XY grid but not the six-decimal angle grid. Consequently the D4
center quotient is useful for reporting, not a sound rule for excluding search
states. [The audit](pilot-center-audit.json) includes both inputs, hashes and
reference overlap evidence.

## Harder-target follow-up

The formal gate and verified improvement justified one four-minute production
probe. It fitted a continuous starting frame to the newly verified packing,
then ran native search for 150 seconds and the hybrid for 90 seconds at cap
`2.9485269238656113`, leaving a strict margin below first even after the benchmark
guard. This was a production probe using the known incumbent, not another blind
or equal-budget method comparison. Neither arm improved the new incumbent or
reached first place. The hybrid spent 5.61 seconds on exact work.
The [compact probe record](first-target-probe-results.json) preserves the
parameters, timings and independently graded results.

The newly observed quarter-turn effect motivated one further bounded test:
fixed-center angular repairs across all eight square symmetries of four frozen,
invalid near-winners whose current radii are already below first place. The
inputs and hashes are in `tests/fixtures/angle-orbits/`. This differs from the
old incumbent-only symmetry tests; each input, symmetry, width and finite
threshold must remain explicit. No geometric equivalence is assumed for rounded
angles.

All **64 models** (four inputs × eight symmetries × widths ±3 and ±12
microradians) finished in **2.885 seconds** with `INFEASIBLE` at target
`2.9485269238806113`. This exhausts these finite neighborhoods only; it does not
exclude wider angular repairs, center moves, or other packings. No further
improvement was found. [The compact results](angle-orbit-results.json) retain
input, source and domain hashes, solver statuses and counts.
The target is slightly below the recorded leader; the tiny interval above this
target is not excluded. A separate review confirmed 64 distinct, nonempty domains:
52 models failed the initial conservative relaxation and 12 required exact GEOS
pair exclusions. These models include enclosure constraints, so the result is
not an overlap-only impossibility claim.

To reproduce this last audit in a fresh output directory:

```bash
uv run --extra lattice --extra constraint python scripts/experiment_angle_orbits.py \
  --input tests/fixtures/angle-orbits/tied-three.json \
  --input tests/fixtures/angle-orbits/mirror-five.json \
  --input tests/fixtures/angle-orbits/broad-four.json \
  --input tests/fixtures/angle-orbits/baseline-three.json \
  --target 2.9485269238806113 \
  --output experiments/pantheon-pilot/angle-orbits-replay \
  --seconds 60 --per-model 2 --angle-width 3 --angle-width 12
```

## Final verification and Git contents

The workspace suite passed **214 tests in 13.32 seconds**. An export containing
only Git-eligible files passed **211 tests in 13.07 seconds**, with three optional
large-archive replay tests skipped. This verifies that the ignored research
outputs are not needed by the portable suite. The export reused the installed
dependencies and selected its own `src/` through `PYTHONPATH`.

The Git-eligible tree contains 284 files totaling about 1.9 MB. Source, small
fixtures, compact results and API promotion evidence remain eligible; raw run
directories, caches and native build products are ignored. `git diff --check`
passes. No files were staged or committed, and no public entry was submitted.

Independent reference validation confirms `solution.json`, `experiments/best.json`
and the curated candidate are byte-identical, valid at **2.948526950912552**, and
have SHA-256 `cefac3368caae7f9dea0c4cdc0e87d869f3a9bd76ba75292b1cd4b7fff7c8fb9`.
