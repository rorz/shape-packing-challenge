# Rounding-event search under the prize deadline

The user authorized execution on 2026-11-04, with a two-hour prize deadline.
This experiment uses the API-verified incumbent and two archived continuous
origins as production starts. It is not a blind benchmark.

## Hypothesis and limits

The previous recovery benchmark showed that both physical deformation modes
matter. The quarter-turn improvement also showed that rounding-related angle
changes can repair an existing center arrangement. Instead of choosing unrelated
continuous frames, this test visits neighboring finite pose domains along the
two physical parameters and global rotation. Fractional XY translation is already
covered by the translation-union generator.

A ray first brackets a change in the actual generated domain, then bisects that
bracket. This does **not** prove the first event was found: a nonlinear ray may
leave and re-enter a domain between probes. The union of both endpoint pose sets
contains both sampled sides; it does not claim coverage of every intermediate
continuous frame. Full scored-domain hashes prevent repeated models. No D4 or
center-only equivalence is used to exclude angle realizations.

The event queue prioritizes the measured overlap/enclosure deficit of each
rounded frame. This is a heuristic score for scheduling. It is not a lower bound
for that domain, and its frontier statistics are not exact-solver feasibility
statistics. The ordinary arm chooses fresh frames from a 400-microunit Gaussian
around the fitted origins, with one quarter of rotations drawn globally.
Both arms use the existing lazy exact solver at the same radius target.

## Frozen protocol

The construction gate requires reference-valid recovery of the incumbent and at
least nine distinct endpoint-union domains within 60 seconds. The comparison
then allows 180 seconds per arm, with a cap of 200 exact models per arm and
1.5 seconds per model. The angular width is ±1 microradian; XY width is ±1
microunit before translation union. The target is `2.9485269238806113`, five
picounits below the recorded leader, with the existing solver's two-picounit
acceptance guard.

Each submitted candidate is independently reference-validated. Pair-feasible
leaves above the strict target are also checked because they can improve the
incumbent. Late validations are marked. There are no API calls or global
promotions in this runner. Root coordinates API checks and promotion separately.

```bash
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python scripts/experiment_rounding_events.py \
  --output experiments/pantheon-pilot/deadline-events-v1 \
  --seconds 180 --max-models 200 --per-model 1.5
```

Source and input snapshots, hashes, fit parameters, domain hashes, exact model
statuses, candidate files and timing records are saved under that ignored run
directory. The default starts are `solution.json`, `random-d-origin.json` and
`neighbor-g-origin.json` from the frozen recovery fixtures. The seed is 3112401.

Five focused tests passed before the run. They check actual-domain bracketing,
endpoint-union containment, bounded failed bracketing, invalid rays and invalid
continuous-section handling. The construction gate passed in 1.320 seconds,
recovering the incumbent and producing ten distinct endpoint-union domains.

## Version 1 result and correction

The initial comparison hit its 200-model cap before either 180-second budget:
event search took 68.924 seconds and ordinary frames 33.703 seconds. All 400
finite target models were infeasible; no valid candidate was produced.

Inspection exposed a scheduling flaw: the global rounded-frame priority spent
all 200 event models on neighbor-G and starved both other origins. Its best
rounded-frame deficit was 395.60 billionths, versus 320.71 for ordinary frames.
Those are merely rounded-frame proxies. This run does not exclude a useful
three-origin event search. Its original source remains frozen in the run's
`source-snapshot/` directory.

Root authorized exactly one corrected comparison. Version 2 selects origins
round-robin, with the best queued event within each origin; every tenth model
is a fresh random restart. It uses 256 native lattice proposals per candidate
frame to rank cells, and checks all promising complete raw submissions. The
reported frontier now retains full raw poses plus independent GEOS overlap and
reference-MEC diagnostics. Native proposals may lie outside the narrow exact
domain; each record states whether the proposal was contained. Such improvements
are attributed to native sampling, never to the exact model.

The revised limit is 180 seconds and at most 600 models per arm. This changes
both scheduling and priority, so it compares complete search policies rather
than isolating the causal effect of event traversal. A sixth focused regression
checks that round-robin quotas prevent origin starvation.

```bash
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python scripts/experiment_rounding_events.py \
  --output experiments/pantheon-pilot/deadline-events-v2 \
  --seconds 180 --max-models 600 --per-model 1.5
```

The v2 construction gate passed in 1.308 seconds. Its first reference-valid
native proposal recovered the known G score, `2.948526986433127`; this is not an
incumbent improvement. The full comparison is recorded in the separate v2
directory, preserving the failed first policy without overwriting its evidence.

## Corrected comparison result: stop this policy

| Arm | Models | Elapsed | Per-origin counts | Exact status |
|---|---:|---:|---|---|
| Event plus native preview | 486 | 180.011 s | 162 / 162 / 162 | 485 infeasible, one deadline |
| Ordinary plus native preview | 600 | 107.137 s | 200 / 200 / 200 | 600 infeasible |

The ordinary arm hit its 600-model cap; it did not consume the full 180-second
allowance. Event search used three initial models, 435 adjacent-event models and
48 fresh restarts. Neither arm produced an incumbent improvement. The deadline
model is incomplete and must not be counted as infeasible.

Both policies found the same center arrangement on their native frontiers,
with reference radius **2.9485269312942926**, still 7.409 billionths above first.
Its maximum exact normalized overlap deficit is **8.602 billionths**. Ordinary
sampling reached this arrangement after 23.77 seconds; the event policy first
reached it after 41.65 seconds through a random restart. This gives no evidence
that event traversal is helping, and this policy was not extended.

The useful three-conflict raw witness is
`deadline-events-v2/event/frontier-0008.json`, SHA-256
`85e253831a3c21aad38837fc1133d35522d32802acc56d84f9b65427218924cd`.
It violates pairs (0,8), (5,7), and (6,10), with overlap areas
`1.0003031812390632e-6`, `1.0002791423852047e-6`, and
`1.0000396184901302e-6`. It is invalid and cannot be submitted.

The later `frontier-0009.json` merely translates those centers and changes two
angles, adding a fourth overlap. Its tiny native-priority improvement is
estimator/frame-scaling noise, not an exact improvement. The current runner now
requires a 20-picounit priority change before updating that heuristic frontier;
reference-valid candidates remain checked independently of this threshold.

## Candidate-funnel audit

The v2 runner originally reference-checked only the best native preview in each
256-candidate batch. That could hide a valid alternative behind a lower-deficit
invalid candidate, or behind a tie. The current implementation retains every
candidate passing the permissive native area screen and radius cap; a seventh
regression verifies that tied and non-winning candidates survive this funnel.

A separate replay reconstructed all **1,086 recorded frames** with exactly the
same source inputs, parameters and per-domain RNG seeds, then checked every
eligible proposal. It finished in **15.380 seconds**: twelve distinct raw
submissions passed the reference validator (eleven from event frames, one from
an ordinary frame). They only recovered the known G and D scores; the best was
**2.9485269864331265**. No incumbent improvement was hidden in those batches.
This audit is outside the timed comparison and its results are not retroactively
credited to it. Unselected queued frames were not preserved and are outside the
replay's scope. No new parameter proposals were made.

The full replay evidence is in
`deadline-events-v2/batch-eligibility-audit/report.json`. The current seven focused
tests pass in 0.96 seconds. Sources from both actual campaigns are frozen in
their respective run directories; the registration and frontier-reporting fixes
described above were made only after those runs ended. No API call, public
submission, or global incumbent replacement occurred in this experiment.

A subsequent implementation review found another limitation: the fitted frames
retain a common translation, while `TiedFrame` assumes continuous enclosure
center zero. Some native previews therefore failed their stress test and used
the rounded-frame fallback. Native outputs and exact models were still checked
independently, but this further limits the search-policy comparison. The newer
[adaptive-frame runner](adaptive-frames-run.md) corrects this by building the
centered continuous frame first and applying only its fractional grid phase to
the integer anchor. An exact per-origin fallback audit was deferred, then
cancelled by the user's wind-down request; no count is inferred from preview
source labels alone.
