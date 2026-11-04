# Adaptive outer-frame search

This is a distinct follow-up to the failed event-priority policy. It changes the
selection of continuous physical modes, rotation and fractional XY grid phases.
The earlier learned-slack experiments changed the proposal distribution inside
a frame; this experiment keeps that distribution fixed and optimizes the frame.

## Model correction and control

While preparing this runner, we found that the event runner's fitted frames
included a common translation, whereas `TiedFrame` evaluates enclosure constraints
at center zero. Some native previews consequently failed their stress test and
fell back to rounded frames. This weakens the earlier coverage claim; it does not
change the validity of its separately checked submissions or finite exact models.

The adaptive runner uses the centered continuous section to build `TiedFrame`,
then applies the fitted translation's fractional grid phase through the integer
anchor and `b0` adjustment. It freezes input files before fitting them.

A native positive control at the incumbent score plus `1e-11` recovered
**2.9485269509125525** in its first 1,024-proposal batch, in **1.096 seconds**.
All three permissively eligible control candidates passed the reference
validator. The production arm at the first-place target then recovered the same
incumbent after 1.147 seconds. These are recoveries, not improvements.

## Frozen comparison

There are three starts: the incumbent, the invalid three-overlap event frontier,
and the new hop provided by root. Each start has a population of twelve frames.
The five outer parameters are two physical modes, rotation, and two periodic
fractional translation phases. Physical/rotation offsets are scaled by 400
microunits and bounded to ±1,200 microunits; translation phases cover a full grid
period.

Every frame decodes the same 512-proposal random batch with seed 3112501. The RNG
is restarted for each evaluation. The evaluator normalizes every scored packing
by an exact common integer translation and uses one frozen vector of pair-area
scales. Thus the same scored pose has the same fitness regardless of its source
frame. Full pose keys include every angle; center-only identities affect parent
sampling weights, never candidate eligibility.

Fitness is lexicographic: first the maximum overlap/enclosure deficit, then the
sum of positive deficits. Differential evolution uses mutation factor 0.7 and
crossover probability 0.8. Parent sampling gives repeated center patterns less
weight without discarding them. The random arm shares the initial population
and proposal kernel, then draws fresh frames from the initial distribution.
Both have 240 seconds, run sequentially with one numerical thread.

Every candidate passing the permissive native area screen and radius cap is
reference-checked. Pareto witnesses retain their full raw submission and exact
GEOS/reference-MEC diagnostics. Native screens and Pareto nomination remain
heuristics; no global coverage or infeasibility claim follows from this search.
The runner does not call the API or replace the global incumbent.

```bash
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python scripts/experiment_adaptive_frames.py \
  --input solution.json \
  --input experiments/pantheon-pilot/deadline-events-v2/event/frontier-0008.json \
  --input experiments/pantheon-pilot/deadline-hops-v1/hop.json \
  --output experiments/pantheon-pilot/deadline-adaptive-v1 \
  --seconds 240 --batch 512 --population 12
```

Inputs, source snapshots, hashes, fixed scales, parameters, full frame records,
validated candidates and exact Pareto witnesses are saved in that ignored run
directory. Four focused tests passed before launch: integer-gauge identity,
angle-sensitive pose keys, fixed-scale fitness, phase periodicity and positive
diversity weights. These tests are narrow algebraic checks; the incumbent
recovery above is the real-geometry integration control.

## User-requested wind-down

The adaptive arm completed **1,143 frames in 240.034 seconds**. Its four
reference-valid records only recovered the incumbent; the best was
**2.9485269509125525**, equal to the stored incumbent apart from floating-point
roundoff after a common integer translation. Five exact Pareto witnesses were
retained, all nonwinning.

The user then requested an immediate wind-down while the random arm was active.
Runner PID 27697 received SIGINT and exited with code 130. Its last complete
atomic checkpoint contains **200 frames at 49.510 seconds**, three valid
incumbent recoveries and four exact Pareto witnesses. Work after that checkpoint
was interrupted and is not included in the count. The comparison is incomplete;
no adaptive-versus-random performance conclusion is claimed.

The native-preview failure audit by origin, contemplated for the preceding
event experiment, was not run after the stop request. The zero-center correction
is independently supported by the successful native recovery control above.
No owned search process remains, no first-place result was found, and no API
call, public submission or global incumbent replacement was made.
