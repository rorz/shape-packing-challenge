# Exact recombination across physical deformation frames

The old 25-frame pool was inspected before this run. Its `phase_domains()`
function rotates one base by ±300 microradians about piece zero, then unions
local corrections. It never varies the two physical deformation coordinates.
Those old mixed domains therefore do not directly cover this experiment.

Five reference-valid sources are used: the current incumbent, random-D,
neighbor-G, fresh-baseline and the newly discovered valid center hop. Each is
fitted to the same continuous family, retaining both physical coordinates and
its geometric flat-angle differences. The fitted sections are then placed at
the incumbent's common rotation and translation. Rigid alignment preserves the
flat geometry modulo 2π; exact raw six-decimal differences can change at wrap
seams, and both versions are recorded. Other sources need not remain valid
after rerounding in the common frame.

Each aligned physical frame contributes translation-complete local domains
with XY width one and angle width one. The per-piece union is capped at 500
poses without truncation. Every actual incumbent pose is explicitly added at
index zero. This is an informed production model, not a blind recovery test.
Balanced enclosure supports are generated at the incumbent and every aligned
physical anchor, then passed through the existing factored support encoder and
lazy exact GEOS oracle. No shared solver source is edited.

The complete mixed model first has to recover a reference-valid packing at the
incumbent radius plus 1e-11, within 30 seconds. Only then does it receive a
180-second first-place target solve at 2.9485269238806113; its 2e-12 acceptance
guard still leaves a margin below the recorded leader. The run is offline,
with no API check or global promotion. A finite exclusion applies only to
these explicit union domains, not their continuous hull or intermediate
physical frames. Widely separated templates may prevent most mixed choices;
this bounded test measures that limitation instead of presuming recombination
will help.

Four focused tests passed in 1.70 seconds: preservation of genuine physical
deformations and flat geometry, union coverage without truncation, validity of
all-anchor support bounds under mixed choices, and restoration of the private
support hook after an exception.

```bash
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/experiment_physical_frame_pool.py \
  --input experiments/pantheon-pilot/candidate-2.948526950912552.json \
  --input tests/fixtures/recovery/random-d.json \
  --input tests/fixtures/recovery/neighbor-g.json \
  --input tests/fixtures/recovery/fresh-baseline.json \
  --input experiments/pantheon-pilot/deadline-hops-v1/hop.json \
  --output experiments/pantheon-pilot/deadline-physical-pool-v1 \
  --seconds 180 --control-seconds 30 --seed 3111901 --max-poses 500
```

Inputs, dependency versions, source snapshots, fitted coordinates, full domains,
pose-source membership and support hashes are retained in the ignored output
directory. No source subset will be silently changed after freezing the model.

## Result

The user requested immediate wind-down just after preparation. The full mixed
domain contained 144–204 poses per piece and 330 support cuts. Its incumbent
control passed in **3.79 seconds**, recovering **2.948526950912552**.

The root agent sent SIGINT to the running target process. CP-SAT returned
`UNKNOWN` after **4.01 seconds** of the planned 180-second target budget, and
the process exited. **The target search is unfinished, not an infeasibility
proof.** No candidate or improvement was found before interruption. No further
search or tests were started after the stop request.

The final report is marked `INTERRUPTED_BY_USER`; the solver's original
`UNKNOWN` result remains intact. The report before adding that interruption
note is also preserved. The complete finite domains and support data remain
available for a future explicitly authorized continuation. Their domain hash
is `78a5fd3a5b6e86d34e833bbb23db69795c92c6881548154c912e50affe08b102`.
