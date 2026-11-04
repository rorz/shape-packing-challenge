# Semicircle Packing Challenge

Pack **15 unit semicircles** (radius = 1) into the smallest possible enclosing circle.

Your **score** is the radius of the enclosing circle. Lower is better.

The baseline scores 3.50. The area lower bound is ~2.74 (you can't beat it, but you probably can't reach it either). How close can you get?

![Baseline packing](baseline.png)

## Getting Started

```bash
uv sync
uv run python run.py
```

This scores `solution.json` — the file you edit. Open it and tweak the coordinates.

## Format

`solution.json` is a JSON array of 15 semicircle placements:

```json
[
  { "x": -2.0, "y": 0.0, "theta": 0.0 },
  { "x": -2.0, "y": 0.0, "theta": 3.141593 },
  ...
]
```

- **(x, y)** — center of the full disk
- **theta** — angle (radians) the curved part extends toward. The flat edge passes through (x, y) perpendicular to theta.

**Precision:** Coordinates are rounded to 6 decimal places before validation and scoring. Angles are rounded, wrapped with `atan2(sin(theta), cos(theta))`, then rounded again. This matches the server, including angles with extra full turns. The local `run.py` prints 15 decimal places so small improvements remain visible.

This is the same format the arena accepts — paste it directly to submit.

The saved solution currently scores **2.948526950912552**, verified by the API
on 2026-11-04. First place was **2.9485269238856113** in the latest recorded
leaderboard, a gap of **2.70269e-8**. The original packing is preserved in
`experiments/original.json`; the new validation evidence is in
`experiments/pantheon-pilot/candidate-api.json`. No entry was submitted.

The November 4 adversarial search is documented in
[`experiments/2026-11-04/README.md`](experiments/2026-11-04/README.md), with
separate hypothesis logs, failures, commands, seeds, and validation evidence.
The public @rorzio entry still has the older submitted score, **2.948526966725367**.
The improved local packing has not been submitted.
The [blind pilot](docs/pantheon-pilot.md) recovered known scores without target
coordinates and produced the latest verified improvement through native sampling.
The [latest strategy review](docs/oblique-strategy-review.md) records the archive
census, support geometry, a successful angular-repair control, and ranked next
experiments with explicit budgets and stopping criteria.

## Other Commands

```bash
uv run python run.py my_solution.json         # score a different file
uv run python run.py --save-plot packing.png   # save a visualization
uv run python run.py --visualize               # open plot in a window
uv run --extra lattice --extra constraint python -m pytest  # run tests
```

## Fast optimization loop

The offline recovery benchmark checks whether generated neighborhoods contain
known good packings, then asks the exact solver to recover a valid packing at
each recorded score. It makes no API calls and does not update `solution.json`.

```bash
uv run --extra lattice --extra constraint python search.py recovery \
  --output experiments/recovery/run-001 --require-recovery
```

Use a new output directory for each run. `--coverage-only` omits the solver;
`--seconds-per-case` bounds each solve. Four controls currently recover in about
11 seconds. Alignment uses the known targets, so this is a coverage and solver
benchmark, not blind discovery. The [method and recorded results](docs/recovery-benchmark.md)
explain the distinction and the missing physical deformation coverage it exposed.

Install the search dependencies with `uv sync --dev --python 3.13`. A C++17
compiler is required; the native kernels compile automatically into the OS
temporary directory on first use.

```bash
uv run python search.py benchmark
uv run python search.py polish solution.json
uv run python search.py lattice --input solution.json --steps 15 --flat-cuts --seconds 5 --rounds 8
uv run python search.py feasibility --seconds 120 --seed 91
uv run python search.py basins --seconds 60 --seed 11
uv run python search.py arena leaderboard
uv run python search.py arena check experiments/feasibility_best.json --save experiments/check.json
```

The strongest search uses reduced lattice bases to propose joint moves on the
six-decimal grid. Install its optional dependencies and run the preset:

```bash
uv run --extra lattice python search.py chase --seconds 120 --seed 992
```

It preserves a promising flat-contact branch, samples the shallow deformation
modes and coordinate frames, and checks candidates with the original validator.
Only valid improvements are written to `experiments/challenger.json`. The last
JSON line reports the best score and whether the goal was met. `chase --help`
lists the underlying controls; flags after the preset override its defaults.
The preset's goal is the recorded first-place score, so refresh the leaderboard
and override `--goal` if it changes.

`polish` solves the continuous problem using the allowed intersection area. Its
rounded output can be invalid: the printed validation result is essential.
`lattice` works directly in integer microunits and adds contact cuts after exact
checks. `feasibility` refines the input, searches different coordinate frames,
and saves a candidate only after the reference validator accepts a score below
`--target`. Its default target is just below the recorded first-place score;
check the live leaderboard before a new run. `basins` tries centroid-preserving
rotations, flips, cluster moves, and compression to discover different layouts.
Each recipe accepts `--help`; random searches expose a seed.

The geometry loop uses exact separating axes and Green's theorem integrals,
with analytic derivatives. A chord correction follows the server's inscribed
polygons as contacts move; measured area error is below 7e-12 near this packing.
On the development machine, all 105 polygon intersection estimates and their
Jacobian take about 35 microseconds; continuous refinement takes roughly
0.1–0.4 seconds. The fast enclosing-circle solve takes about 0.5 milliseconds.
`benchmark` measures the current machine, including the reference validator
separately. Final candidates still need the original validator and API checker.

The server API helper only reads the leaderboard and checks candidates. It
does not submit entries. `experiments/original.json` preserves the input
packing, and `experiments/api_baseline.json` records its server score.

## Reproducible research runs

Record a hypothesis before a bounded run. The wrapper saves its command, input
hashes, source snapshots, dependency versions, output, duration, and exit status;
it rejects an identical successful run unless `--repeat` is supplied.

```bash
uv run --extra lattice python scripts/research_run.py \
  --label centers-angles --hypothesis 'Search integer centers; round angles with 2-SAT' \
  --timeout 150 -- \
  .venv/bin/python scripts/experiment_position_projection.py --seconds 120 --seed 110440
```

New experimental recipes are `projection` (residual-feedback lattice search),
`positions` (28 integer center coordinates followed by angular 2-SAT), `family`
(curved continuous sections plus LLL), and `kbest` (inequality-aware beam).
`cpsat` needs `--extra constraint`, plus `--extra lattice` when rebuilding its
basis from `--input`. The separate quadratic experiment needs `--extra quadratic`.
Contact-graph and directed contact-release experiments need `--extra topology`
for NetworkX.
These methods are exploratory; the dated research log states which actually
produced a better verified packing. A continuous or linear-model score does not
establish a valid six-decimal solution.

The follow-up audit and new experiments are recorded in
[`experiments/2026-11-04/round2/README.md`](experiments/2026-11-04/round2/README.md).
It includes known-valid counterexamples to two old search filters, their repairs,
feasible neutral-angle walks, explicit topology surgery, and a C++ batch scorer
for overlaps plus minimum enclosing circles (about 21 microseconds per local
candidate in the recorded benchmark, with a convex-solver fallback).

```bash
uv run --extra lattice python search.py breakout --seconds 180 --seed 110458
uv run python search.py neutral --seconds 180 --seed 110459 --output experiments/neutral-new
uv run python search.py slack-plot --output experiments/contact-slack.png
```

These are research commands, not claims of a better score. Use a new output path
and `scripts/research_run.py` to preserve the exact command and source snapshot.

The [third research round](experiments/2026-11-04/round3/README.md) produced the
preceding **2.948526962953499** incumbent through a reflected packing and
curved-family continuation. The [blind pilot](docs/pantheon-pilot.md) produced
the current saved score.
It also adds a C++ batched lattice decoder, checking about 21,000–23,000 distinct
candidate geometries per second in the recorded runs. For a reproducible run:

```bash
uv run --extra lattice python scripts/research_run.py \
  --label native-global --hypothesis 'Test fresh global frames with full integer moves' \
  --timeout 330 -- .venv/bin/python scripts/experiment_batch_lattice.py \
  --input experiments/branch-3.141616-3.141592.json \
  --output experiments/native-global-new --seconds 300 --seed 110480 \
  --target 2.94852698
```
