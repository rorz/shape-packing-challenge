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

The saved solution currently scores **2.948526966725367**, verified by the API
on 2026-10-31. First place remains **2.9485269238856113**, a gap of
**4.28398e-8**. The original packing is preserved in `experiments/original.json`;
validation evidence is in `experiments/api_best.json`. No entry was submitted.

## Other Commands

```bash
uv run python run.py my_solution.json         # score a different file
uv run python run.py --save-plot packing.png   # save a visualization
uv run python run.py --visualize               # open plot in a window
uv run pytest                                  # run tests
```

## Fast optimization loop

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
