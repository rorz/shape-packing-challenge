# CP-SAT research log

## 01 — 2026-11-04: hypothesis and setup

The previous reduced-basis MILP repeatedly returned similar invalid points.
Test a distinct discrete search engine with clause learning, integer contact
inequalities, warm starts, and retained nonlinear tangent cuts. This is a search
heuristic: these nonlinear tangent cuts are not a global exclusion proof.

Read `experiment_lll_milp.py`, `experiment_lll.py`, `polish.py`, and existing
experiment records before choosing this approach. OR-Tools was absent.
Installed it into the existing virtual environment with:

`UV_CACHE_DIR=/tmp/packing-uv-cache uv pip install --python .venv/bin/python ortools`

Primary API reference: https://developers.google.com/optimization/cp/cp_solver
All CP-SAT constraints use integers. Center variables will have much finer
resolution than the required placement lattice; geometric acceptance always
uses the native model followed by the reference validator and API.

## 02 — first three bounded experiments

Exact command, source hashes, captured stdout, and exit status are in `runs/`.

- `cpsat-projected`: prior projected near-solution, coefficients ±3, movement
  50 microunits, minimize worst violation, seed 110401, 120 seconds.
  Initial best violation 23.8717e-9; the solver repeatedly returns integer
  translations of that same geometry.
- `cpsat-radius`: old near-solution, hard overlap constraints, minimize radius,
  coefficients ±3, movement 50, seed 110402, 120 seconds. No feasible model
  solution in the first four 20-second slices.
- `cpsat-valid-warm`: rebuild BKZ basis at the actual valid incumbent, use it as
  the warm start; coefficients ±2, movement 50, seed 110403, 120 seconds.
  Correctly recovers the valid incumbent; a 20-second continuation proves the
  same linear radius optimum inside that coefficient box. This is not a proof
  about other bases or nonlinear geometry.

Implementation corrections: source-record filenames now avoid `Path.with_suffix`
truncating the timestamp suffix; added a valid radius-slack warm start. Inspection
of returned placements exposed duplicate global integer translations. Fix the
first piece's x/y displacement to zero in subsequent runs; any integer-grid
configuration has a translated representative in this gauge, though a bounded
basis coefficient box can still exclude that representative.

## 03 — bounded outcomes and next decision

All four CP-SAT runs completed normally (120 seconds each):

| Run | Outcome |
| --- | --- |
| projected | 6 rounds, no better than initial 23.8717e-9 normalized violation |
| radius | 6 rounds, no feasible integer solution found; no infeasibility proof |
| valid-warm | 8 rounds, valid original score only |
| gauge | ±8 coefficients, 200 microunits, seed 110404; 4 rounds, valid original score only |

The gauge-fixed run also returned one-microunit *angle* changes that leave the
MEC unchanged. Translation deduplication was necessary but insufficient: several
angular directions can move without improving the three supporting outer arcs.
Do not repeat these configurations. Next investment goes to residual-feedback
projection and curved constraints rather than more CP-SAT time on this model.
