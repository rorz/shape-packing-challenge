# Fresh coherent phase CSPs

Fresh global rotations and subgrid translations of full-precision sources. Each phase retains all per-pieceXYZ±1/theta±3 choices, with free enclosing center, exact pair relations, Hamming interval encoding and55 balanced support inequalities. Invalid rounded bases remain explicit; they are not positive controls. Support-only integer pruning precedes exact pair preparation. Each finite model has a separate deadline within the global budget.

```text
.venv/bin/python scripts/experiment_fresh_phase_csp.py --input experiments/branch-3.141616-3.141592.json --input experiments/2026-11-04/round2/topology/continuous-best.json --output experiments/2026-11-04/round3/coherent-phases/fresh300 --seconds 300 --per-phase 8 --seed 3110721
```
