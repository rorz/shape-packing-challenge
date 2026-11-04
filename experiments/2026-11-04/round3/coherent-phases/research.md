# Coherent exact phase submodels

## Cached diagonal benchmark

The mixed25-phase categorical model uses96million pair checks and permits each piece to choose a different physical phase. Restricting every piece to the same63-pose block gives25 separate exact models. Their diagonal blocks and all support coefficients are read from immutable caches; no pair geometry is regenerated.

The first10 phases completed in33.85seconds; the remaining15 completed in43.06seconds. All25 are exactly INFEASIBLE at leader−5pm:23 CP proofs plus2 empty-domain proofs from exact arc/support propagation. The all1,454-cut external known-valid control passed unmasked. This proves only the same-phase diagonal domains, not the larger mixed-phase union.

Each phase retains original-index maps and replayable pruning traces. Source block order is0,-25,+25,-50,+50,...,-300,+300microradians. Commands and full results are in `benchmark10/` and `remaining15/`; the runner is `experiment_coherent_phases.py`.

## Fresh ensemble

The cached benchmark demonstrates low per-phase cost. A separate300-second ensemble now alternates full-precision original and mirror seeds, rotates them globally, and adds a random subgrid translation. Every rounded base is frozen and labelled with its actual validity and overlap errors. Invalid bases are not positive controls.

Each phase contains all per-pieceXYZ±1/theta±3 poses with a free enclosing center. Integer support-only pruning precedes exact pair construction; pair relations use the frozen native loss band with GEOS completion, Hamming interval encoding and55 balanced support cuts. Each model has an8-second deadline within the total budget. Full source copies, hashes, continuous/raw base snapshots, angles/translations and RNG seeds are retained under `fresh300/`.

The original source in this live batch has continuous radius2.948526554729897; mirror2.948526549416161. Root subsequently prepared a free-flat original source at2.948526549441705. It is reserved for a separately labelled later comparison, preserving this batch's provenance.
