# Exact center-hop gate, 2026-11-04

The deadline campaign authorizes this offline experiment. It asks whether the
current valid packing has a different valid center arrangement reachable by
coordinated local moves at radius barriers 50 or 100 billionths above its score.
It does not count angle changes or a common integer translation as a hop.
Rotations/reflections remain distinct angular-grid search states.

`scripts/experiment_exact_center_hops.py` privately extends the lazy finite
solver; shared production solver source is unchanged. Relative integer XY
coordinates are linked to pose choices by element constraints. Every blocked
center vector is excluded by a reified disjunction over all 28 relative
coordinates. Both directions of each equality are enforced. Every accepted
packing passes the original reference validator before its deadline.

The gate schedule is angular widths ±1 then ±3, relay barriers +50 then +100
billionths, and all eight D4 orientations. Translation-union XY width one covers
every common subgrid translation, with conservative exact-grid endpoints;
the orbit helper supplies seam-safe angular choices. There is a 120-second
overall gate budget and a five-second cap per finite model. The first valid
center hop ends the gate, then a separate 30-second local tightening attempt
targets `2.9485269238806113`. A hop can revisit a center seen in older campaigns:
it establishes a transition from this start, not historical novelty or a win.

```bash
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/experiment_exact_center_hops.py \
  --input solution.json \
  --output experiments/pantheon-pilot/deadline-hops-v1 \
  --seconds 120 --per-model 5 --tighten-seconds 30 --seed 3111601
```

Six focused tests passed in 1.70 seconds, including a real valid singleton
control, exclusion of its own centers, translation invariance, joint blocking
of visited center states and preservation of special angle aliases. A finite
infeasibility result excludes only its explicit domains, blocked vectors and
radius threshold. Deadlines are not infeasibility results. No API call or global
promotion is made by this script. Raw output, source snapshots and exact domains
are kept in the ignored run directory.

## Initial result and authorized continuation

The gate passed in **6.29 seconds**. All eight width-one angular models at the
+50-billionth barrier exhausted. The identity orientation at +100 produced a
reference-valid new center arrangement at **2.948527037883059**, 86.970507
billionths above the frozen incumbent. It is saved as `deadline-hops-v1/hop.json`.
Its initial first-place tightening model exhausted in another 1.78 seconds.
This is a useful valid transition, not a score improvement.

The parent authorized a 360-second graph continuation. Both +50 and +100 relay
barriers remain anchored to the **frozen global incumbent**, never the latest
node. A beam of at most 16 nodes alternates best-score expansions with a
least-visited exploration turn; a new center receives its own first-place
tightening immediately. Full domains and orientations remain distinct. A
cached exclusion is reused only for identical domains, a no-looser target,
and a blocked-center set containing every center excluded in the old proof.

```bash
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/experiment_exact_center_hops.py --graph \
  --input solution.json \
  --seed-input experiments/pantheon-pilot/deadline-hops-v1/hop.json \
  --output experiments/pantheon-pilot/deadline-hops-graph-v1 \
  --seconds 360 --per-model 4 --tighten-seconds 5 --beam-size 16 --seed 3111701
```

## Graph result

The graph queue finished in **299.48 seconds**, before its 360-second cap.
It retained four oriented valid center arrangements representing **two D4 center
classes**: the incumbent class and the new hop class. Subsequent transitions
recovered reflected versions of those two classes. There was no better score.

The 130 hop models comprised 128 finite exclusions and two accepted valid
transitions. All 32 first-place tightening models were infeasible. These results
exclude only the recorded local domains and radius barriers; they do not exclude
wider angular neighborhoods, larger coordinated center steps or a larger radius
barrier. The graph reached no beam eviction or timeout. The initial hop's center
class was absent, under both labeled and D4 keys, from the previously frozen
10,016-source archive. That is novelty against recorded history, not a global
novelty claim.

Relative to a common (-1, 0) integer-microunit translation, the initial hop moves
center 12 by (+3, +1) and center 14 by (+1, +2). Angles 1, 12 and 14 increase by
one microradian; angles 2 and 5 decrease by one. These coordinated changes explain
the valid hop that the earlier angular walk did not produce.

[Compact evidence](exact-center-hops-results.json) preserves every accepted raw
packing, independent reference scores, prior-archive novelty checks, source
hashes and all model domain hashes/statuses. The full exact domains and source
snapshots remain ignored under their immutable run directories. Seven focused
tests now pass in 2.04 seconds, including a graph-beam regression and the fixed
global relay-cap policy.
