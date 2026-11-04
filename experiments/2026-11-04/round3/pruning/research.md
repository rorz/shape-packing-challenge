# Exact finite-domain propagation

## Hypothesis

Interval encoding preserves the allowed relation but can expose less propagation to CP-SAT than explicit tables. Restore exact supports before solving using packed-bitset arc consistency, then alternate with integer support-cut bounds until a fixed point.

For a supplied inequality `sum(coeff_i[choice_i]) <= rhs`, eliminate a value only when its integer coefficient plus the current minimum coefficients of all other involved variables exceeds rhs. Pair elimination requires no remaining support in the original exact relation. Original pose IDs are never changed. Every elimination records its reason and is replayed independently against the contemporaneous domain state.

Twenty-one exhaustive small-model tests pass, including negative coefficients, chained pair/cut pruning, interior relation holes and preservation of every feasible assignment.

## Phase-pool results

Immutable input digest: `73c3dc5f588885280f440cff3a1aba290d5ec2dadcc9a2ef1ff45240038e045d`. Fifteen domains contain1,575 poses each;30 exact pair relations are packed in both directions.

-79 cuts at old target2.9485269238:1,578 poses removed,1,480 via pair supports and98 via integer cuts;19milliseconds propagation after0.62seconds loading. Fixed point, no infeasibility.
-79+1,375 cuts at NEW target2.9485269238806113 (leader−5pm):1,598 poses removed,1,480 via pairs and118 via cuts;249milliseconds propagation after1.43seconds loading. Every elimination replayed exactly. Fixed point, no infeasibility.

The older masks use a stricter target and must not be applied to the weaker new target. New masks are in `phase1454/original-index-masks.npz`, with exact target/digest in its sibling `summary.json`. Goal-specific masks must not be applied to a known-valid incumbent control at its larger radius. Arena's isolated runner now supports target-only masks and checks the supplied threshold.

The stronger cut bundle added only20 unary eliminations globally; this is measured evidence that these local bounds alone do not resolve the mixed phase model. Coherent phase restriction is a separate, stronger conditioning step.
