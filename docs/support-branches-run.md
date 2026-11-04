# Integer flat branches during support-conditioned completion

The deadline campaign's first support-triangle experiment produced 32
continuous completions but no integer solution. This follow-up imposes the two
integer flat differences **before** repacking: zero-based pairs (1, 14) and
(4, 8), alongside four relative-XY constraints for supports 5, 10 and 14.
Common translation remains free and all 15 pieces participate in optimization.

For each frozen continuous completion, test the nearest pair of integer
microradian differences and all eight neighbors in its ±1 square. The schedule
tries the nearest choice for all 32 inputs first, then each neighboring offset:
288 continuous solves at most, within a 240-second wall budget. Inputs, source
hashes and full-precision completions are retained. No shared solver is edited.

Only completions with native enclosing radius below the first-place target,
model violation at most 1e-9, and support/flat equality errors at most 1e-10
enter exact repair. These are numerical proposal criteria, not certified
continuous feasibility. GEOS pair areas are separately recorded before rounding.
The existing `pinned_domains` generator then supplies angular widths ±1 and ±3,
with every other center movable in the translation-union domain. Domain hashes
avoid duplicate work; UNKNOWN remains unresolved. A final candidate must pass
the original reference validator. No API call or global promotion is performed.

```bash
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/experiment_support_branches.py \
  --source-directory experiments/pantheon-pilot/deadline-support-v1 \
  --output experiments/pantheon-pilot/deadline-support-branches-v1 \
  --seconds 240 --seed 3111801
```

Three focused tests passed in 0.75 seconds: independence of the six equalities,
preservation of common translation, and the complete nearest-plus-eight integer
branch schedule. The continuous solver minimizes radius with SLSQP under the
existing analytic overlap/enclosure constraints; integer validation remains
entirely downstream.

## Result and one numerical retry

All 288 original branch choices completed in **136.99 seconds**. Of those,
281 met the continuous proposal criteria; all 562 distinct exact integer models
were infeasible. The seven remaining SLSQP results missed the 1e-9 model-violation
cutoff. A single warm restart of those same branches (300 iteration ceiling)
resolved every numerical failure in **3.24 seconds**, and its 14 distinct exact
models also exhausted. The retry introduced no new branches or radius targets.

```bash
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/experiment_support_branches.py \
  --source-directory experiments/pantheon-pilot/deadline-support-branches-v1 \
  --output experiments/pantheon-pilot/deadline-support-branches-retry-v1 \
  --seconds 60 --seed 3111802 --retry-failed
```

Combined: **140.23 seconds**, every one of the 32 × 9 branch choices has an
eligible completion, and **576 distinct finite integer domains are infeasible**.
No candidate improved the score. The equalities are affordable continuously;
fractional flat differences alone do not explain the failure to quantize these
support-conditioned packings. This does not exclude different completions of a
fixed triangle, larger integer neighborhoods, or other support triangles.

[Compact results](support-branches-results.json) retain input/source hashes,
each branch, continuous diagnostics, separately measured GEOS areas and every
finite domain hash/status. The original source snapshot remains intact; the
retry flag was added only after the first run completed. Full-precision rows and
exact domains remain under the ignored immutable run directories.
