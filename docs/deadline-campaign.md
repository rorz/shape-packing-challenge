# Prize deadline campaign, 2026-11-04

User authorization: execute the oblique strategy review now, with a submission
window of two hours. Start **14:44:07 UTC**; stated deadline **16:44:07 UTC**.
The user will submit the JSON themselves. Keep `solution.json` at the strongest
reference- and API-verified result, preserving every preceding candidate.
The target at start is 2.9485269238856113; local incumbent is
2.948526950912552. No public submission is delegated to these workers.

**Paused at the user's request, approximately15:13UTC.** All search processes
were stopped and their absence verified outside the shell sandbox. No new
API-verified improvement was found. `solution.json` retains its original hash
`cefac3368caae7f9dea0c4cdc0e87d869f3a9bd76ba75292b1cd4b7fff7c8fb9` and passed
reference validation again at2.948526950912552 during shutdown. It remains the
submission file for the user. There was no public submission or Git commit.

All workers use one numerical thread. Independent experiments run concurrently;
wall timings are observations under that workload, not isolated machine benchmarks.
Only root performs API checks and global promotion. Source snapshots and full
raw logs live under ignored `experiments/pantheon-pilot/deadline-*` directories.

## Archive completion — complete, negative

The census contains17 records in13 shortlisted center classes. Identical raw
files reduce this to16 input angular realizations. All8 D4 orientations at
widths ±3/±12 yield256 scheduled models. The completed run took9.455 seconds:
16 were covered by prior finite proofs and240 new models were INFEASIBLE.
No candidate. No wider search is inferred to be impossible.

Proof reuse checks actual scored placement sets, with new domains contained in
proved domains and old radius cap greater than or equal to the new cap. It
checks old domain hashes, acceptance guard and solver/oracle source hashes;
center identity is never used as an exclusion. Two focused tests pass.

```bash
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/experiment_archive_orbits.py \
  --output experiments/pantheon-pilot/deadline-archive-v1 \
  --seconds 300 --per-model 2 \
  --prior-report experiments/pantheon-pilot/angle-orbits-v1/report.json
```

[Compact ledger](deadline-archive-results.json); full evidence is in
`deadline-archive-v1/report.json`. Reproduction requires a new output directory
and the archived input files. The runtime stops on completion of the frozen
schedule; the five-minute budget is a ceiling.

## Parallel routes

- [Rounding-event traversal](rounding-events-run.md): first scheduler overfocused
  one origin; its result is not a three-origin exclusion. A single corrected
  comparison with origin quotas and native preview is in preparation.
- [Exact center hops](exact-center-hops-run.md): the initial gate found a real
  reference-valid new center class, absent from the frozen prior archive, at
  radius2.948527037883059. This is87billionths worse than the incumbent and is
  a relay, not an improvement. First-place tightening failed in its tested box.
  A360-second graph continuation keeps the radius cap anchored to the original
  incumbent+100billionths and tracks D4 novelty separately.
- [Joint fission](joint-fission-run.md): all1,152 trials completed in70.01seconds.
  Eight competitive outputs were all congruent to known geometry. No new seed
  qualified for integer probing; this schedule is closed.

## Support-conditioned completion — complete, negative

This tests the review's secondary proposal: enumerate subtarget boundary-center
triangles, fix their relative integer coordinates, and optimize the remaining
continuous packing jointly. Common translation remains free during optimization.
Re-anchor afterward, pin the three support centers in the exact integer model,
and permit all other pieces to choose local positions and angles.

The first run selects32 of the144 subtarget triangles in the ±2-microunit
neighborhood, ordered by normalized overlap deficit before radius. It has a
180-second total ceiling, with ±1 and ±3 angular exact repairs only when a
continuous completion is sufficiently promising. Continuous output is never
accepted as a submission. Two focused pinning/translation tests pass.

```bash
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/experiment_support_completion.py \
  --output experiments/pantheon-pilot/deadline-support-v1 \
  --seconds 180 --limit 32
```

All32 initial continuous completions were sufficiently promising; all64 exact
models were infeasible in16.87seconds total. A follow-up selected64 fresh
support triangles from the wider±8-microunit enumeration, explicitly excluding
the32 previously attempted offsets. It completed in34.24seconds without a
candidate. [Compact results](deadline-support-results.json) retain every offset,
continuous result and exact-domain status. These are finite local exclusions.

The [integer-flat branch follow-up](support-branches-run.md) imposed both nearest
integer flat-angle differences and their8neighbor combinations before integer
repair. All288 branch choices eventually received eligible continuous completions;
576 distinct exact domains were infeasible. Seven initially numerical failures
were resolved by one bounded warm restart, not relabeled as infeasibility.

## Higher-cost flat-angle branches — complete, negative

The existing map contains612 feasible continuously solved flat-angle branches.
The old broad campaigns considered222 with continuous radius cost below20
billionths and sampled with a strong remaining-budget power weighting. The
remaining strata contain135 branches at20–50,166 at50–100 and89 at100–126
billionths. Their additional continuous cost might buy better grid alignment;
that is a hypothesis, not evidence of better discrete packings.

A bounded equal-coverage native comparison will test those three strata for90
seconds each, using the correct constrained Jacobian stress. It will retain
reference-valid scores and exact overlap/enclosure frontier evidence. Repeating
the same low-cost distribution would not answer this question.

All390 fibers received equal-attempt coverage in their respective strata.
7,395,637 distinct geometry checks found no improvement. Best valid radii were
2.948527009207927 and2.948527045780615 in the first two strata; the third found
none. [Full summary](branch-strata-run.md) distinguishes explicit new equal
allocation from possible incidental visits by older unrestricted searches.

## Further measured results and unfinished work

- The [center-hop graph](exact-center-hops-run.md) completed in299.48seconds:
  four orientations representing two center classes, no better score, all32
  first-place tightening models infeasible.
- The corrected [event comparison](rounding-events-run.md) balanced all three
  origins. Both methods recovered only known valid scores. A later audit found
  off-center native-frame construction in some previews; the result does not
  establish that a corrected event search is ineffective. Replaying every
  recorded native batch and checking all permissive candidates found no missed
  improvement.
- [BKZ20/30/40](basis-strength-run.md) produced bit-identical bases and proposals
  on all three frozen control frames. The viability test stopped after11.86seconds.
- [Adaptive frame search](adaptive-frames-run.md) corrected native-frame centering,
  fixed common random draws and normalization scales, and checked every eligible
  proposal. Its adaptive arm finished240.03seconds/1,143frames without improvement.
  The random comparison was interrupted; its last atomic checkpoint had200frames
  in49.51seconds. No comparative conclusion is warranted from unequal completion.
- The [mixed physical-frame pool](physical-frame-pool-run.md) combines distinct
  fitted deformations rather than the old rotation-only phase pool. Its incumbent
  control passed in3.79seconds. The user stop interrupted the target search;
  it returnedUNKNOWN after4.01seconds. This remains unfinished, with no exclusion
  or candidate claim. Its source, domains and report are preserved.

Focused implementation tests passed before their runs. The full repository suite
was not rerun during shutdown; `git diff --check` passed. No further optimization
should start without the user's next instruction.
