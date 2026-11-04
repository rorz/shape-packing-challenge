# Batched global lattice proposals — round 3

## Hypothesis and distinct objective

The existing global-frame LLL sampler evaluates one Babai completion and one geometry solve at a time, at roughly 1,800 proposals/s. The independently checked native batch area/MEC helper can score around 50,000 states/s. This task targets the orchestration gap: batch target generation, precomputed canonical-to-GSO transforms, native conditional rounding/back-substitution, and batched full nonlinear scoring.

All 45 physical placement dimensions remain present. Only exact integer translations are quotiented during deduplication. Opposed active flat pairs are detected from geometry rather than hardcoded indices. Frames remain globally random, and proposal counts are separated from unique translation-normalized configurations.

Every candidate promotion requires the reference validator and a server check. No shared solution file will be overwritten.

## Initial audit

- Reviewed `experiment_lll.py`, `breakout_geometry.py`, and the geometry-based `flat_contact_pairs` detector.
- Candidate inputs: original continuous branch; requested new mirror seed (path clarification pending).
- Check fpylll's exact tie convention before implementing native rounding.

## Native decoder validation and measured bottleneck

- Native nearest-even ties match fpylll for positive and negative half-integers.
- 8,192 proposals over four original and four mirror frames matched fpylll exactly, before and after the guarded near-half fallback. One conservative ambiguity flag was patched; there were zero raw mismatches.
- Translation-equivalent keys are dropped; a one-microradian physical angle change is retained. A known valid packing survives native output canonicalization and native certified MEC scoring.
- Evidence: `decoder-check.json` (seed 3110499).
- Direct old Babai+multiply microbenchmark: 329.95 microseconds/proposal. Native batch decode: 1.65 microseconds/proposal (~200x faster).
- Sustained 10-second profiles: batch 512 gives 10,652 unique full geometry checks/s; batch 2,048 gives 14,479/s. Gross proposals are not reported as unique: the 2,048 profile dropped 758,238 exact-translation-normalized duplicate keys from 903,168 generated points.

## Reuse the expensive basis across exact lattice quarter turns

Signed x/y coordinate permutations preserve the embedding Gram matrix. The same reduced GSO therefore serves four exact 90-degree physical frames. Each frame still gets its own wrapped angles, six-decimal anchor, and flat-angle offset; all 45 dimensions remain decoded. This tests different angular-grid phases without repeating LLL/BKZ work.

Independent checks used an explicitly rotated integer embedding and fpylll GSO. All 3,072 quarter-turn proposals matched; raw wrapped-angle rounding errors stayed within half a microradian. Evidence: `quarter-turn-check.json`.

## Sustained campaigns

Two independent 300-second, single-thread campaigns now start: the original negative-flat branch and the full-precision mirror seed. Both use global random base frames with four exact quarter-turn frames per basis, 2,048 targets/frame, auto-detected opposed flat pairs, target radius 2.94852695, and first-place goal 2.9485269238. Only reference-valid improvements are written as candidates; API confirmation controls best-score promotion and the shared first-place stop marker.

- Root requested the original full-45 campaign at the looser radius 2.94852698 to complement its tied-angle campaign at 2.94852695. The just-started original .52695 run was interrupted and retained under `original/`; its replacement uses a new `original-r98/` output directory so manifests/events are not mixed. Mirror .52695 continues unchanged.

## Completed initial campaigns and archive correction

- Mirror: 300.013 seconds, 38,750,208 generated proposals, 6,267,526 deduplicated full area/MEC evaluations (20,891/s), 18,921 physical frames and 4,731 LLL/BKZ bases. No incumbent improvement. Best combined violation at the search radius was 15.140 nm.
- Original at radius 2.94852698: 300.013 seconds, 36,814,848 proposals, 6,878,983 full evaluations (22,929/s), 17,976 frames and 4,494 bases. No incumbent improvement. All MECs received native certificates; neither run needed the convex fallback.
- Reproducible decoder tests now include the three independently constructed quarter-turn embeddings (3,072 comparisons), alongside the original 8,192 comparisons. All match fpylll. The promoted mirror incumbent at score 2.948526962953499 also passes the native positive-witness check, with radius error 1.2e-13.
- Root requested preserving feasible starts slightly above the current incumbent. Added a fixed-cap, maximum-ten elite archive, deduplicating relative xy states. Each reference-valid elite has JSON plus a compressed origin/move sidecar; the sidecar's `basis` is in latent base-frame coordinates, while `moves` and `continuous_frame` are in its physical quarter-turn frame.
- Started `original-r98-elites/` with the same seed 3110413 and 300-second bound to recapture the earlier valid point. This is a replay for evidence retention, rather than an independent exploration sample. The fixed cap is 2.9485270629534988, the starting incumbent plus 1e-7; later elite scores cannot raise that cap.
- Early replay recovered the old incumbent at 2.9485269667253675 (`elite-00001.json`). Thus the earlier negative combined violation was a rediscovery of that packing; it did not establish a new 2.9485269688-score configuration. Other recovered relative xy states score 2.9485270379518154 and 2.9485270489766346.
- Added an elite archive serialization/translation-dedup regression to `experiment_batch_lattice_check.py`; it passes.
- Root's estimator audit identified a screening limitation: the original 2e-12 native-area prefilter is narrower than the documented approximately 7e-12 estimator discrepancy. Those campaigns did not exhaust all potentially GEOS-valid candidates. Widened the batch sampler and the upcoming native breakout follow-up to 1e-11, retaining GEOS as the promotion validator. This wider empirical guard is not claimed as a rigorous approximation bound.
- Preserved raw decoded input separately from native canonical geometry. `Frame.raw` reconstructs the anchor plus lattice move and quotients xy translation; `checked` and output serialization now consume that raw input. A 3*pi seam regression demonstrates why exactly one input canonicalization matters. Ordinary small moves around canonical frames need not encounter this edge, and no archived elite failed; this fixes a latent robustness issue rather than an observed cause of the current search misses.
- Elite provenance/D4 audit selected `elite-00003.json` at radius 2.948527013256507 as the closest center state distinct from both known incumbents. Its saved coefficient/move reconstruction is exact; a bounded 180-second native breakout with the matching physical basis starts under `elite-followup/`. The complete command, reference checks, and equivalence exclusions are retained there.

## Relay outcome

The180-second matching-basis breakout completed6,752,859 native nonlinear evaluations over14,034 iterations, accepting6,530 distinct search states with371 distinct center states. These counts include infeasible states. It produced no reference-candidate checks and no incumbent improvement; its best combined violation remained89.457nm at the first-place target. All owned searches are stopped. Structured campaign, check, archive, and limitation evidence is in `final-summary.json`.
