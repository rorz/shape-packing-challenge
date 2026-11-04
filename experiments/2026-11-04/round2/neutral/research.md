# Native-feasible neutral-network research

## Audit and distinct hypothesis

Earlier strict hillclimbing screened out every proposal without an immediate radius decrease. Existing violation hillclimbers and tempering walked mainly through invalid states and treated feasible plateaus as a common zero-energy point. This experiment retains actual valid angle configurations, their different contact-slack vectors, and a portfolio of feasible relays. Exact translations are quotiented; angle changes remain physical.

The driving objective is the feasibility of a *future* center move. Current-state feasibility is a hard native GEOS condition. Crossover considers all 32,768 binary angle hybrids of two valid parents, using exact native current compatibility as a hard barrier and future-hop violation as the optimization cost.

## Smoke test

Command: `.venv/bin/python scripts/experiment_neutral_network.py --output experiments/2026-11-04/round2/neutral/smoke --seconds 20 --seed 110421`

Outcome: 5,028 proposals; 73 natively valid proposals; 64 accepted neutral-angle states; 335 global hybrid solves; no center hop and no score improvement. Correction after checking the frozen start: worst overlap stayed at 9.9962047e-7. Native pair (2,4) decreased by 2.80e-9 while pair (1,5) increased by 7.56e-9, with every pair still valid and radius unchanged. This demonstrates redistribution, not an improved worst overlap.

## Follow-up changes and bounded run

- Replaced scalar hop scoring with the independently checked native batch area/MEC helper; accepted states still require 4096-point GEOS intersections.
- Preserve the best improving hop longer instead of randomly switching among difficult hops.
- Include bounded-radius valid relays even when invalid improving hops exist; the smoke had inadvertently excluded them from the target pool.
- Bias angular steps toward a selected hop's blocked contact while retaining random exploration and paired moves.
- Fix piece 0's center gauge in hop vectors; retain all physical angle states.

Next command and every accepted transition are recorded in the main run's `steps.md` and `events.jsonl`.

## Main valid-state walk: result and a distinct missing edge type

The 180-second run made 96,492 proposals and accepted 486 distinct natively valid neutral-angle states (6,432 global hybrid solves). No center hop or radius improvement occurred. This is evidence that slack redistribution exists, but did not connect to a new center configuration under the tested fixed-angle hop edges.

The next edge definition changes: after proposing center translations, solve a small discrete angular CSP at those *new* centers. Its final endpoint must be natively valid; orientations that would be invalid before the simultaneous center move are allowed as part of that endpoint. The search still never accepts an invalid state. Enumerating piece 14's boundary-angle choices and checking an enclosure of all other full disks avoids fixing an arbitrary wrong MEC center.

The follow-up is bounded at 180 seconds and begins from the natively valid neutral portfolio endpoint. It accepts neutral center relays and occasionally worse valid radii within the same 3e-7 cap, retaining the best verified solution separately.

## Final native-validated result

The smoke, main, and conditional walks accepted 1046 transitions, representing 505 distinct native-valid configurations after adding the frontier endpoints, and 2 distinct relative-center configurations. All accepted native intersections stayed below 1e-6; maximum logged area was 9.999788916302039e-07. No score improvement was found.

The broad joint frontier discovered a valid worse-radius relay by moving piece 8 one x micro-unit and adjusting several angles. Its reference score is 2.94852724440804, within the fixed global cap 2.948527266725367. A dedicated 60-second expansion found only a valid return to the incumbent centers within its tested prefix. The raw relay run's old `distinct_centers=1` field counted enqueued/expanded centers because `--max-centers=1`; the independent endpoint audit establishes two observed center configurations. The reporter is corrected for future runs.

No first-place candidate was found, no shared solution was overwritten, and all runners have exited. Exact endpoint checks, counts, explicit cutoffs, and paths are in `final-summary.json`.
