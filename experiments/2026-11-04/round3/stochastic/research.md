# Stochastic nearest-plane research

## Hypothesis and algorithm

Deterministic nearest-plane target sampling has high duplicate rates. A separate native stochastic API records the deterministic residuals, samples one to three decisions using exponential-race weights favouring near-half residuals, and repeats the descent with those conditional alternatives. A configurable positive GSO-norm exponent biases selection away from domination by cheap angular rows. Pure translation rows alone are excluded; every physical angle remains represented and scored. This is a proposal heuristic, with no exact Klein/distribution claim. The default deterministic API is unchanged.

## Controls

4,100 zero-temperature comparisons match deterministic decoding/fpylll, including half ties and the non-idempotent3*pi seam. Repeated decoder seeds give bitwise-identical output; changed seeds alter4,052/4,096 proposals. Conditional back-substitution checks verify1–3 actual alternative decisions, and raw coordinates map to the reported geometry after exactly one canonicalization. Evidence is `decoder-check.json`.

## First60-second comparison

Equal frame/target seeds3110511 and independent decoder seed371911. Baseline and stochastic jobs run concurrently; all physical frames remain fresh, with exact quarter-turn basis reuse.

- Baseline: 1,150,732 deduplicated full geometry checks, 19177/s, 3,011 frames, 18 samples within25nm, 5829 within100nm. The feasible witness was the known old incumbent; no improvement.
- Stochastic (temperature.15, GSO exponent.25,1–3 flips): 1,757,290 checks, 29269/s, 1,119 frames, 10 within25nm, 5192 within100nm. No valid improvement. Mean squared GSO distance rose20.4%.
- Duplicate fraction improved81.3%→23.3%, but near-valid rate per wall time declined. Extra uniqueness is insufficient evidence for a long campaign.

## Tighter single-alternative refinement

The first policy spends too much distance and scores many more states per frame, reducing angular-grid phase coverage. Testing temperature.03 with exactly one alternative against a fresh matched baseline (main seed3110512, decoder seed371912). Added a separate center-diversity diagnostic; this never filters physical angle variants. Each comparison remains bounded60seconds and uses the widened1e-11 prefilter plus GEOS nearest-state audits and reference-valid elite retention. Exact commands/events are in each campaign's `steps.md` and `events.jsonl`.

## Tighter-policy result and stopping decision

At60seconds the single-flip policy produced1,265,102 full states and697,391 center states, versus baseline1,253,640 and694,516. Estimated samples within25nm were5 versus11; GEOS nearest violations16.093nm versus10.121nm. The GSO distance inflation fell to3%, but useful yield per wall time still declined. No150–300second campaigns were justified. All owned sampler runs stopped; the optional deterministic-compatible API and full negative evidence remain in `final-summary.json`.
