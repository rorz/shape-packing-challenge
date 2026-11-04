# Recovery benchmark

The benchmark asks whether our search representation can contain known good
solutions, then whether the exact solver can recover their score. It is offline
and leaves the incumbent and paused optimization campaigns alone.

```bash
uv run --extra lattice --extra constraint python search.py recovery \
  --output experiments/recovery/run-001 --require-recovery
```

The initial native build needs a C++17 compiler. Each output directory must be
new: the benchmark refuses to overwrite earlier evidence. `--coverage-only`
performs the approximately two-second coverage audit. Each solver call has a
15-second budget by default; change it with `--seconds-per-case`. A timeout is
reported separately from an absent target or proven finite infeasibility.
`--require-recovery` exits nonzero unless every case is recovered. Without that
flag a completed diagnostic can report negative results with exit zero.

## What is measured

Four immutable reference-valid fixtures are recorded with hashes and provenance
in [`tests/fixtures/recovery/manifest.json`](../tests/fixtures/recovery/manifest.json).
The incumbent supplies calibration; random-D, neighbor-G and fresh-baseline are
held out from width calibration. All required data is in that small fixture
directory; the ignored research archive is not a runtime dependency.

For each target the benchmark compares three alignments:

1. Rigid rotation and translation of the continuous seed, conditioned on the
   target's flat-angle branch but with both physical deformation coordinates zero.
2. Joint fitting of rotation, translation and both physical coordinates, using
   the existing nonlinear `Manifold.section()` generator.
3. Rigid alignment of the historical saved origin, as a provenance diagnostic.

It rounds each frame using the production canonicalizer, then calls the actual
`experiment_lazy_phases.make_domains()` generator. Exact integer membership
checks identify every omitted piece. Raw input angles are normalized once.
The production profile permits XY offsets of ±1 grid unit and angle offsets of
±12 microradians: 225 poses per piece, before seam deduplication. Calibrated
widths can only increase based on calibration cases, never the held-out cases.
Oversized domains are reported, never truncated.

Only covered cases proceed to the lazy exact solver. The known solution's rows
and membership assignment are **not** supplied as hints or added to the domain.
The solver retains its ordinary index-zero rounded-frame hints.
The solver returns a raw submission that must pass the reference polygon/MEC
validator at the recorded score plus a 1e-11 benchmark guard and the solver's
2e-12 acceptance guard. Target membership, solver success, exact pose recovery,
and center recovery modulo common integer translation are reported separately.

This is deliberately **target-assisted alignment**. It establishes useful
representation and local solver recovery; it does not establish that a blind
outer search can find these five alignment parameters. Fits use local least
squares, so failure at one fitted frame does not exclude every possible fit or
prove a minimum necessary box width. A separate control accepting a manually
supplied singleton incumbent would not test neighborhood coverage.

## Recorded result, 2026-11-04

All four family-aligned neighborhoods contained their known target and all four
solver calls, without target-assignment hints, recovered a valid packing with exactly the recorded
radius. The centers matched their respective controls; some non-supporting
angles differed. The existing ±1/±12 profile passed without widening. In fact,
the fitted targets each needed only ±1 in all three coordinates.

| Control | Rigid XY width needed | Rigid angle width needed | Family XY / angle widths | Recovered radius |
|---|---:|---:|---:|---:|
| Incumbent (calibration) | 210 | 385 | 1 / 1 | 2.948526962953499 |
| Random-D (held out) | 109 | 229 | 1 / 1 | 2.9485269867883153 |
| Neighbor-G (held out) | 102 | 167 | 1 / 1 | 2.9485269864331265 |
| Fresh-baseline (held out) | 232 | 404 | 1 / 1 | 2.9485270133102732 |

Widths are integer micro-units about the particular fitted frame, not global
minimum widths. Total runtime was approximately 11 seconds on the development
machine. The compact [machine-readable record](recovery-benchmark-results.json)
preserves source/fixture hashes, parameters, fit diagnostics, scores and timings.
Full local frames, domains and solver traces remain in the ignored run directory.

The evidence supports varying the two physical deformation coordinates before
spending effort on wider local boxes. Historical lattice moves displaced the
packing far along those modes from their saved continuous origins; refitting
the modes absorbs that displacement. It does not show that first place lies in
this family. No new leaderboard improvement was produced or submitted here.

## Verification and archive policy

```bash
uv run --extra lattice --extra constraint python -m pytest
git diff --check
```

Regressions cover held-out calibration separation, absence of target injection,
angle wrap boundaries, rigid translation invariance, a real incumbent whose
coverage needs the physical modes, fixture integrity, offline solver behavior,
and final enclosure validation. The benchmark also runs from the commit-eligible
tree without the ignored experiment archive.

The workspace suite passes 189 tests. The clean export passes 186, with three
optional large-archive replay tests skipped because those inputs are not committed.

Source, tests, fixture JSON, these notes and the compact result belong in Git.
Bulk experiment output, NumPy tables, copied source snapshots, native binaries,
plots and `.DS_Store` files remain local. The dated archive has an explicit
allowlist for selected evidence and continuous seeds; see
[`experiments/README.md`](../experiments/README.md).

## Next experiment: independent strategy review

Three independent reviews (geometry, integer search, and adversarial evaluation)
agreed that the successful target-assisted benchmark does not justify a broad
random exact search. The old native lattice campaigns already varied the two
physical modes. Productive frame discovery remains untested by this benchmark.
The following is a proposal only; no optimization campaign was resumed.

1. Cover every fractional XY translation in one domain union. For each piece,
   union the rounded center choices over one grid period, plus the existing
   local offsets. Typically this gives four values per coordinate. Closed
   phase endpoints and ties-to-even can require five: derive conservative
   bounds from production rounding and test containment against actual common
   translations. Independent choices enlarge the search space safely because
   every accepted packing still passes the exact validator. This removes two
   outer variables, leaving two physical modes and global rotation.
2. Replay the controls with angle widths ±1, ±3 and ±12. All four fitted
   controls require only ±1. With the typical translation union, ±1 gives 48
   poses per piece versus the existing 225, approximately 22 times fewer raw
   pair combinations. This is not a measured runtime speedup. Retain an
   explicit wider-angle budget; narrow infeasibility is not a wider exclusion.
3. Run a blind, equal-budget pilot on the three held-out saved origins. Give
   both methods only the origin, branch and score cap; withhold target poses
   and fitted parameters from the search process. Compare the existing native
   sampler against a candidate funnel that uses native proposals to select
   fresh frames, then invokes the smaller exact models. Use two minutes per
   method per origin (12 minutes total), equal CPU limits, fixed seeds and
   source hashes. Record time to a qualifying score, best validated radius,
   distinct center configurations and time spent in exact solving.

Proceed to a first-place campaign only if the challenger demonstrates useful
blind recovery and an advantage over the native baseline. More infeasibility
proofs, cheaper individual models or repeated recovery of known centers alone
are insufficient. The next possible extension is traversal across rounding
boundaries with incremental pair-cache reuse, after the pilot establishes value.
Avoid repairing an archived witness again when its new domains are contained
in a previously exhausted box; compare actual domain contents and thresholds.
