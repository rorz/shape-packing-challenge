# First-place goal continuation

The goal remains **a server-verified result below first place**. Tools, speedups,
tests and new seeds are progress, not completion. The previous goal turn was
progress: it fixed actual false rejections, found a distinct mirror topology,
and preserved the evidence. It did not produce a winning score.

At start: incumbent **2.948526966725367**, leader **2.9485269238856113**;
`leaderboard-start.json` is the fresh API record. The promotion below happened
later in this round.

## Verified improvement

Mirror-family continuation produced **2.948526962953499**, independently checked
with the API and the reference validator, then promoted to `solution.json` and
`experiments/best.json`. This improves the former incumbent by
**3.771867707e-9**, leaving **3.906788759e-8** to the observed leader. This is
progress, not first place. `promotion.json` records the source and SHA-256;
`candidate-api.json` is the independent API result. `previous-incumbent.json`
preserves the former submission.

The winning section within this campaign has mirror topology, physical mode
coordinates (-0.00014, -0.00014), and raw flat differences
(3.141570, 3.141593). See `mirror/campaign-b/sections.json`, section 32.

## Initial routes (see latest results below)

- **Batch global lattice search:** replace Python-per-proposal Babai work with
  batched/native decoding and native overlap/MEC evaluation. Verify output against
  fpylll, then spend the gain on genuinely different global frames.
- **Conflict master:** exact finite placement compatibility tables and free-center
  witnesses, directed at the new mirrored candidate with five small overlaps.
- **Mirror family:** follow the two-mode curved contact family in the new geometry,
  deliberately cover all raw-angle wrapping cells and quantized flat branches.
- **Pinned frame:** choose a global rotation that puts the endpoint-sensitive
  angle exactly on-grid, fix it, and search the other44 integer coordinates. This
  aims to remove its rounding kink and large implicit rotation curvature. It is
  a restricted proposal family, not an exclusion argument about other packings.

Every root run is bounded and recorded in `runs/` with input/source hashes,
snapshots, parameters and output. Agent-owned logs are under `batch/`,
`conflicts/` and `mirror/`. Final candidates must pass the original polygon
validator and remote API before promotion. Do not overwrite the incumbent on
the strength of approximate geometry or a lower radius with overlaps.

## Initial pinned-frame check

Seed110470,12seconds,new mirror input:23,248proposals across78frames,
best remaining violation104.97e-9, no valid improvement. The automatically
identified endpoint piece is12, leaving44integer dimensions. Two300-second
campaigns compare original and mirrored geometries at target2.94852698.

`experiment_pinned_frame.py` checkpoints have44basis vectors, not45. Old scripts
that assume a45×45 basis must not consume them without adapting the dimension.

## Pinned-frame result: stop this route

Both 300-second runs finished. Original: 574,715 proposals, 1,916 frames,
best remaining violation 64.51e-9. Mirror: 550,625 proposals, 1,836 frames,
best violation 67.03e-9. Neither improved the incumbent. Fixing the endpoint
angle restricts useful coordinated moves; it did not justify more compute.

## Exact flat-angle lattice

`experiment_tied_flats.py` ties both flat-pair angular increments, leaving 43
integer coordinates. Input sections must already have exact six-decimal raw
angle differences. Sampling is restricted to rotation intervals preserving
those raw branches. This is a proposal restriction, never an exclusion proof.

The initial 12-second scalar smoke tested 22,888 proposals across 77 frames,
best violation 43.36e-9. `experiment_tied_batch.py` uses the validated native
decoder and keeps raw input angles separate from once-canonicalized geometry.
Its four-cell smoke tested 1,153,024 proposals / 180,640 unique geometries in
12 seconds, reaching 21.71e-9. A 300-second campaign follows at target
2.94852698. Critical transform and wrapping tests passed.

## Reflection inventory without a subset-size cutoff

The former inventory covered connected subsets of at most four pieces at 45
axes. A graph formulation removes that restriction: reflecting both members of
a pair preserves overlap; reflecting exactly one has the same overlap in either
direction. Therefore every incompatible cross pair forces equal reflection
bits. Every allowable subset is a union of connected components of this graph.

`experiment_reflection_components.py` tests every subset size implicitly at
201,035 axes, including axes exchanging any pair of endpoint/radial features.
The native area calculation is a permissive discovery screen; GEOS checks
candidate seeds. This does not prove completeness over continuous axis angles.
Tests compare the graph result against exhaustive subsets and check the
reflection identities against actual polygons.

The full finite scan finished in **26.1 seconds**: 201,035 axes, five disconnected
conflict graphs, five subset representatives, all matching already known
geometry invariants. No new seed. Do not repeat this grid or restore the old
four-piece size cutoff; a further reflection search needs a new axis-generation
or deformation argument.

## Full native tied-angle campaign

Seed 110475 completed 300 seconds: 28,708,864 proposals, **4,663,654 distinct
geometry checks**, 14,018 frames. No valid improvement; nearest combined
violation was 6.24e-9 at a radius slightly above the incumbent. The next campaign
uses the section that actually produced the improvement, strengthens the
movement regularization, and archives valid states even when they do not beat
the global incumbent. Those states can be starts for another local search.

## Exact categorical enclosure cuts

The conflict master first proved three small finite placement domains overlap
incompatible. Adding local one-unit XYZ corrections enabled overlap-free leaves,
but 270 of them failed enclosure. Each failure involved only three supporting
shapes. Additive support-function cuts then exhausted the **same** finite domain
in 0.388 seconds after two leaves and three cuts, compared with 125 seconds of
tuple exclusions without exhaustion. These are valid free-center bounds with
conservative arithmetic guards. They prove only the supplied finite domains
infeasible. A fresh model now targets the improved mirror incumbent.

The new-incumbent model also exhausted: 191,623 direct GEOS entries took 56.58
seconds; a forced incumbent positive control passed; the first-place target
was infeasible after three leaves and four cuts in 0.75 seconds. Domains were
the **union** of the base XYZ ±1 box and ten signed complete LLL moves. They did
not include local corrections around every displaced state. The next model
tests that wider sum of domains, with certified true-area bounds accelerating
table construction.

## Further completed comparisons

- Full 45-coordinate native runs checked 6.268 million mirror and 6.879 million
  original geometries in 300 seconds each. Neither improved the incumbent.
- The 43-coordinate winning-section run checked 3,796,757 geometries across
  15,719 frames. No valid state below its elite cap; best violation 9.14e-9.
  Both saved tied-angle near-solutions were independently GEOS-invalid.
- The stronger mirror neutral walk accepted 59 angular moves and **zero** center
  hops. Its repeated `target_hop` records are proposed repair targets, not
  accepted configurations. Do not confuse them with successful transitions.
- Focused mirror continuation completed 3,034,112 proposals. Its nearest state
  independently validates as the existing incumbent, not an improvement.
- An elite-archive replay retained three additional original-topology center
  states distinct under translation and quarter-turn/reflection symmetries.
  The closest scores 2.948527013256507 and is undergoing a separate local search.

The old 2e-12 native area prefilter was an empirical heuristic; documented
estimator errors can exceed it. Future native runs use a 1e-11 guard and still
require GEOS. Checks of the three saved near-witnesses found errors below
9e-13 and no actual missed improvement. Earlier sampling results are not
completeness claims.

Search code now retains raw submitted angles separately from scored geometry.
The concrete non-idempotence witness is a raw angle near **3π**, whose first
normalization can round just outside ±π; another pass changes it. Raw
±3.141593 alone maps to the opposite ±3.141592 and is then stable. No existing
promoted candidate failed validation; this change prevents a latent mismatch.

`new-incumbent-slack.png` and `.svg` show the improved packing and its actual
enclosing supports (pieces 9, 10 and 13). The plot no longer assumes the old
packing's support indices. The full suite passed **111 tests** with
`.venv/bin/python -m pytest`. Invoking the bare pytest executable lacks the
repository import path in this environment; that failed collection, not tests.

## Wider exact domains and faster compatibility tables

Precision scope: these finite exclusions use the explicit target
**2.9485269238**, which is 85.6 picounits below the observed leader. They do
not exclude the tiny interval between that threshold and the leader. New
proof runs use a threshold 5 picounits below the leader, with a final 2-picounit
acceptance guard; API comparison against the actual leader remains mandatory.
Likewise, G's incumbent target 2.9485269629 leaves a 53.5-picounit sliver below
our actual incumbent. Never silently relabel a threshold-specific proof as
excluding every possible strict improvement.

The wider mirror and original neighborhoods now have finite infeasibility
proofs at the first-place target. These include XYZ ±1 corrections and angle
±3 around every selected signed LLL displacement, rather than just a union
of displaced states with a single local box.

- Mirror width-three model: 297.59 seconds total, including 114.64 seconds
  building tables; four leaves and five learned enclosure cuts before UNSAT.
- Original width-three model: the initial 300-second run ended UNKNOWN.
  Reusing its exact tables with all 55 baseline enclosure cuts and interval
  compression proved INFEASIBLE in 22.50 seconds, before any candidate leaf.
  See `conflicts-original/bundled.report.json`.
- Mirror six-direction pair-sum model: base, 12 signed single moves and 60
  signed pair sums, each with the same local corrections. At most 2,169 poses
  per piece; 36.47 million pair decisions. All 55 initial support cuts plus
  intervals exhausted this model in 144.78 seconds, with zero candidate leaves.

Each valid-incumbent positive control passed. Pair-only diagnostic models
remained UNKNOWN; do not describe these as overlap-only proofs or global
packing impossibility results. No domain was silently truncated.

Native true-area upper bounds and local polygon-loss bounds reduced expensive
GEOS decisions to roughly one percent in these builds. The geometric band is
conservative, including a 1e-9 arithmetic allowance; it is different from the
empirical native discovery screen. Exact relation compression was verified on
37,031,666 original categorical cells. Hamming-tour column ordering reduces
the hardest interval tables another 2–4 times without changing their meaning.
Details and reproducible commands: `interval-research.md`.

## Branch coverage and the constrained-stress correction

Three 300-second random-section campaigns D/E/F checked about 24.3 million
distinct geometries. They retained useful nonwinning valid states but did
not improve the incumbent. Matched neighbor campaigns G/H/I then tested the
previously neglected +3.141569 / −3.141615 first-flat branches. G's best valid
state was 2.9485269864331265; H was 2.9485270278418; unlocked I found no valid
relay. G also produced the one-overlap repair target below.

A further audit mapped 612 exact continuous sections across integer flat
differences in under one second. All 612 are geometrically feasible. The old
full-J stress test admitted only 306; with fixed differences, the correct
test uses J times the allowed 43-column transform, including the effect of
angular reaction forces. All 612 pass that test. The least expensive formerly
rejected section costs 23.32 billionths, so the correction does **not** add
new branches to the under-20-billionth shortlist. It matters for broader runs.
There are 49 sections within 5 billionths and 222 within 20 billionths of the
continuous minimum, far more than the four default branches previously used.

`branch-costs.png` displays the full map. The new real-geometry regression
fixture reproduces the false rejection and checks native versus fpylll
decoding, raw-to-scored conversion, and an integer Dirichlet concentration
with a noninteger bias. Three focused tests pass.

## Exceptional wrap angles: bounded negative result

The earlier extra-turn audit correctly ruled out general extra precision,
but missed a small coverage distinction: raw ±9.424778 reaches scored
∓3.141593, whereas ordinary raw ±3.141593 reaches ∓3.141592. This gives two
extra endpoint-angle states. The API evidence already exists in
`experiments/api_angle_probes.json`.

`experiment_seam_alias.py` fixes each of those states and searches the other
42 integer coordinates across exact curved sections. Raw aliases are kept
separate from scored geometry. Two 60-second runs checked 1,107,596 and
286,412 distinct geometries; neither produced a valid candidate. Best combined
violations were 99.66 and 173.68 billionths. The initial two invocations aborted
immediately on an integer-array Dirichlet dtype bug; the fixed runs are clearly
named and both records are preserved in `runs/`. The fix is covered by the
regression above. Stop this particular proposal family unless a new argument
changes its poor geometry; this is no exhaustive exclusion of wrap aliases.

## Rescuing the archive instead of discarding it

`experiment_rescue_frontier.py` froze 3,447 distinct submitted geometries from
4,035 files, ranked their true enclosure radius, and GEOS-checked 1,331 relevant
states in 55 seconds. Raw submission normalization happens once. The old
combined violation objective obscured a useful radius/overlap tradeoff.
`rescue-frontier.png` makes it visible. The GEOS audit covers every frozen
state at radius ≤ first place plus 5 billionths, and selected near-valid
controls above that threshold; the displayed frontier is among those checked
states. New repair targets:

| Source | Radius | Radius minus first, billionths | Bad pairs | Largest normalized overlap, billionths |
|---|---:|---:|---:|---:|
| Tied winning-section nearest | 2.948526904838365 | −19.047 | 3 | 9.154 |
| Mirror campaign A nearest | 2.9485268737802413 | −50.105 | 5 | 18.969 |
| Mirror neighbor G nearest | 2.948526956011454 | +32.126 | 1 | 3.579 |
| Verified incumbent | 2.948526962953499 | +39.068 | 0 | 0 |

These are **invalid repair targets**, except the incumbent. The three-overlap
target's matching 43-dimensional saved basis has been frozen for an exact
finite repair. G's single overlap cannot be fixed by turning piece 9 alone:
that resolves 0/9 but creates 4/9 and 8/9 conflicts. Angular and position
repair must coordinate the affected neighbors.

The stronger fixed-XY angular search for G exhausted all ±20-microradian
choices in 3.26 seconds. Pair 0/9 requires piece 9's angular offset to be in
[+1,+20]; pair 8/9 requires [-20,0], independently of the allowed angles of
pieces 0 and 8. Thus at least one of centers 0, 8 or 9 must move within this
angular range. The source hash and exact certificate are retained in
`conflicts/neighbor-g-position-conflict.json`. A separate root experiment
allows all centers XYZ ±1 and angles ±20 (369 poses per piece), first targeting
first place and then a strict incumbent improvement with the same cached tables.

That G position-box search finished: both targets are **INFEASIBLE**, with
zero candidate leaves after 55 support cuts. First-place build/solve took
54.32 seconds; the incumbent-improvement replay took 3.29 seconds using the
same exact tables. Overlap-only CSPs are feasible, so enclosure is the blocker.
An external known-valid engine control passed; the invalid G seed was never
misrepresented as a positive control. Evidence: `rescue-g-box/summary.json`.

The three-overlap witness's smaller 63-pose XYZ ±1/angle ±3 domain also
exhausted with feasible overlap-only CSP. Two complementary expansions follow:
all 41 nontranslation saved basis directions with local corrections, and a
625-pose XYZ ±2/angle ±12 box. These domains differ, and their provenance is
recorded separately; an UNKNOWN or table-build timeout is not an exclusion.

The 625-pose box is now **INFEASIBLE at 2.9485269238**: 15,234,375 native
pair decisions, 274,182 GEOS decisions, 116.06 seconds through model build,
126.68 seconds total, zero candidate leaves. The overlap-only diagnostic
remained UNKNOWN. The 41-direction union hit its 300-second table-building
deadline after 17.5 million native decisions and at least 706,475 completed-pair
GEOS checks; its scope remains **UNKNOWN**, with partial exact caches retained.

The full suite currently passes **143 tests in 11.30 seconds**. This supersedes
earlier test counts above; those remain as chronological evidence.

Current independent searches are a 25-phase physical rotation pool (±300
microradians, each with local pose corrections), exact repairs of the two
promising witnesses, and broad 49/222-branch native campaigns. Every result
still requires independent GEOS and API validation before promotion.

## Further changes after the rank check

Fresh leaderboard evidence (`leaderboard-latest.json`) still places our
2.948526962953499 candidate **third**, behind second at 2.948526931599013.
The public third-place entry remains the older submission; no new public
submission was made.

Broad J/K/L completed 300 seconds each: 8,084,954 / 8,083,661 / 7,884,588
distinct geometries, with 31 / 24 / 0 retained valid elites and no incumbent
improvement. L's saved nearest state has radius 2.94852690780946 and four
overlaps, so it is another invalid repair target below first. The next sampler
archives the radius/overlap frontier online, including raw input and matching
basis, rather than retaining only one combined-violation minimum.

The mixed 25-phase model completed 600.49 seconds with **UNKNOWN**, zero
target leaves, and a passing incumbent control. A new cached model adds all
55 balanced support cuts at every phase: 1,375 distinct inequalities, 6,457,500
int64 coefficients (51.66 MB), prepared in 0.257 seconds. Exact pair-support
and integer support-bound pruning has independently replayable removals and
21 exhaustive solution-preservation tests. At the new leader-minus-5-picounit
target, 1,598 of 23,625 poses are eliminated; this is pruning, not infeasibility.

A complementary experiment solves individual phase blocks separately. The
mixed model spent 96.7 million native pair decisions on cross-phase choices;
the 25 diagonal 63-pose blocks total only about 3.9 million. Existing cached
tables permit this comparison with no additional geometry evaluations. It
tests whether recombination across distant phases justified its extra cost.

`experiment_contact_release.py` introduces directed structural exploration:
maximize the exact separation of one active contact under a specified radius
barrier, keep all 15 pieces full-sized and overlap-constrained, then compress
again. The initial 15-second smoke attempted 34 releases. Its only new graph
came from a failed opening solve violating the radius/overlap constraints; that
is **not** evidence of a valid barrier-crossing path. The corrected full run
rejects such failures before recompression and records them explicitly. It
enumerates both baseline geometries, all 41 active contacts and four radius
barriers (0.0001, 0.001, 0.01, 0.05), rather than repeating removal/regrowth.

The corrected contact-release run completed **all 328 scheduled cases in
145.90 seconds**. Sixteen opening solves violated their constraints and were
rejected. Of the recompressed states, 308 returned near the competitive
continuous minimum. Three additional graphs relative to the two input seeds
had radii 2.9619098, 2.9935443 and 2.9974672; they are not useful competitive
starts, and have not been claimed novel against the entire historical archive.
No valid improvement. Stop this single-contact release schedule; a further
structural run needs a genuinely different transition mechanism.

## Learning clearance patterns, with a memorization check

`prepare_slack_patterns.py` extracted 177 aligned, 41-contact patterns from
frozen raw inputs and matching continuous frames: **175 reference-valid**
mirror elites and two invalid G/L witnesses. It evaluates each packing at its
actual free enclosing center, uses constrained J@T stress, then projects
weighted clearances onto the simplex at the same total budget. The NPZ retains
source labels, scores, paths and hashes; the report also hashes sidecars.
Twenty-four rigid-rotation checks had maximum allocation-fraction discrepancy
6.85e-7; all row sums agree with one within 2.3e-16. Runtime: 28.13 seconds.

The allocation diagnostic is useful: the valid examples put **22.92%** of
weighted clearance on boundary constraints on average; the incumbent uses
**16.28%**. A uniform target puts 26.83% there; the old boundary-bias-five target
puts about 64.7%, consistent with its poor valid-candidate rate.

A matched 60-second comparison used identical continuous frames and an
independent phase RNG. Baseline checked 1,382,234 distinct poses and obtained
12 valid records; uniform learned targets checked 1,371,218 and obtained 23.
However, the novelty audit found **21/23 learned records reused training XY
centers**, versus 3/12 baseline records. The best learned radius exactly
recovered an old D elite with different angles. Its supposedly new near-witness
also reused old H centers. These counts therefore do **not** establish improved
exploration. The phase seed itself was new; recovery of existing centers is
the issue. Future reports separate new centers, known-center recovery, and
radius/overlap frontier quality.

The next controlled comparison weights valid training patterns by score with
a 20-billionth temperature, retains 20% ordinary proposals and uses an entirely
new phase seed shared only by that matched pair. It excludes invalid training
labels by default. The sampling distribution passed a 100,000-draw frequency
audit and rejects a mismatched active-constraint order.

The online frontier now saves raw input, origin, anchor, complete basis and
integer coefficients. Initial checking cost roughly 31% throughput; deferring
farther-than-25-billionth contenders into ten-second batches cut the measured
GEOS fraction to about 12–14%, while close contenders remain immediate.

## Exact phase search after factoring the support functions

All 25 coherent cached phase blocks proved **INFEASIBLE at leader minus
5 picounits**, in about 77 seconds total without new pair geometry: 23 CP
proofs and two empty-domain propagation proofs. The next fresh 300-second
ensemble completed 58 independent phases, 57 exact exclusions and one
preparation timeout; no candidate leaves. These are finite XYZ ±1/theta ±3
domains, not proofs covering all integer placements or global rotations.

The mixed phase model also finally exhausted. A support function separates
into affine XY terms and an angle-only term. Sharing position/angle variables
shrinks 6,457,500 pose-table entries to 126,875 angle entries plus 70,875 shared
pose mappings; 650 of 1,375 cuts become pure sparse linear inequalities.
All integer activity bounds fit int64, and numerical/floor allowances were
audited against every original pose/cut. The actual incumbent control passed.
The resulting target model proved INFEASIBLE in **59.75 seconds** (96.26 seconds
including construction/control/diagnostics), versus an external timeout beyond
330 seconds with the old encoding. `conflicts/phase-separable.*` retains the
actual CP response and bound. This changes efficiency, not the finite geometry.

Next: a lazy polygon oracle starts from conservative allowed-pair supersets,
checks uncertain cells only when selected by CP, and learns exact pair
exclusions. An infeasible relaxed model already excludes its exact counterpart;
a feasible relaxed leaf is never accepted without GEOS. This avoids paying for
thousands of polygon intersections in phases that are impossible anyway.

## Recovery benchmark after pausing the search

The user requested the benchmark separately; optimization campaigns remain
paused. The latest partial lazy campaign had completed 114 finite exclusions.
Core shrinking finished without a small repair patch: G still involves 14 pieces,
and the three-overlap witness involves all 15. Neither is a minimum-core proof.

`search.py recovery` now fits and audits four immutable controls, with the
incumbent for width calibration and three held-out valid elites. Rigid alignment
leaves required XY widths of 102–232 grid units. Fitting both physical modes
reduces all four to ±1 in XYZ/angle coordinates. The existing production
±1 XY/±12 angle domains contain all controls without widening; the offline lazy
solver recovers all four recorded scores in about 10.5 seconds. Target poses are
not inserted or hinted, though frame alignment explicitly uses the known targets.
This establishes coverage and local recovery, not blind outer-search success.

The workspace suite passes 189 tests; a clean export containing only Git-eligible
files passes 186 with three optional archive tests skipped, and repeats the
four-case benchmark successfully. Bulk run data is ignored without deletion.
Method, caveats, command and compact results are in
[`docs/recovery-benchmark.md`](../../../docs/recovery-benchmark.md).
