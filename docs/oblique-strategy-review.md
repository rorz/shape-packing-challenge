# Oblique strategy review, 2026-11-04

The objective remains an API-verified score below first place. This review
consulted three independent agents: geometry, integer search, and adversarial
evaluation. It adds a cheap recovery control and read-only diagnostics; it does
not launch the proposed production campaigns or change the saved solution.

## Current evidence

The [fresh leaderboard](oblique-leaderboard.json), checked at 14:29 UTC, still
has first at **2.9485269238856113** and second at **2.948526931599013**. Our
API-verified local best remains **2.948526950912552**, leaving
**2.7026940685459522e-8** to first. No public entry was submitted.

The score alone does not identify the leader's method or geometry. A better
integer-grid realization within our current family is a working hypothesis,
supported by our recent progress, not an established description of first place.

The reviewers disagreed about priority: geometry favored deliberate rounding
events; integer search and adversarial evaluation favored a short archive
completion pass first. They agreed that unchanged broad random sampling has
weak justification after the existing campaigns.

### The previous gain is reproducible by the repair solver

We transformed the old invalid witness clockwise, applied its known integer
translation, and generated fixed-center angle domains of ±1 microradian.
The lazy solver recovered a reference-valid packing at
**2.9485269509125525** in **0.306 seconds of solver time**, after two CP calls
and two exact pair exclusions. Independent reference validation passed.
The target assignment was not supplied as a hint.

This is a target-assisted positive control using the known orientation and
score, not blind discovery or a new improvement. Previously we had tested only
that these domains contained the improved packing. [The complete control
record](oblique-orbit-control.json) retains input poses, transformations, domains,
source hashes, parameters and returned geometry. Reproduction uses
`transform_scored`, `make_angle_domains(..., 1)`, then
`solve_phase(..., seconds=5, seed=3111501, local_only=True)` with its recorded
input, target and translation. Full diagnostics remain local under
`experiments/pantheon-pilot/oblique-orbit-control-v1/`.

### The archive is larger than the last orbit test, but mostly difficult

A [census](oblique-archive-census.json) of seven mirror-frontier directories read
1,281 saved sidecars. It found 421 records with stored radii below first,
representing 319 distinct center patterns even after D4/translation grouping.
All 421 have recorded GEOS violations. These are saved measurements, not fresh
validation, and D4 grouping is used only for reporting.

| Maximum normalized overlap deficit | Records below first | Center classes |
|---|---:|---:|
| <20 billionths | 1 | 1 |
| <30 billionths | 3 | 3 |
| <50 billionths | 17 | 13 |
| <100 billionths | 421 | 319 |

Here and in older `_nm` fields, “billionths” means normalized radius units;
these are not physical nanometers. The one under-20 case was already included
in the four-input orbit audit. The next two are baseline-frontier witnesses
00341 and 00307, with recorded deficits 25.916 and 28.682 billionths.
Their exact historical domain coverage still needs checking. Do not call the
entire census unsearched or turn 319 difficult patterns into a success estimate.

### Our current radius is locally fixed by three centers

The [support audit](oblique-support-audit.json) identifies zero-based pieces
5, 10 and 14 as radial arc supports. Their angular margins before support type
changes are about 0.396, 0.956 and 1.034 radians. Small angle adjustments alone
cannot lower this packing's radius; supporting centers must move or the
support topology must change.

Fixing center 5 and moving centers 10/14 through relative ±2-microunit boxes
gives 625 triangles, of which 144 leave the full packing's native radius below
first. At ±4 the counts are 6,561 and 1,600. None passes the native overlap
screen. A two-conflict example has reference MEC radius 2.9485269108296706,
but its normalized overlap deficit is about **1,091 billionths**, much worse
than archived near-winners. Fewer bad pairs is not enough to rank a repair.

## Recommended execution order

### 1. Finish a deliberately small archive repair schedule

Use the 13 strongest census classes as the first queue, retaining their distinct
angular realizations. Freeze raw hashes and actual domain contents before
searching. Rank by radius opportunity, exact overlap deficit and contact pattern.
Stored radius is a priority, not a sound exclusion: angular completion can alter
enclosure. Conservative support bounds on each actual domain can exclude it.

Test all eight orientations with ±3 and ±12 angular neighborhoods, using
`experiment_angle_orbits.py` and the lazy solver. Skip a domain only when it is
identical to, or contained in, an already exhausted domain whose radius cap is
at least as large: `D_new ⊆ D_proved` and `R_proved ≥ R_new`, under the same
validation and numerical policies. A proof at leader minus 85 picounits cannot
exclude a model at leader minus 5 picounits. UNKNOWN and preparation timeouts do
not qualify.
Keep a ledger of input/orientation/domain/threshold/status. Preserve seam aliases.

**Budget:** at most 300 seconds, stopping earlier if the frozen queue finishes.
Only a better independently valid score is a production success. Record completed
and unresolved coverage separately. Widen XY only when the resulting domain is
genuinely new; no automatic indefinite expansion after negative results.

This directly exploits the observed gain and covers different inputs from the
64-model audit. It is a cheap exploitation test, not a claim that the archive
contains the winner. Eight orientations of every frozen center would instead
mean roughly 37,000 models and is not the proposed first pass.

### 2. Traverse rounding boundaries in the fitted family

The two physical deformation coordinates and global rotation are the three
remaining outer parameters after common translation is covered by the union.
The old phase pools used spaced rotations or random draws. We should follow
the places where the actual generated integer domains change.

Start from the fitted incumbent and two distinct promising archive frames.
Follow physical-mode axes, their diagonals and rotation. Derivatives predict
events; bracketing/bisection against the production domain generator locates
candidate transitions. It does not certify the next event on a nonlinear ray:
an interval can hide multiple crossings, and an approximate root can miss an
exact rounding tie. Include both sides and conservative unions from outward
coordinate/angle bounds at unresolved boundary brackets. Where such enclosures
are unavailable, record possible missed events rather than claim coverage.
Exact ties can create larger domains. Include raw-angle wrap events. Hash
domains, cache unchanged pair geometry, and keep a small random-jump share.

This is an event-guided search, not complete enumeration of the nonlinear
three-dimensional cell arrangement. Adjacent models may share work without
sharing feasibility. Reused exclusions require identical actual geometry and
sound numerical handling; acceptance always returns to the reference validator.

**Gate:** a 60-second construction diagnostic, then at most 180 seconds against
an equal-time ordinary-phase control. Cap each arm at 200 distinct models;
record actual counts if the time limit comes first. Primary metrics are best
validated score and time to improvement; secondary metrics are the exact
radius/overlap frontier, duplicate-domain rate and construction cost. Large
counts of infeasibility certificates alone do not justify a longer campaign.

### 3. Allow temporary worsening, but require real center moves

The old neutral walker made 59 angular moves and zero center hops. A different
experiment uses CP to require a center configuration distinct from every visited
one modulo common integer translation, at relay cap best +50 or +100 billionths.
Let all 15 pieces move jointly, enqueue new valid centers, then try tightening.
Do not merge D4 orientations: their reachable angle grids differ.

**Gate:** 120 seconds to obtain at least one independently valid center hop.
Angular variants cannot satisfy that gate. A finite no-hop result closes the
tested neighborhood and barrier, not the whole valid-state graph.

### 4. A genuine topology escape: fuse both pairs, then split jointly

The earlier topology log explicitly leaves simultaneous two-disk fission
untested. Fuse both opposed-flat pairs, optimize eleven semicircles plus two
disks, jointly choose both cutting diameters, and release all four halves.
This tests a coupled transition absent from single-disk fission and the 328
single-contact releases. Reuse the mixed-shape geometry and fission machinery.

**Gate:** two macro starts, at most eight partitions per disk and three stagger
choices per disk: at most 1,152 trials, with a 180-second wall cap. The count is
a ceiling, not a throughput prediction. Retain by geometry/contact graph as
well as competitive continuous radius; radius-only deduplication previously
hid the mirror topology. Give each genuinely new competitive seed the same
short integer-search budget. Continuous novelty alone is not a score gain.

## Search policies to change or retain

- `blind_packing_worker.py` currently deduplicates nominations by center hash
  before fitting. Different angles can produce different fitted domains. Use
  scored-pose identity for nomination and complete-domain identity for exact
  work; skip already-qualified centers only when the objective is recovery at
  that cap. That skip is too coarse for optimizing their radius further.
- Center novelty remains a useful exploration statistic, not a success gate
  for the overall hunt. Our latest gain came from previously invalid centers.
- “Repair only offending pieces” is unsupported: earlier shrunken target cores
  still involved 14 or all 15 pieces. Preserve coordinated motion.
- Center projection plus binary/min-cost angle completion already received
  about 1.85 million center proposals and 900,000 angular checks. A new version
  needs a materially stronger oracle or proposal mechanism.
- Keep the existing C++ kernels. The evidence points to proposal selection and
  coverage, not a need to rewrite working native code in another language.

## Secondary hypotheses, not the next broad campaign

Support-first conditional completion could pin promising integer support
triangles, reoptimize the remaining continuous poses, then quantize. Test 32
choices before expensive CP work; the raw-support diagnostic shows why immediate
angle patching is a poor bet. This constructs enclosure-compatible proposals
instead of merely adding enclosure cuts afterward.

Polygon-facet phase is another selector for the two physical modes. The
4096-point arcs have about 73.6-billionth maximum chord sagitta, but that is
already represented by our geometry; it is not extra unclaimed slack. Rigid
rotation preserves relative facet phase before rounding. A useful experiment
must show improved blind integer outcomes from phase-selected physical modes,
not just a lower continuous radius or the existence of polygon approximation.

## Sources and scope

The recommendations primarily use the repository's experiment records,
especially [round 3](../experiments/2026-11-04/round3/README.md), the
[pilot](pantheon-pilot.md), and the linked diagnostics. Primary external
background checked during this review:

- [OR-Tools response definitions](https://github.com/google/or-tools/blob/stable/ortools/sat/cp_model.proto):
  returned infeasibility cores are sufficient and need not be minimum; time
  limits and UNKNOWN statuses do not prove infeasibility.
- [Grosso, Locatelli, Jamali and Schoen, population basin hopping](https://optimization-online.org/2008/06/1999/):
  precedent for perturbing and reoptimizing packing populations. It does not
  establish performance for this quantized semicircle puzzle or justify
  repeating our unsuccessful basin experiments unchanged.

All new artifacts are diagnostics and review records. The saved incumbent and
production solver source are unchanged. Full census inventories and control
traces are ignored locally; the compact census, shortlist, hashes and control
geometry remain Git-eligible. No new first-place result is claimed.
