# Scoring and validation audit — 2026-11-04

All times below use Asia/Kuala_Lumpur (UTC+8). This audit uses ordinary finite
geometric submissions. It does not alter `solution.json` or the shared scorer.
No public submission operation is included in the audit tools.

## 09:03–09:05 — Baseline, prior evidence, and audit design

Read `README.md`, `experiments/README.md`, `experiments/best_run.json`,
`experiments/api_best.json`, `src/semicircle_packing/scoring.py`,
`src/semicircle_packing/geometry.py`, and `src/semicircle_packing/config.py`.
The existing angle rule remains round → wrap with atan2 → round; the disproved
extra-turn angle encoding approach was not repeated.

Refreshed the public leaderboard:

```sh
python scripts/arena.py leaderboard --save /tmp/packing-2026-11-04-leaderboard.json
```

Copied that response to `leaderboard.json` in this directory. Results:

| Rank | Author | Radius |
| --- | --- | --- |
| 1 | @cameronspirals | 2.9485269238856113 |
| 2 | @fordgt1111 | 2.948526931599013 |
| 3 | @rorzio | 2.948526966725367 |

The checked-in best is already the user's public third-place score. Its SHA256
is `088950a795745e2757fd462e6a9085220777ca00422bca66fb3d9c6bba0e475e`.
The remaining gap is `4.28397557428184e-8`.

Local reference MEC:
`(-2.0454804319884002e-5, -4.910663767119286e-5, 2.948526966725367)`.
The independent convex solver returned radius `2.9485269667253693`.
Shapes 3, 5, and 7 support the enclosing circle on their outward arcs; their
maximum containment excess is `4.440892098500626e-16`.

The scorer has a `1e-10` inclusion tolerance during Welzl and a final analytic
containment tolerance of `1e-9`. Consequently, a valid equivalent packing
cannot gain the full 42.84 nm gap merely by understating its enclosing radius:
final containment caps such understatement at 1 nm, provided coordinates are
numerically well resolved. Large cancellation-induced changes are measured
against stable centered geometry and do not qualify as a geometric improvement.

Started a bounded, reproducible numerical and rounding audit:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_scoring_freedom.py \
  --input solution.json --output-dir experiments/2026-11-04 \
  --seed 20261104 --orders 1024 --seconds 180 \
  > /tmp/scoring-audit-20261104.log 2>&1
```

The audit checks 15 cyclic orders, reversal, and 1024 seeded random orders;
traces all MEC refinement rounds; checks 20 moderate translations up to 10000
units with analytic containment and centered-radius comparisons; checks the
eight grid isometries; and enumerates distinct coordinate rounding patterns at
half-microunit translations. Half-grid candidates are saved as ordinary
six-decimal placements, so any geometric gain comes from the resulting valid
coordinates rather than hidden payload precision.

Intermediate and final machine-readable results are in `scoring-audit.json`.
The audit script records failures as well as successful cases and never calls
the remote checker automatically.

## 09:05–09:08 — Numerical audit completed; routes rejected

The bounded audit completed in **19.33 seconds**. Detailed arrays are in
`scoring-audit.json`.

- **1040 orderings:** radius range `[2.948526966725367,
  2.9485269667253675]`; all contained. Permutation gains are immaterial.
- **MEC refinement:** sampled radius `2.9485247799855627` initially missed
  the true boundary by `5.8380111e-6`. Three analytic refinements reached
  `2.948526966725367`, with remaining excess `4.44e-16`. The 20-round cap is
  not binding on this packing.
- **Exact geometric bound:** the acute triangle of centers 3, 5, and 7 has
  circumradius plus one equal to
  `2.948526966725367362010126017397889977821401763914302201435353244844873`.
  The corresponding three outward arc points are real members of the packing
  and support this minimum circle. Equivalent geometry cannot beat this bound.
- **Grid isometries:** identity and reflection across the x axis remained
  valid and had the same radius. The other six transformations became invalid
  after mandatory six-decimal angle rounding; their overlap lists are retained.
- **Half-grid coordinate seams:** 13 distinct x patterns and 16 y patterns
  gave 208 combinations. Twelve passed the native prefilter and the original
  validator, but none improved geometry. Several worsened the radius to
  `2.948527370123706`.
- **Translations:** the best within the server's coordinate domain differed
  by only `2.22e-15`, at `(-10,+10)`. Larger absolute-coordinate computations
  mainly overestimated the radius. Stable centered evaluations continued to
  agree with the original geometry.

One rejected API probe is important to preserve. The best *local* numerical
artifact was translation `(-100,+100)`, score `2.9485269667250016`, an apparent
gain of only `3.65e-13`. It was geometrically valid locally, but the server has
an additional coordinate domain that was absent from the repository README:
**x and y must each lie in `[-100,100]`**. The translation crossed that domain.

```sh
python scripts/arena.py check experiments/2026-11-04/scoring-best.json \
  --save experiments/2026-11-04/api-scoring-best.json
```

Server result: `valid:false`, `Item 1: coordinates out of bounds (x, y must be
in [-100, 100])`. The payload is preserved as
`scoring-best-out-of-bounds.json`; the response record retains its SHA256 and
original path. It was never promoted. The audit script was corrected to mark
the server bounds and exclude such candidates from its best-candidate selection.
There were no further remote roundoff probes.

Conclusion: ordering, legal equivalent translations, reflections, and the tested
half-grid seams do not supply a meaningful improvement. The true geometric gap
is roughly 43 times the final containment tolerance. Further work must change
the valid rounded packing, rather than rely on MEC understatement.

## 09:08–09:15 — Independent review of the new search models

Reviewed `experiment_cpsat.py`, `experiment_oblique_kbest.py`,
`experiment_alt_branches.py`, `experiment_alt_geometry.py`,
`experiment_alt_quadratic.py`, `experiment_difference_map.py`, and
`research_run.py` without changing their owners' files.

### Corrected rotated-incumbent guidance bug

The initial k-best implementation added a frame angle to the incumbent's
orientations without wrapping them, then computed `known_gaps` against the
wrapped continuous seed. Its regularization prior wrapped the angular
difference later, so the two guidance terms disagreed across ±pi.

A probe using `solution.json`, `angle=1` radian, and the same continuous seed
crossed the seam for pieces **2, 9, and 13**. The raw guidance differed from
the wrapped guidance by **3.6172097855** in normalized length units. Correct
wrapped gaps ranged only from `6.4867e-9` to `6.6959e-7`. Inputs, raw/wrapped
deltas, and resulting gaps are in `oblique-angle-guidance-probe.npz`.

Reported this to the owner, who corrected wrapped `known_delta` before
computing `known_gaps`. Completed angle-zero runs were unaffected. Actual
geometry checks prevented this guidance error from producing false winners.

### CP-SAT numerical scope

Integerizing a tangent row with `round(j*1e16)` and a ceiled RHS is not a
conservative relaxation unless coefficient-rounding error is included. At the
current boundary Jacobian, the maximum center-coefficient error across the
allowed box is **0.0454, 0.0908, and 0.1817 nm** for movement bounds of 50,
100, and 200 microunits respectively. The 0.1 nm center grid independently
has up to **0.0707 nm** Euclidean rounding error. These are small relative to
the remaining 42.84 nm gap, but can exclude a marginal first-place witness;
an integerized model's infeasibility is not an exact geometric proof.
Details: `cpsat-rounding-audit.json`. The reference/API promotion guards still
prevent false improvements.

### Difference-map projection and candidate center

The weighted free-center mathematics checks out. With
`C = -weights * sum(placement xy derivatives)`, eliminating the center uses
`P = I - C C+`; for a decoded integer move, the optimal fitted center is
`C+ (x_contact - WJ*n)`. Subtracting that center times `1e-6` from the candidate
matches the derivative sign and units. The implemented Douglas–Rachford update
has the stated direction.

Seed **20261104** gave center rank 2, residual orthogonality `7.07e-17`, and
`||P*P-P|| = 4.51e-16`; projected and explicit-center residual norms matched.
Details: `difference-map-projection-audit.json`.

The anchor's atan2 angular error is correct. A remaining local-model caveat is
that a future candidate crossing ±pi receives the server's additional decimal
rounding, so the angular lattice has a small seam discontinuity. Exact candidate
evaluation remains essential. Finite coefficient/movement boxes and the bounded
Babai approximation are search restrictions, not completeness proofs.

### Alternative models and research records

The alternative branch's odd-pi centering, the tied flat-angle increments, the
continuous center columns, and the power-transformed area Jacobian are
consistent. Taylor models and tangent cuts are candidate generators; their
logged model errors and final reference/API checks are necessary.

Reported reproducibility gaps in `research_run.py`: only pre-run source hashes
are captured, so concurrent edits are not detected; Python/package versions
are absent; default input paths and `--input=file` arguments are not hashed;
and timeout cleanup targets the immediate process rather than an experiment's
whole process group. These do not change geometry, but can undermine claims of
an exact replay. Record end hashes and environment versions, explicitly track
default inputs, and retain source snapshots or otherwise mark changed-source
runs as non-reproducible.

## 09:15–09:18 — Focused exact-contact repair of new DR inputs

Novel hypothesis: the new residual-feedback search produced phases and contact
branches different from the previous CSP inputs. Their nearby integer packing
domains might admit an exact repair even when the projection's linear model
does not. This is a new-input repair, not a rerun on the old continuous seed.

Implemented `scripts/audit_repair.py`. Five nearest-point files were copied
into `repair/*-input.json` at run start; all copied byte hashes match their
recorded source hashes. They include `best`, `broader`, `broader32`, `locked`,
and `nonlinear`. The original executed source is retained as
`repair/audit_repair_v1.py`.

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/research_run.py \
  --label audit-dr-exact-repair \
  --hypothesis 'New DR near-solutions have new grid phases; exact local pair compatibility with support-derived free-center scans may repair overlaps missed by approximate projections.' \
  --timeout 240 -- .venv/bin/python scripts/audit_repair.py \
  --seconds 180 --seed 110414 --centers 96 --nodes 10000
```

For steps 1, 2, and 3 the solver built exact analytic pair-area compatibility
tables around each input. A fixed polygon deficit plus a `2e-11` area allowance
kept this a permissive candidate screen. Any putative leaf would be checked
against exact GEOS polygons, incompatible pairs learned, then the original
scorer and API. Center options included the input's actual MEC center, nearby
scans, and intersections of container-center disks associated with its support
arcs/endpoints. Targets were **2.94852694** and **2.9485269238**.

Result in **37.20 seconds**: 15 models, **3552** center tests, 670 empty domain
cases, 2882 without an assignment, **zero candidate leaves**, and zero API
calls. No candidate was promoted. The result is bounded by the center sample,
coordinate boxes, node budget, and approximation used only for pair screening;
it is not a proof about other nearby centers or larger joint moves.

A positive control verified that this machinery can recover the actual current
packing at its recomputed MEC center with 1 nm of radius slack:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/research_run.py \
  --label audit-repair-sanity \
  --hypothesis 'Verify fresh repair machinery can recover a known valid packing at its true MEC center with one nanometer of radius slack before interpreting repair failures.' \
  --timeout 60 -- .venv/bin/python scripts/audit_repair.py \
  --directory experiments/2026-11-04/repair-sanity-input \
  --output experiments/2026-11-04/repair-sanity --seconds 30 \
  --steps 1 --centers 0 --nodes 10000 --targets 2.948526967725367
```

Positive-control result: **one reference-valid packing recovered**, zero learned
overlap rejections, zero node-limited cases, in **0.283 seconds**. The other 24
shifted center samples failed, illustrating why a fixed center can strongly
restrict an otherwise valid packing. Run metadata and captured stdout are under
`runs/`; detailed repair events and summaries are under `repair/` and
`repair-sanity/`.

## 09:22 — Final position-projection and angular solver review

Reviewed `experiment_position_projection.py` and the new `angular_rounder.py`.
The 28 integer position indices correctly omit piece 0's x/y. The 17 eliminated
variables are 15 angles plus two enclosing-center coordinates. The block matrix
K, its Euclidean pseudoinverse elimination, the reduced basis extraction, and
the sign/unit of the returned center are consistent.

One search restriction was reported: the cache key contains only integer
positions and angle floor cells, but `round_angles` fixes containment at the
continuous-angle packing's MEC center. That center can vary within the same
floor cells when an endpoint supports the circle. Caching a first UNSAT result
for the whole cell can therefore suppress a later, different-center attempt.
Include the tested center in the key or avoid treating fixed-center UNSAT as
a center-independent result. This cannot create a false improvement because
the final reference validation is separate.

Independently checked 1800 random binary systems, seed **110430**, sizes 1–6,
against exhaustive assignments. All 2-SAT satisfiability and returned assignments
matched. Also checked every variant-pair index for original sizes 2–15 against
an independently constructed pair list. Results: `angular-rounder-audit.json`.

All diagnostic model probes have a consolidated replay command and retained
source hashes:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/research_run.py \
  --label audit-model-probes \
  --hypothesis 'Replay angle-seam guidance, center projection, CP-SAT row rounding and exhaustive small 2-SAT checks independently of the search algorithms.' \
  --timeout 60 -- .venv/bin/python scripts/audit_search_models.py \
  --input solution.json --output experiments/2026-11-04/model-audit-replay.json
```

The replay completed successfully. The old rotated-guidance formula is tested
explicitly against its wrapped equivalent even though the actual search owner
has corrected that bug; no old search code is reinstated.

## 09:25–09:27 — Repair around genuinely feasible continuous-angle points

The position solver generated a different kind of input: positions are already
integer, while angles remain continuous and all native inequalities are
satisfied. This is stronger evidence than the earlier all-grid invalid DR
points, so a second, explicitly new-input repair was authorized.

Snapshotted both files and recorded original SHA256, MEC, and native clearances
in `repair-continuous-input/manifest.json`:

| Input | Continuous radius | Minimum area clearance |
| --- | --- | --- |
| positions/best_continuous.json | 2.9485269214795395 | 22.59 nm |
| positions/angular-feedback_continuous.json | 2.94852693227288 | 26.55 nm |

These are continuous-angle results, **not valid grid submissions**. The copied
files retain full precision. The CSP forms nearby grid choices around each.

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/research_run.py \
  --label audit-continuous-exact-repair \
  --hypothesis 'Fresh integer-center continuous-angle solutions are native-feasible; allow neighboring integer center moves and angle grids together, with support-derived MEC-center scans, rather than duplicate fixed-center angular rounding.' \
  --timeout 240 -- .venv/bin/python scripts/audit_repair.py \
  --directory experiments/2026-11-04/repair-continuous-input \
  --output experiments/2026-11-04/repair-continuous --seconds 180 \
  --seed 110431 --centers 256 --nodes 20000 \
  --targets 2.94852695 2.9485269238
```

Result: 6 models, 3268 center tests, 672 empty domains, 2596 no-assignment
cases, zero node-limited cases, zero leaves, and no promotion/API call. Total
time **39.90 seconds**. The finite-center and finite-domain limitations remain.

A useful structural obstruction emerged. The best continuous point is supported
by outward arcs 7 and 13 and an almost tied pair of endpoints on shape 14:

- theta14 = `-0.5540058427136408`.
- Endpoint distances are `2.948526920815874` and `2.9485269214795364`.
- Rounding only theta14 down to `-0.554006` raises the true MEC to
  `2.9485269778566683`, **53.97 nm above first**.
- Rounding it up to `-0.554005` raises the true MEC to
  `2.948527255678027`, **331.79 nm above first**.

The subset consisting only of shapes 7, 13, and 14 already gives radii
`2.9485269778566687` and `2.948527252093041` respectively. Thus the apparent
continuous improvement uses an endpoint orientation that its adjacent angle
grid values cannot preserve at those positions. Reported a promising model
change: include theta14 as an integer variable alongside the 28 positions,
while eliminating the remaining 14 angles and the circle center. Data:
`endpoint-angle-audit.json`.

## 09:28 onward — Minimum-cost angular completion for residual feedback

New authorized component: return the least-violating angular assignment even
when the binary feasibility test is UNSAT, so the position projection can react
to actual grid choices instead of repeating blind rounding.

Implemented `scripts/angular_min_cost.py` and `_angular_min_cost.cpp`, exposing:

```python
candidate, info = minimize_angles(p, target, center=None, area_scale=None)
```

All 15 floor/ceiling angle choices are precomputed with server normalization.
The 105 pair tables use native polygon-area excess divided by the caller's
area Jacobian scales. Shape containment supplies unary maximum-distance excess.
The primary objective is the sum of squared length violations. Angular distance
from the continuous proposal breaks only exact energy ties; a feasible energy
zero always outranks an infeasible assignment.

Native enumeration follows Gray code through all 32768 assignments, updating
only the 15 terms affected by each flipped angle. It periodically recomputes
the nonnegative full cost and independently rescores prospective incumbents
with a floating-error guard. With no supplied center, the function alternates
assignment and true-MEC center updates up to three times, retaining the best
proposal. This is exact binary enumeration at each tested center, not a global
joint assignment/center proof. A supplied center remains fixed. The return is
always a six-decimal proposal; `feasible_estimate` still requires the original
polygon validator and API before promotion.

Validation command:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_angular_min_cost.py -q
```

**13 tests passed.** Coverage includes native-vs-independent exhaustive search
for n=1–8 at normal and very small cost scales, a full vectorized exhaustive
15-bit comparison, feasible-first preference, recovery of the actual valid
packing, the public n=1 empty-pair case, and a genuinely infeasible family that
must still return a quantized proposal.

Benchmark seed **110440**, 100 native repeats and 20 full-pipeline repeats on
the snapshotted continuous input:

| Operation | Time |
| --- | --- |
| All 32768 native assignments | 0.523 ms mean |
| Pair-table construction + fixed-center completion | 0.713 ms median |
| Completion with true-MEC center refits | 2.233 ms median |

Full timings and returned assignment/violation details are retained in
`angular-min-cost-benchmark.json`. The benchmark's best assignment remains
infeasible (three overlap pairs); it is a feedback signal, not a claimed packing
improvement. No new server request or shared solution edit was made for this
component.
