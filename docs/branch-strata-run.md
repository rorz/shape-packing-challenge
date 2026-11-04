# Equal-coverage search of expensive flat branches

The deadline review identified an untested allocation choice. The exact branch
map contains 612 geometrically feasible fibers. Earlier broad campaigns searched
the 222 fibers with nominal continuous cost below 20 billionths and weighted
selection by remaining radius budget to the fortieth power. This gate instead
gives equal attempted coverage to three strata outside those campaigns' cap:

| Nominal extra continuous radius | Fibers | Budget |
|---|---:|---:|
| 20–50 billionths | 135 | 90 s |
| 50–100 billionths | 166 | 90 s |
| 100–126 billionths | 89 | 90 s |

Older unlocked 45-coordinate searches could have visited some of these branches
incidentally; no claim of complete historical absence is made.

The question is whether a more expensive continuous section can fit the integer
grid better. Cost is measured at the original zero-mode section; fresh mode
perturbations change the actual radius. These observations cannot establish a
causal correlation between nominal cost and packing quality.

Every shuffled pass attempts every branch once, so attempted exposure differs
by at most one within a stratum. Rejected sections still count as attempts;
successful frames and evaluated proposals are reported separately per branch.
Fresh physical coordinates use independent normal samples with sigma 0.0002.
Visits alternate the original and reflected families, negating raw flat
differences under reflection. Preferred orientations cycle through the four
quarter-turns near the current best; 75% request these local neighborhoods,
with a global wrap-cell fallback, and 25% request global valid wrap cells.

The actual tied frame computes stress from `J @ T`. The obsolete full-J sign
test is never used to exclude a frame. Per-branch counts retain how many
accepted frames had negative full-J stress. The native decoder is checked
against 64 fpylll proposals at the start of each block. A separate eight-second
known-branch control uses a fit of the current verified packing before the
three strata; that control is explicitly target-assisted, not blind discovery.

The existing GEOS frontier preserves raw JSON, all 105 areas, scale, continuous
origin, basis, coefficients, mode coordinates, branch and orientation. Final
GEOS flush work is included in elapsed time; validation after a block's deadline
is marked explicitly. Only the reference validator decides a valid candidate.
This runner performs no API requests and never promotes a global incumbent.

Four focused tests passed in 0.57 seconds, including an actual expensive branch
whose negative full-J stress becomes positive in the allowed tied coordinates.

```bash
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/experiment_branch_strata.py \
  --output experiments/pantheon-pilot/deadline-branch-strata-v2 \
  --seconds 90 --control-seconds 8 --seed 3111701
```

## Result

The v1 control recovered the current best in 0.49 seconds, then parent-report
serialization stopped the process before any stratum ran: the fitted branch
metadata retained a NumPy array. Passing the branch as a Python list fixed the
serialization; a direct real-fit JSON check passed. No sampling algorithm
changed. The failed v1 evidence remains intact, and v2 restarts in a new output
directory with the same declared seeds. This repeats the short control only;
all three cost-stratum budgets were untouched by the failed run.

The v2 control again recovered the incumbent, in 0.48 seconds. All three strata
then completed, checking **7,395,637 geometries** after native deduplication.
That deduplication uses a bounded cache; this is not a global uniqueness count.
All 390 intended fibers were visited, with no continuous-section or stress
rejections. The three nominal 90-second blocks took 270.97 seconds including
final GEOS flushes; the control took 10.81 seconds including its final flush.

| Stratum | Attempts per fiber | Geometry checks | Best reference-valid radius |
|---|---:|---:|---:|
| 20–50 | 11–12 | 2,425,791 | 2.948527009207927 |
| 50–100 | 9–10 | 2,405,408 | 2.948527045780615 |
| 100–126 | 19–20 | 2,564,438 | None |

**No improvement was found.** The first two best scores are worse than the
saved incumbent. The last stratum produced no reference-valid packing within
the configured validation radius window. This does not exclude valid packings
outside that window or other samples from these fibers.

The obsolete full-J test would have rejected 598 accepted frames in the first
stratum, 1,619 in the second and all 1,753 in the third; the actual tied stresses
remained positive.
The nominal 75% local-orientation request rate did not become 75% actual local
frames: raw-sign wrap-cell restrictions caused many requests to fall back to
global cells. Actual orientation counts are retained, so this is not presented
as a controlled local/global comparison.

The best below-leader saved GEOS witnesses from the first two strata have
48.51 and 41.80 billionths normalized maximum overlap deficit, each with eight
violated pairs. They are weaker by that metric than earlier known witnesses.
This gate therefore does not justify a larger run of the same allocation
policy, and no extension was launched. It does not prove that continuous cost
predicts integer quality or that every higher-cost branch is unproductive.

The [compact result](branch-strata-results.json) retains inputs/source hashes,
counts, coverage, actual stresses, orientation mix and the best stored frontier
witnesses. Full inputs, source snapshots, per-branch records and geometry data
remain in the ignored output directory above. The final runner matches its
frozen source hash; no API requests or global promotion occurred.
