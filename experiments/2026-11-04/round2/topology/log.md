# Round 2: contact-topology surgery

Goal: change the packing contact graph through explicit removal and reinsertion, rather than repeat local Gaussian basin perturbations. Outputs stay in this directory; `solution.json` will not be edited. Solver blocks are bounded to180seconds.

## Step 1 — audit the old basin methods

Commands:

```sh
sed -n '1,300p' scripts/experiment_basins.py
sed -n '1,260p' scripts/experiment_compression.py
sed -n '1,260p' scripts/experiment_symmetry.py
sed -n '1,260p' scripts/experiment_profile_csp.py
sed -n '1,300p' src/semicircle_packing/_fast_geometry.cpp
rg --files src/semicircle_packing
```

Two speculative reads (`src/semicircle_packing/optimizer.py` and `src/semicircle_packing/search.py`) returned missing files; implementation is organized differently. No action depended on them.

Findings: old basin moves already include centroid-preserving rotations, pi flips, center swaps, Gaussian displacement, cluster twists, full random starts, and container expansion/compression. Repeating those would not establish new coverage. They do not change individual shape size or optimize a13-shape skeleton after explicitly removing two shapes.

## Step 2 — shrink/remove, recompress, reinsert, regrow

Hypothesis: the jammed15-shape contact network prevents large rearrangements that become accessible after two shapes are removed. Reoptimizing the13-shape skeleton can erase the original cavities. Reinserting small ghosts in different pockets and growing them while the container temporarily stays expanded can reach a different15-shape contact graph.

Implementation plan:

1. Choose a removed pair, beginning with the two opposed-flat pairs and then covering other pairs.
2. Optimize the remaining13 unit semicircles using exact separating-axis geometry.
3. Search global candidate pockets for two replacement semicircles, scoring hypothetical full-size insertions against the frozen skeleton.
4. Insert the replacements at radius0.15 and grow through several sizes to1 while relaxing all shapes in an expanded container.
5. Jointly recompress all15 full-size semicircles, then use the original polygon-area polishing/checker on promising minima.

A private C++ kernel will generalize the existing support-function constraints to per-shape radii. The native unit-radius path will be cross-checked against the existing kernel before search. No shared source file will be changed.

## Step 3 — implement and audit generalized exact geometry

Created `scripts/experiment_surgery.py`. Its private C++ support-function kernel accepts a fixed radius for every shape and retains all three containment rows per semicircle. It compiles to the system temporary directory, not the repository core. The search records every removal pair, skeleton radius, pocket score, regrowth stage, final radius/residual, and contact-graph WL signature.

Command, audit seed81104:

```sh
.venv/bin/python scripts/experiment_surgery.py --audit-only
```

Step3 audit result: unit-radius constraint and Jacobian errors were exactly zero across60 random configurations at n=3,13,15. Unequal-radius finite-difference Jacobian error was2.108e-9 (h=1e-7). The WL contact signature implementation was corrected before searching to hash full iterative descriptors, avoiding accidental dependence on graph-local color numbering.

## Step 4 — first bounded surgery block

Command, seed8110401, temporary container radius3.12,512 candidate pockets per insertion:

```sh
.venv/bin/python scripts/experiment_surgery.py --seconds 180 --seed 8110401 --container 3.12 --pockets 512
```

The pair schedule begins(1,5),(6,10), then a seeded permutation of the other103 pairs. Each ghost grows through radii[0.15,0.35,0.55,0.75,1]. The final strict packing and promising polygon-area-polished packing are both retained. Novel graph signatures are descriptive diagnostics, not proofs of non-isomorphism or global basin completeness.

## Step 5 — inverse design through a full disk

Hypothesis: a full unit disk can be split into two complementary unit semicircles along any diameter while its union stays exactly fixed. Optimizing13 semicircles plus a disk, choosing a new internal diameter, and releasing the halves changes which external contacts belong to each half at zero occupied-area cost. This is more targeted than random regrowth.

The private geometry kernel now supports full disks as well as semicircles; default all-semicircle behavior is unchanged. Added `experiment_surgery_split.py`, which optimizes fused disk configurations, enumerates diameter partitions between external contact directions plus an eight-angle grid, and releases the halves at tangential staggers0,+/-0.3,+/-0.85. Nonzero staggers are relaxed in a temporarily expanded container before recompression. No Gaussian displacement is used.

Commands:

```sh
.venv/bin/python scripts/experiment_surgery.py --audit-only
.venv/bin/python scripts/experiment_surgery_split.py --audit-only
```

The first audit is repeated after extending the kernel. A separate disk/semicircle derivative audit checks the new object type before search.

## Step 6 — bounded disk-fission block

Both generalized-geometry audits passed: unit values/Jacobians remain exact matches; unequal-radius and disk derivative errors are2.108e-9. Disk support checks found no positive separation inconsistency in50 randomized complementary-pair comparisons.

Command, seed8110402,180-second bound:

```sh
.venv/bin/python scripts/experiment_surgery_split.py --seconds 180 --seed 8110402 --container 3.12
```

The first regrowth block finished normally. Its complete records are in `trials.jsonl`; summarized metrics are in `summary-regrow.json`. It returned the known continuous optimum rather than a better basin. Detailed numbers are recorded in that summary so that differing contact signatures are not mistaken for better radii.

## Step 7 — area-preserving unequal cleavage

Hypothesis: equal-half disk fission may still miss paths opened by temporarily moving material between the two pieces. A disk cut at signed offset a produces circular segments with areas A(a)=acos(a)-a*sqrt(1-a*a) and A(-a)=pi-A(a). Their total area remains exactly pi for every a. At coincident centers and opposite normals their union is exactly the original disk, even for an off-center cut. Once the two pieces are released and recompressed, moving a toward zero transfers area from one piece to the other and ends at two unit semicircles.

The private kernel was generalized to circular segments. Its support is1 when u·n>=a, otherwise a*(u·n)+sqrt(1-a*a)*abs(u·t). The same flat normals and circle/endpoint stationary axes still give exact separating-axis constraints. Defaults a=0 recover semicircles; the unit and full-disk audits were rerun unchanged.

Added `experiment_surgery_caps.py`. It checks nonzero-cut derivatives, overlap signs against2049-point Shapely polygons, analytic containment against sampled boundaries, and complementary-segment union areas before search.

Command, audit seed811042:

```sh
.venv/bin/python scripts/experiment_surgery_caps.py --audit-only
```

Step7 circular-segment audit passed: max Jacobian error2.108e-9; zero overlap-sign mismatches against200 sampled polygon pairs; containment undershoot below9e-16; complementary-segment union-area error1.576e-7 from polygon discretization. The underlying analytic area sum is exactly pi.

## Step 8 — bounded area-transfer homotopy block

Command, seed8110403,180-second bound:

```sh
.venv/bin/python scripts/experiment_surgery_caps.py --seconds 180 --seed 8110403
```

It uses disk-packing seeds from the previous block, chord offsets+/-0.8 and+/-0.4, tangential staggers+/-0.3 and+/-0.85, and five area-transfer stages ending at offset0. Every intermediate pair keeps total area pi. Each stage is relaxed and recompressed before proceeding; all final shapes are unit semicircles.

## Step 9 — independently audit topology equivalence

The original surgery signature is already invariant under shape permutation and reflection: its initial labels are(degree,boundary-contact count), each WL round sorts neighbor labels, and its final hash sorts all node labels. No shape index or orientation sign is included. Different signatures cannot be explained solely by relabeling shapes. Finite contact thresholds and incomplete optimization can still inflate counts of physically meaningful packing basins, so an independent audit is necessary.

NetworkX was absent. Installed networkx3.7 into the existing venv with a writable temporary cache:

```sh
UV_CACHE_DIR=/private/tmp/shape-alt-uv-cache uv pip install --python .venv/bin/python networkx
```

Added `experiment_surgery_graphs.py`. It reconstructs all archived strict-feasible graphs, verifies each original hash, applies a random shape permutation and a physical reflection to every packing, then clusters by NetworkX graph isomorphism. It records both uncolored contact graphs and graphs whose nodes carry boundary-contact counts. WL buckets only prefilter comparisons; exact isomorphism determines the final class counts. Audit seed811043; contact threshold3e-6.

Command:

```sh
.venv/bin/python scripts/experiment_surgery_graphs.py
```

Step9 equivalence result: the original1402 signatures were already permutation/reflection invariant, but11 belonged to stalled/infeasible trials. All1391 archived feasible graphs reconstructed their original hashes exactly. All1391 random shape permutations and physical reflections preserved the hashes. Exact NetworkX isomorphism clustering gives1391 classes when boundary-contact counts are node labels, or1338 plain unlabeled contact-graph classes. Thus coverage should be reported as1338 unlabeled feasible contact graphs, not1402 certified physical basins. Contact-threshold and optimizer-tolerance limitations remain explicit.

Step6 disk-fission block finished normally. Its complete split/fusion records and summary are retained in `split/trials.jsonl` and `summary-split.json`. It again reached the known strict/continuous optimum and produced no valid rounded improvement.

## Step 10 — finite untested extension and final interpretation

No more solver blocks will be started after the active unequal-cleavage block ends.

One genuinely untested finite extension is joint two-disk fission: optimize11 semicircles plus2 full unit disks; enumerate both disks' diameter contact partitions together; release all4 halves simultaneously. A compact future schedule could retain6 distinct macro packings, at most8 contact partitions per disk, and3 tangent stagger choices per pair, giving at most6*8*8*3*3=3456 joint releases. This campaign only fused/split one disk at a time, so it did not test that coupled move. This is a proposal, not a result or a guarantee of useful coverage.

The negative inference is limited: no lower-radius basin has appeared within the tested removal/regrowth, one-disk fission, and area-transfer families. Contact graphs are measured with finite geometric thresholds; some feasible outputs stopped at iteration limits. These results do not prove a global optimum or identify the leader's basin.

Added `experiment_surgery_summary.py` to combine trial counts, feasibility, equivalence audits, environment/script/input hashes, and an independent read-only geometry alignment of competitive outputs against the known continuous packing. Alignment permits shape permutations, rigid rotation/translation, and reflection. It quantifies similarity without treating matching radii as proof of equal geometry.

Final summary command, after the active block ends:

```sh
.venv/bin/python scripts/experiment_surgery_summary.py
```

## Step 11 — equal-radius geometry audit corrects radius-based deduplication

The read-only alignment found that the removal/regrowth best is not congruent to the reference: center-distance and centroid-distance multisets differ by0.369205 and0.062379, invariants under rigid motion, reflection, and shape permutation. Repeating alignment with225 orientation-anchor starts did not remove the discrepancy. The area-active contact graphs are also non-isomorphic, with and without boundary labels. Matching radii therefore did not establish the same basin; earlier shorthand about returning the same basin was unjustified for this output.

`experiment_surgery_patch.py` identifies the mechanism:12 pieces match after a global rotation within1.1microunits; the other3(source indices1,12,14, reference set1,5,14) match under a local reflection with maximum center error9.8e-10 and angle error4.4e-9. Its reflection axis is approximately a container diameter. Reflection preserves distances from the enclosing center; cross-patch contacts rewire. The full-precision source was preserved unchanged. A relabelled copy maps opposed-flat pairs to(1,5),(6,10) for old lattice scripts; `noncongruent-seed-audit.json` records the mapping and source hash.

Commands:

```sh
.venv/bin/python scripts/experiment_surgery_patch.py
.venv/bin/python scripts/experiment_surgery_summary.py
```

## Step 12 — newly authorized finite reflection inventory

The root agent authorized one final60-second inventory after the patch mechanism was established. No further extension follows it.

`experiment_surgery_inventory.py` enumerates connected1–4-piece subsets and diameter axes that exchange equal-radius rim centers, plus radial axes. A known3-piece flip across the bisector of rim centers3/4 is an explicit positive control. Native polygon areas prefilter; cross-patch GEOS intersections and a final full GEOS area check decide feasibility. Numerical correction is allowed only inside a fixed radius budget of referenceR+1e-9, with each coordinate adjustment bounded to20nanounits. Saved seeds must differ from the baseline in rigid/permutation/reflection-invariant pair-distance data by more than1e-5. Duplicate invariant spectra are conservatively merged.

Command:

```sh
.venv/bin/python scripts/experiment_surgery_inventory.py --seconds 60
```

## Final inventory and campaign result

The finite inventory finished in1.157seconds, well inside its60-second bound. Connected subsets:15 of size1,30 of size2,72 of size3,178 of size4; total295. There were45 distinct candidate diameter axes, yielding13275 tested candidates. Native polygon-area checks rejected13273. GEOS accepted2: one baseline-invariant match and the known3-piece flip. No additional distinct noncongruent seed was found in this inventory.

Positive control passed. `inventory/seed-001.json` has maximum full GEOS overlap area9.999998033568569e-7 and MEC radius2.948526549763155. Its numerical correction moved at most0.14737nanounits. Baseline correction moved at most0.29614nanounits. Both stayed below the common referenceR+1e-9 radius limit. Accepted-seed feasibility is decided by GEOS; native prefiltering and the finite axis/subset family limit what can be inferred about rejected or unenumerated transformations.

The three180-second surgery blocks completed5600 trials. No valid six-decimal improvement was produced. They did uncover a genuinely different equal-radius contact topology, hidden by the prior score-only archive filter. Its full-precision source remains unchanged at `continuous-best.json`, SHA256 `88d33e4a36c1406de98c02c41caccfdaa45934e4eb877f518da0923b2ab8eacd`. The noncongruence, graph-isomorphism, patch-reflection, GEOS positive-control, and complete inventory results are preserved in their named audit JSON files and combined into `report.json`.

All owned runner sessions exited normally. No further experiment follows this inventory. `solution.json` was not written by this agent. The central conclusion is a new equal-radius embedding for lattice searches, not a proof of global optimality or a lower bound on six-decimal rounding cost.
