# Round3 mirror-family lattice search

Objective: find a valid six-decimal improvement, ultimately below2.9485269238856113. Preserve all earlier full-precision mirror seeds and `solution.json`.

## Step1 — inspect and reuse the curved-contact model

Read `experiment_alt_manifold.py`, the updated generic-pair `experiment_lll.py`, `experiment_family_lattice.py`, `experiment_breakout.py`, and `breakout_geometry.py`. The existing native batch evaluator returns nonlinear polygon areas and free-center MEC scores; unsupported MEC supports fall back to the convex solver.

The mirror geometry has41 active contacts, flat pairs(1,14),(4,8), and endpoint-sensitive piece12. Its continuous radius is2.9485265494161608. Current raw flat differences are[3.141569507835555,3.1415934401103573].

## Step2 — exact manifold and seam-aware integer branches

`experiment_mirror_family.py` reuses the earlier Newton contact-manifold construction:41 contact equations plus two fixed-flat equations; project global rotation out of the three-dimensional null space; parameterize the remaining two flexes. Each branch/soft-coordinate section retains positive contact stress and full nonlinear feasibility before integer search.

Positive and negative raw angle differences define interlaced lattices because2pi is not an integer number of microradians. All four raw-sign cells are sampled deliberately; mixed-sign cells are narrow and uncommon under uniform global rotation. The central raw branches are(+3.141570,+3.141593),(+3.141570,-3.141592),(-3.141616,+3.141593),(-3.141616,-3.141592). Neighbor branches can be enabled.

Frames alternate free endpoint angle, endpoint12 alignment to its integer grid, and explicit endpoint seam values when the raw-sign cell permits them. This alignment does not remove endpoint12 from the45-dimensional lattice. Every frame also samples a submicro-unit global translation.

Submitted raw6dp angles are retained separately from their once-canonicalized geometry. An audit compares vector canonicalization against the scalar reference, including±pi edge cases; checker and API receive the raw input.

## Step3 — smoke test

Command, seed3110401:

```sh
.venv/bin/python scripts/experiment_mirror_family.py --seconds 20 --smoke --samples 128 --output experiments/2026-11-04/round3/mirror/smoke
```

Smoke invocation1 stopped at parsing: a compact ternary expression omitted whitespace after `else`. Corrected that syntax and added reusable seam-cell frame exports. Each exported frame already has both chosen raw flat differences exactly on the microradian grid, making it suitable for a43-coordinate tied-angle lattice. The root agent is implementing that independent reduced-coordinate comparison.

Smoke result:8 frames,1024 proposals in0.340seconds; all8 sections feasible within9.9e-13 and positive stress; no canonicalization mismatch or native MEC fallback. Frame000–003 cover all four raw-sign cells at the center section, and004–007 at alpha=(0.0008,0). Exact raw-flat frame differences agree with6dp requested values within1e-15. These frames were sent to the root agent for an independent43-coordinate tied-angle lattice test.

## Step4 — bounded batched campaign

Command, seed3110401,300-second bound:

```sh
.venv/bin/python scripts/experiment_mirror_family.py --seconds 300 --samples 512 --neighbor-branches --output experiments/2026-11-04/round3/mirror/campaign-a
```

This covers36 nearby raw branch pairs across13 exact curved-family sections. Every frame evaluates true nonlinear areas and a free-center MEC in the batch kernel. Only original-checker-valid improvements are sent to the arena checker; no leaderboard submission operation exists in this workflow.

## Step5 — bootstrap a valid mirror-grid foothold, then continue its score

Strict incumbent caps can discard valid states of the new topology that start slightly above the old topology's best score. Added `--bootstrap`: its first radius target may be looser; once an original-checker-valid mirror state is found, subsequent frames target that topology's best valid radius. Global improvements still require an arena check. This supplies valid mirror starts for the root agent's neutral-network search even before beating the old incumbent.

Command, seed3110402,300-second bound, initial target2.94852712 and boundary bias1:

```sh
.venv/bin/python scripts/experiment_mirror_family.py --seconds 300 --samples 512 --seed 3110402 --target 2.94852712 --boundary-bias 1 --bootstrap --amplitude .0002 --output experiments/2026-11-04/round3/mirror/campaign-b
```

CampaignA remains unchanged in its running process as the tighter-target, broader-family comparison. CampaignB uses smaller flex amplitudes so curvature spends less of the radius budget.

Bootstrap result: a new-topology6dp foothold passed the arena checker at radius2.9485270260454817. Its immutable raw snapshot and API response are `campaign-b/foothold-api-input.json` and `foothold-api.json`. Continuation then improved the mirror-only score further; the live best and checkpoint remain separate from that API snapshot.

## Step6 — valid-network descent on the mirror foothold

The root agent authorized pursuing continuation/valid-network descent. Reused its existing `experiment_neutral_network.py` in a private `experiment_mirror_network.py` copy. Changes are limited to an optional explicit basis checkpoint and a generic endpoint index(default12) for the optional conditional-angle repair. The run uses a snapshot of CampaignB's valid mirror state and its matching basis, rather than the original topology's hardcoded basis files. Every accepted state still passes native GEOS checks; improvements are reference/API checked.

Command, seed3110403,300-second bound:

```sh
.venv/bin/python scripts/experiment_mirror_network.py --input experiments/2026-11-04/round3/mirror/network/start-raw.json --basis experiments/2026-11-04/round3/mirror/network/start-state.npz --endpoint 12 --output experiments/2026-11-04/round3/mirror/network --seconds 300 --seed 3110403 --relay 2e-7 --repair-every 300
```

## Step7 — API-verified improvement and stronger network restart

CampaignB found an actual API-valid improvement at255.21seconds: radius2.948526962953499, improving the old incumbent by3.771868nanounits and leaving39.067888nanounits to first. It came from section32, alpha=(-0.00014,-0.00014), raw flat differences(+3.141570,+3.141593). Raw artifact: `campaign-b/valid-best-raw.json`; proof: `campaign-b/api-001.json`. The root agent independently rechecked it and promoted a stable snapshot at `round3/candidate-2.948526962953499.json`.

At the root agent's request, the older valid-network run starting at2.948527005039101 was interrupted with Ctrl-C(exit130); its logs, initial/best/current snapshots, and accepted-transition records remain intact. A new start/checkpoint was copied from CampaignB and checked for exact micro-unit agreement with the root's promoted snapshot and section32 alpha/branches.

Command, seed3110404,300-second bound:

```sh
.venv/bin/python scripts/experiment_mirror_network.py --input experiments/2026-11-04/round3/mirror/network-best/start-raw.json --basis experiments/2026-11-04/round3/mirror/network-best/start-state.npz --endpoint 12 --output experiments/2026-11-04/round3/mirror/network-best --seconds 300 --seed 3110404 --relay 1e-7 --repair-every 100 --repair-steps 3
```

## Step8 — focused valid-prior continuation

Added `--near-valid` and `--focus-only`. This uses the promoted grid point as the Babai movement prior and concentrates global-frame sampling near its fitted pose, with angular jitter scales1e-7,1e-6,1e-5,1e-4 radians. One-third of frames retain global cell exploration. Exact raw-flat branches remain fixed, and endpoint12 remains a free integer coordinate. The continuous starting origin is copied from the matching section32 checkpoint, not regenerated from rounded input.

Smoke command, seed3110405:

```sh
.venv/bin/python scripts/experiment_mirror_family.py --input experiments/2026-11-04/round3/mirror/focus/origin.json --near-valid experiments/2026-11-04/round3/mirror/network-best/start-raw.json --focus-only --seconds 20 --smoke --samples 128 --seed 3110405 --target 2.9485269628 --boundary-bias 1 --amplitude .00005 --output experiments/2026-11-04/round3/mirror/focus-smoke
```

Focused smoke recovered the promoted incumbent as its nearest lattice point (native radius differs from the reference by1.2e-13), confirming prior/origin consistency. Some deliberately global frames produced unsupported MEC fallbacks because their unrotated valid-point prior was incompatible with a distant frame. In `--focus-only` mode all frames now stay near the fitted valid pose; distant endpoint-seam overrides are disabled. Endpoint grid alignment remains available and shifts the frame by at most half a microradian.

## Step9 — focused score campaign

Command, seed3110405,300-second bound:

```sh
.venv/bin/python scripts/experiment_mirror_family.py --input experiments/2026-11-04/round3/mirror/focus/origin.json --near-valid experiments/2026-11-04/round3/mirror/network-best/start-raw.json --focus-only --seconds 300 --samples 512 --seed 3110405 --target 2.9485269628 --boundary-bias 1 --amplitude .00005 --output experiments/2026-11-04/round3/mirror/campaign-c
```

## Correction and focused-run result interpretation

The earlier status saying the stronger-incumbent valid walk accepted center hops was incorrect: that conflated the older .527005-start run with the .962953499 run. The latter completed with59 accepted neutral-angle transitions, zero center hops, and zero conditional solutions. Repeated `target_hop` events were proposals, not accepted states. No rerun of that walker is warranted without target novelty/tabu.

The root independently GEOS-checked focusedCampaignC's nearest snapshot: it is the exact incumbent, valid at2.948526962953499. The reported0.1536nm residual is merely the incumbent sitting above the slightly tighter target, not a new improvement. Native area error8.30e-13 was below the prior2e-12 guard. Future native guards are broadened to1e-11; GEOS remains authoritative.

## Step10 — fresh random curved sections and native batched Babai

New root hypothesis: thirteen fixed soft-coordinate sections are sparse. Sample alpha fromN(0,sigma²I) freshly for every frame, sigma about0.0004; solve its exact-D nonlinear manifold section, then decode integer proposals natively. This balances phase coverage against quadratic radius cost. The fast-geometry agent supplied a verified native Babai decoder and43-dimensional tied-angle frame implementation; the manifold consumer reuses both, with all four raw-sign cells sampled equally.

The new `--random-sections` route retains exact raw angle differences, chooses fresh submicro translations, verifies128 Babai coefficients on each of its first four frames, and archives distinct reference-valid XY elites even above the global best. Proposal target2.94852698 is intentionally looser than the incumbent; actual score still decides promotion.

Smoke command, seed3110406:

```sh
.venv/bin/python scripts/experiment_mirror_family.py --random-sections --native-dimension 43 --sigma .0004 --target 2.94852698 --seconds 20 --samples 2048 --boundary-bias 1 --seed 3110406 --smoke --output experiments/2026-11-04/round3/mirror/random-smoke
```

FocusedCampaignC finished300seconds/3,034,112 proposals with no improvement; it repeatedly recovered the incumbent as a positive control. The native random-section smoke passed512 coefficient comparisons and16,384 proposals, then its final event emitter hit a duplicate `seconds` key. Corrected that reporting-only emitter error before the campaign. JSON serialization checks confirmed the saved metadata was valid.

Native43 smoke v2 passed8 fresh sections,16,384 proposals,13,119 deduplicated evaluations in0.665seconds; all512 audited Babai coefficients matched fpylll. No MEC fallback occurred. Raw-wrap cell crossings were counted and filtered for this strict-family control.

The root authorized three complementary300-second native campaigns concurrently, one BLAS thread per job, with sigmas0.0002,0.00035,0.0005 and target2.94852698. The wider third run uses45 integer coordinates and keeps seam-crossing proposals as additional candidates; every accepted result still requires original GEOS validation. The two43-coordinate runs retain the exact canonical raw-cell family as controls.

A45-dimensional native smoke check precedes those runs:

```sh
.venv/bin/python scripts/experiment_mirror_family.py --random-sections --native-dimension 45 --sigma .0005 --target 2.94852698 --seconds 20 --samples 2048 --boundary-bias 1 --seed 3110409 --smoke --accept-seam-crossings --output experiments/2026-11-04/round3/mirror/random45-smoke
```

Native45 smoke passed512 coefficient comparisons,16,384 proposals,13,678 unique nonlinear evaluations in0.690seconds, with no MEC fallback. Filtering now happens before insertion into the global deduplication cache, so a rejected raw-cell crossing cannot suppress a later eligible occurrence.

## Step11 — authorized three-job random-section campaign

Each process sets OpenBLAS/OMP/Accelerate thread counts to1. These jobs are independent local computations, with no service or external spending.

```sh
.venv/bin/python scripts/experiment_mirror_family.py --random-sections --native-dimension 43 --sigma .0002 --target 2.94852698 --seconds 300 --samples 2048 --boundary-bias 1 --seed 3110407 --output experiments/2026-11-04/round3/mirror/random-d
.venv/bin/python scripts/experiment_mirror_family.py --random-sections --native-dimension 43 --sigma .00035 --target 2.94852698 --seconds 300 --samples 2048 --boundary-bias 5 --seed 3110408 --output experiments/2026-11-04/round3/mirror/random-e
.venv/bin/python scripts/experiment_mirror_family.py --random-sections --native-dimension 45 --sigma .0005 --target 2.94852698 --seconds 300 --samples 2048 --boundary-bias 1 --seed 3110409 --accept-seam-crossings --output experiments/2026-11-04/round3/mirror/random-f
```

## Step12 — neighbor-branch audit and matched companions

The archived F45 valid elites showed only the four central raw branches. Full candidate histograms had not been retained, so that retrospective audit is explicitly limited to saved valid snapshots. Algebra shows rare seam crossings can produce the neighbors: rounded2pi=6283185microradians maps+3141570 to-3141615, and-3141616 to+3141569. Such crossings use a tangent origin from the other physical branch; they do not replace a continuously solved matched-branch campaign.

Eight exact alpha0 sections in `branch-audit.json` show+3.141569 is essentially equal in radius to+3.141570 within the section solver's numerical scale; -3.141615 costs about0.101nm more than-3.141616. Each has positive stress. The second raw difference-3.141592 is about1.144nm cheaper continuously than+3.141593, but grid scores can reverse that ranking.

The root authorized three300-second matched-neighbor jobs. Added explicit raw-branch options, an optional unlocked45-dimensional flat metric, and full submitted/canonical raw-difference histograms for every decoded proposal before deduplication or branch filtering. Each job records actual valid score and distinct-XY elites, rather than comparing only valid counts.

Smoke for the new unlocked neighbor route, seed3110412:

```sh
.venv/bin/python scripts/experiment_mirror_family.py --random-sections --native-dimension 45 --sigma .0005 --target 2.94852698 --seconds 20 --samples 2048 --boundary-bias 2 --seed 3110412 --raw-first=3.141569,-3.141615 --native-flat-weight 0 --accept-seam-crossings --smoke --output experiments/2026-11-04/round3/mirror/neighbor-smoke
```

The unlocked-neighbor smoke passed512 coefficient comparisons and16,384 proposals. As expected, removing the extra flat-angle metric produces many more raw-difference changes and worse short-run violation; it is a complementary proposal family, not a replacement for exact ties.

## Step13 — authorized matched-neighbor campaigns

```sh
.venv/bin/python scripts/experiment_mirror_family.py --random-sections --native-dimension 43 --sigma .0002 --target 2.94852698 --seconds 300 --samples 2048 --boundary-bias 1 --seed 3110410 --raw-first=3.141569,-3.141615 --output experiments/2026-11-04/round3/mirror/neighbor-g
.venv/bin/python scripts/experiment_mirror_family.py --random-sections --native-dimension 43 --sigma .00035 --target 2.94852698 --seconds 300 --samples 2048 --boundary-bias 5 --seed 3110411 --raw-first=3.141569,-3.141615 --output experiments/2026-11-04/round3/mirror/neighbor-h
.venv/bin/python scripts/experiment_mirror_family.py --random-sections --native-dimension 45 --sigma .0005 --target 2.94852698 --seconds 300 --samples 2048 --boundary-bias 2 --seed 3110412 --raw-first=3.141569,-3.141615 --native-flat-weight 0 --accept-seam-crossings --output experiments/2026-11-04/round3/mirror/neighbor-i
```

G and H preserve the D/E sigma/bias settings, respectively, while changing the first-pair branch. I intentionally unlocks relative-angle proposals and keeps all seam crossings, with an intermediate boundary bias; its result is complementary and confounded, not a controlled sigma comparison. Full decoded raw-difference histograms are retained.

## Step14 — requested broad branch-cost map before any broad search

Added a research-data map of exact Newton sections at alpha0, with first-pair raw offsetsk=-8..8 and second-pair offsetsl=-4..4 on both positive/negative raw cells. A section counts only if every native inequality and the contact-system residual are below2e-11 and contact stress is positive. Radius costs are measured against the same unconstrained mirror seed. JSON/CSV preserve all tested values; accepted branches also have full-precision seam-cell frame seeds. This maps actual nonlinear sections rather than assuming a quadratic branch cost. It does not launch an integer search.

Command:

```sh
.venv/bin/python scripts/experiment_mirror_branch_map.py --seconds 120
```

## Step15 — broad low-cost fiber campaigns

The correct constrained stress usesJ@T. All612 mapped exact sections have positive constrained stress (minimum0.02134);49 cost<5nm,222<20nm,357<50nm. Full-J weights incorrectly rejected306 feasible constrained sections. The consumer now applies the transformed stress test before building43-coordinate models, matching the root's corrected frame implementation.

Branch selection is weighted heuristically by remaining radius slack to power40, favoring low-cost fibers while retaining many discrete first/second-angle branches. The following authorized300-second jobs run independently:

```sh
.venv/bin/python scripts/experiment_mirror_family.py --random-sections --native-dimension 43 --sigma .0002 --target 2.94852698 --seconds 300 --samples 2048 --boundary-bias 1 --seed 3110413 --branch-map experiments/2026-11-04/round3/mirror/branch-map/branches.json --max-branch-cost 5 --output experiments/2026-11-04/round3/mirror/broad-j
.venv/bin/python scripts/experiment_mirror_family.py --random-sections --native-dimension 43 --sigma .00035 --target 2.94852698 --seconds 300 --samples 2048 --boundary-bias 1 --seed 3110414 --branch-map experiments/2026-11-04/round3/mirror/branch-map/branches.json --max-branch-cost 20 --output experiments/2026-11-04/round3/mirror/broad-k
.venv/bin/python scripts/experiment_mirror_family.py --random-sections --native-dimension 43 --sigma .00035 --target 2.94852698 --seconds 300 --samples 2048 --boundary-bias 5 --seed 3110415 --branch-map experiments/2026-11-04/round3/mirror/branch-map/branches.json --max-branch-cost 20 --output experiments/2026-11-04/round3/mirror/broad-l
```

The matched-neighbor runs produced no incumbent improvement. Best valid G was2.9485269864331265, compared with D2.9485269867883153; H2.9485270278417763 compared with E2.9485270260454817. These differences are observations, not a statistically controlled sigma result. Unlocked I produced no valid relay. G's near-witness had one actual overlap at0/9 with actual radius2.948526956011455; direct theta9+1,+2,+3microradian probes replaced that overlap by4/9 and8/9 overlaps.

## Step16 — completed broad fibers and exact angular repair coordination

J/K/L completed their bounded300-second blocks with8,084,954 /8,083,661 /7,884,588 unique nonlinear evaluations. No valid improvement was found. J/K wrote31/24 valid elites; L wrote none. Their scalar nearest deficits were8.0803/8.5897/9.0319nm. L's nearest raw snapshot has actual free-MEC radius2.94852690780946,16.08nm below first, but four actual GEOS overlaps at(1,2),(1,14),(5,6),(5,7); maximum normalized excess9.0319nm. It is an invalid rescue witness, not a score improvement.

The arena agent completed G's exact fixedXY free-center angle-domain repair, all angles±20microradians. Its finite-domain model is infeasible, with pair-only core(0,9),(8,9), and recovered the valid incumbent as positive control. This excludes that specific finite repair family; it does not exclude XY moves. Artifacts are in `round3/conflicts/neighbor-g-angle20.*`. No duplicate angular campaign was launched.

## Step17 — online radius/overlap archive and audit

Root hypothesis: scalar nearest-to-target snapshots erase useful low-radius states. Added private `experiment_mirror_frontier.py`, consumed through `--frontier`. It tracks1nm radius bins from first−60nm to first+45nm and keeps separate GEOS overlap and violated-pair champions. Candidate retention is heuristic: initial native overlap guard100nm, native-under25nm contenders can challenge within10nm of the current GEOS champion, and sufficiently improving/fewer-pair candidates can challenge more broadly. This controls archive overhead only; it never removes a proposal from normal search validation. Every retained point stores original raw JSON, source fullprecision frame, exact matching integer moves, anchor, coefficients, once-canonical geometry, free MEC, scale, and all105 actual GEOS overlap areas. The manifest also gives the global radius/overlap Pareto subset. These finite records are not an exhaustive search certificate.

Fixture audit retained the incumbent (zero overlaps) and L (four overlaps,9.031892nm). Stored geometry exactly reproduced the once-canonicalized raw input, and stored GEOS areas exactly matched reevaluation. A preliminary exact-float equality assertion on six-decimal floating-point coordinates was too strict; replacing it with a1e-14 coordinate audit confirmed the intended grid reconstruction. No search candidate changed because of that assertion.

Integration/overhead commands, same seed3110416 and settings:

```sh
.venv/bin/python scripts/experiment_mirror_family.py --random-sections --native-dimension 43 --sigma .0002 --target 2.94852698 --seconds 20 --samples 2048 --boundary-bias 1 --seed 3110416 --branch-map experiments/2026-11-04/round3/mirror/branch-map/branches.json --max-branch-cost 5 --frontier --output experiments/2026-11-04/round3/mirror/frontier-smoke
.venv/bin/python scripts/experiment_mirror_family.py --random-sections --native-dimension 43 --sigma .0002 --target 2.94852698 --seconds 20 --samples 2048 --boundary-bias 1 --seed 3110416 --branch-map experiments/2026-11-04/round3/mirror/branch-map/branches.json --max-branch-cost 5 --output experiments/2026-11-04/round3/mirror/frontier-control
```

With archive:376,965 unique evaluations/20.017s,141 GEOS records across60 radius bins; GEOS5.719s. Control:549,759/20.035s. Initial archive throughput cost31.4%; the first218 frames were identical proposals and recovered the same nearest events/valid elite. All512 native Babai coefficient audits passed. All141 archived points exactly reconstructed from stored anchor+coefficients@moves, followed by the same translation quotient and single canonicalization. No native-under20nm point in the configured radius window appeared in the archive smoke. Initial bin filling contributes much of the GEOS cost; future campaigns should report archive overhead separately.

API requests now bind each response to an immutable `api-NNN-input.json` and its SHA256, while preserving the earlier convenience candidate path.

## Step18 — learned slack-allocation mixture preparation

Root hypothesis: contact slack proportions in valid/near-valid layouts transfer across rigid rotations and fresh lattice phases. Root is independently preparing training data with per-contact weighted gaps evaluated at each source's actual free-MEC center/radius, constrainedJ@T stress, and global active-row labels. Private `experiment_mirror_slack.py` checks those labels against every fresh frame. It mixes80% Dirichlet samples centered on learned patterns (2% uniform pseudocount) and20% the existing unbiased boundary-weighted Dirichlet proposals. Total Dirichlet concentration defaults to the previous frame's active-count×concentration schedule. Each fresh Frame.budget rescales the proportions. This changes proposal probabilities only and keeps all geometry/GEOS checks.

No blind linear/quadratic geometry rejection cutoff has been added. A separate conservative disk-lens certificate remains a possible throughput experiment after the archive and mixture audit, requiring mismatch/coverage measurements before use.

Root supplied `round3/slack-patterns/patterns.npz`, SHA256`998871b8b8833bc4ded185b9fc7adf8b3402b54cdcbb32276bf289a565ad51db`:175 independently reference-valid patterns plus2 separately labeled invalid G/L near-witnesses. Default sampler filters to the175 valid rows. Root's24 rigid-rotation audits had maximum fraction discrepancy6.8514e-7; normalized row-sum error2.2e-16. A100,000-draw sampler audit measured79.83% learned choices, empirical mean discrepancy0.0002233, row-sum discrepancy8.88e-16, and verified that mismatched active-row IDs are rejected. The trained43 smoke passed512 Babai coefficient comparisons including one near-half ambiguity patch. A synthetic fixture was used only to audit the kernel path and is explicitly not training data.

Root approved coalescing distant frontier nominees. The revised archive copies raw input, geometry, scale, source frame, moves, anchor, and coefficients immediately at nomination, before the frame can be replaced. For native deficits≥25nm it retains the best overlap and count contenders per radius bin over ten seconds before GEOS checking; native-under25nm challenges remain immediate. Final shutdown flushes all pending nominations. A delayed GEOS-valid result still goes through original checker/API validation and is bound to immutable input hashes. This changes archive scheduling only.

## Step19 — matched60-second learned-allocation comparison

Added optional independent `--phase-seed`, so manifold branch/alpha, global rotation, and translation phase use a separate generator from contact proposals. The two following60-second jobs ran concurrently with one BLAS thread each, seed3110419 and phase-seed3110419:

```sh
.venv/bin/python scripts/experiment_mirror_family.py --random-sections --native-dimension 43 --sigma .0002 --target 2.94852698 --seconds 60 --samples 2048 --boundary-bias 1 --seed 3110419 --phase-seed 3110419 --branch-map experiments/2026-11-04/round3/mirror/branch-map/branches.json --max-branch-cost 5 --frontier --output experiments/2026-11-04/round3/mirror/slack-matched-baseline
.venv/bin/python scripts/experiment_mirror_family.py --random-sections --native-dimension 43 --sigma .0002 --target 2.94852698 --seconds 60 --samples 2048 --boundary-bias 1 --seed 3110419 --phase-seed 3110419 --branch-map experiments/2026-11-04/round3/mirror/branch-map/branches.json --max-branch-cost 5 --frontier --slack-patterns experiments/2026-11-04/round3/slack-patterns/patterns.npz --output experiments/2026-11-04/round3/mirror/slack-matched-learned
```

All64 saved fullprecision frames were byte-identical between the two jobs. Baseline:1,382,234 unique nonlinear evaluations in60.084s;12 valid relays, best2.9485270002629176; zero native/GEOS-under20nm archive points; GEOS8.453s. Uniform learned:1,371,218 unique in60.498s;23 valid relays, best2.9485269867883153; one GEOS-under20nm archive point; GEOS7.154s. Neither improved the incumbent. Counts/nearest values are not certificates of statistical superiority; the learned run covered693 frames vs baseline804 due to changed deduplication and sampling costs, so a larger matched repetition would be needed for a rate conclusion.

New learned witness`slack-matched-learned/frontier/witness-00115.json`, raw SHA256`7fa0ea57a746d1d30804830a27158facfdd6ab960141751b2c6c2b82263333ca`, has actual radius2.948526931294496(first+7.4089nm), four GEOS overlaps(0,8),(5,7),(6,7),(6,10), max normalized deficit8.602507nm. The matching sidecar preserves all source basis data. This invalid repair witness was sent to root; it is not a valid score.

Source snapshots and hashes for this matched comparison are in`slack-audit-source/`. The immutable root training report records all raw/sidecar source hashes and actual validation statuses.

## Step20 — quality-weighted learned proposals

Root diagnosed training selection: valid patterns put22.92% mean stress slack on boundary rows, compared with uniform26.83%, incumbent16.28%, and bias5's64.7%. The dataset's median radius is77.6nm above incumbent, so uniform source selection underweights its best pattern(1/175). Added optional quality weighting`p(source)∝exp(-(source_score−minimum_valid_score)/temperature_nm)`, retaining20% unbiased proposals and the same2% uniform within-pattern pseudocount. Invalid G/L patterns remain excluded by default.

At temperature20nm, effective source count is32.959; the best valid pattern receives13.4474% source probability. A100,000-draw audit measured80.176% learned choices, empirical mean discrepancy0.0001729, and normalized finite nonnegative allocation rows. This is a proposal heuristic; every resulting layout still requires GEOS and original/API checks. No300-second quality campaign had started at this preparation step.

## Step21 — adversarial training-reuse audit and fresh quality ablation

Root noticed that the learned relay's best score exactly matched the prior randomD training point and the new near-witness's radius matched old neighborH. Full reuse audit confirmed that score equality was insufficient evidence of new geometry. Although phase-seed3110419 was new and used an independent generator, learned23 saved valid layouts included21 training XY center sets and one exact canonical pose; baseline12 included3 training XY sets and zero exact canonical poses. Best learned relay uses randomD elite0039's centers with different angles. Witness00115 is absent from the training XY table but repeats old neighborH nearest's centers, with angles differing up to4microradians. Hence23vs12 cannot support a claim of improved exploration. Raw/geometry hashes and matched source paths are recorded in`frontier-audit/training-overlap.json`.

Root authorized one fresh300-second quality/control ablation, with the same new phase sequence, sigma.0002 and<5nm branch selection. Phase-seed804173291 and sample-seed804173292 have not been used in earlier campaign commands. Both use one BLAS thread and the same delayed frontier archive. The two launch commands are:

```sh
.venv/bin/python scripts/experiment_mirror_family.py --random-sections --native-dimension 43 --sigma .0002 --target 2.94852698 --seconds 300 --samples 2048 --boundary-bias 1 --seed 804173292 --phase-seed 804173291 --branch-map experiments/2026-11-04/round3/mirror/branch-map/branches.json --max-branch-cost 5 --frontier --output experiments/2026-11-04/round3/mirror/quality-matched-baseline
.venv/bin/python scripts/experiment_mirror_family.py --random-sections --native-dimension 43 --sigma .0002 --target 2.94852698 --seconds 300 --samples 2048 --boundary-bias 1 --seed 804173292 --phase-seed 804173291 --branch-map experiments/2026-11-04/round3/mirror/branch-map/branches.json --max-branch-cost 5 --frontier --slack-patterns experiments/2026-11-04/round3/slack-patterns/patterns.npz --slack-quality-temperature 20 --output experiments/2026-11-04/round3/mirror/quality-matched-learned
```

Sampler/source snapshots and hashes are in`quality-audit-source/`. No sampler code changed while these runs were live. Added separate `experiment_mirror_novelty.py` to audit every saved valid center set, quotienting a common translation only, against both175 valid training sources and prior mirror-grid JSON snapshots. It explicitly excludes both current quality groups from the prior archive. Its coordinate-only key preserves all15 piece labels; angle variations do not count as new center layouts. This is finer than basin congruence and appropriate for grid-repair novelty. Commands/audit outputs are preserved in`quality-novelty-live.json`, then final report after both jobs finish.

At about210seconds, baseline archived a below-first witness`quality-matched-baseline/frontier/witness-00366.{json,npz}`: radius first−10.2925nm, maximum actual overlap18.9208nm, three violated pairs. Root was notified immediately. It remains invalid; maximum overlap is larger than the earlier L witness despite fewer violated pairs.

## Step22 — completed fresh quality/control comparison and negative inference

Both owned300-second jobs completed normally(exit0). Neither improved the valid incumbent2.948526962953499. Baseline evaluated8,229,626 deduplicated proposals across4,787 frames; quality evaluated8,310,605 across4,254 frames. These are bounded-cache proposal deduplication counts, not a proof of global geometric uniqueness. Their first64 saved fullprecision phase frames were byte-identical.

Baseline saved37 accepted valid XY layouts:13 recovered prior/training centers and24 were new against the recorded prior mirror archive. Quality saved39:31 recovered centers and8 new. Raw-file hashes were all different; angle variations on the same centers therefore account for much of the apparent validity count. Both reached scalar nearest3.3609706nm, which by itself is not a score improvement. Quality's best saved valid relay2.9485269867883153 repeats the training center layout. Baseline's best saved relay was2.948527010752072. Saved-elite counts are the layouts accepted by the24-entry score archive policy plus its historical improvements, not all valid proposals generated.

The qualifying frontier is less favorable to quality: baseline33 distinct GEOS-checked physical poses below20nm overlap(0.109995/s), quality8(0.026656/s). Distinct here uses full once-canonical micro-grid geometry modulo common translation, with the frontier's persistent seen set; angles remain part of the key. GEOS time was19.006s/14.280s,6.34%/4.76% of wall time, after distant nomination coalescing. This finite comparison supports rejecting quality-weighted pattern replay as a throughput/exploration improvement under these settings. It does not prove a statistical ordering for every temperature or geometry, and no larger quality-only run was launched.

Baseline's best preserved below-first witness is`quality-matched-baseline/frontier/witness-00366.json`, SHA256`e5d408f95d76289a0181aebf5aabbe24fbdfc2dfd3b3a45345019c89c1eba2ca`: actual radius2.9485269135930903(first−10.292521nm), maximum normalized GEOS deficit18.920772nm, overlaps(2,11),(6,10),(12,14). It remains invalid and was sent to root. The earlier L witness has a smaller maximum deficit9.0319nm with four pairs, so these are complementary repair targets.

All421 baseline and335 quality archive sidecars exactly reproduced raw input from their own stored anchor+coefficients@moves and translation quotient, and once-canonicalized geometry matched within1e-14. Results/audits are consolidated in`quality-comparison.json` and`quality-novelty-final.json`. No owned search process remains running. Safe known-center arc-support pruning and conservative geometry filters remain unimplemented/queued; no blanket center taboo or blind Taylor cutoff was introduced.
