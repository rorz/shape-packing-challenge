# Independent alternative experiments — 2026-11-04

Scope: diagnose the local contact geometry and test a strategy distinct from failed lattice tempering. All outputs stay under `alternative/`; `solution.json` will not be changed. Each solver run is bounded to at most five minutes. No dev services are used.

## Step 1 — inspect existing approaches

Hypothesis: the largest reduced-lattice steps approximate global rotation only to first order; their quadratic radial drift creates the apparent barrier. A curvature-aware alternative should therefore either parameterize rotation exactly or remove it before integer search.

Commands:

```sh
rg --files -g 'AGENTS.md' -g 'pyproject.toml' -g 'search.py' -g 'README*'
rg --files scripts | sort
sed -n '1,260p' scripts/lattice_polish.py
sed -n '1,240p' scripts/experiment_newton_family.py
sed -n '1,220p' scripts/experiment_lll_milp.py
```

Results: existing work includes Cartesian and reduced-basis MILP, Newton projection along soft modes, random rigid-frame rounding, and LLL/BKZ target sampling. Plain integer MILP is therefore not a fresh strategy. Next, measure singular values and separate rigid rotation from actual deformation; test a fixed-angle integer-position model only if the diagnostics show a useful reduction.

## Step 2 — audit flat-angle branches and local rank

Commands:

```sh
sed -n '1,260p' scripts/experiment_flat_branches.py
sed -n '1,260p' scripts/experiment_branch_campaign.py
sed -n '1,250p' scripts/experiment_newton_negative.py
sed -n '1,220p' scripts/experiment_sampling_sweep.py
```

A Python diagnostic loaded `solution.json`, `continuous.json`, `best.json`, and `soft_modes.npz`; it evaluated native MEC radii, both opposed-flat angle differences relative to pi, and active-contact singular values (`g < 1e-7`). Results:

- Current grid packing radius: 2.9485269667253693; flat offsets [+23.3464102, -0.6535898] microradians.
- `continuous.json`: radius 2.9485266454389163; offsets [+23.1269647, +0.7857380] microradians.
- `soft_modes.npz`: radius 2.9485265495504214; offsets [+23.1500942, +0.7865382] microradians; 41 active contacts. The smallest singular value is 7.094e-7, versus 0.1773 for the next one: one very shallow direction is present.
- Previous fixed-branch sweep covered first offsets only +21.3464 through +24.3464 microradians; second offsets -1.6536 through +1.3464 microradians. Negative-first continuations on the same unreflected packing were absent.

Hypothesis: independently reversing the first flat-angle sign may expose an overlooked continuation branch. A whole-packing reflection flips both signs and preserves the grid, so reflection alone adds no new feasible grid packing.

Implementation: `scripts/experiment_alt_branches.py` fixes the two angle differences and minimizes continuous radius with all native polygon-area constraints. It restarts every case from the same continuous seed to avoid a warm-start bias. Full precision outputs are search seeds, not submission files.

Command (deterministic, no random seed):

```sh
.venv/bin/python scripts/experiment_alt_branches.py --seconds 180
```

## Step 3 — measure the straight-tangent curvature barrier

Hypothesis: reduced basis steps of millimeters are mostly infinitesimal rotation. Their omitted second-order positional correction is large compared with the tens of nanometers of remaining constraint slack.

Implementation: `scripts/experiment_alt_geometry.py` records the active singular spectrum, the stress budget at the leader radius, each small singular vector's rigid-motion fraction, and eight largest moves in two saved reduced bases. Each move is compared against exact rigid rotation and against the first-order area prediction. The MEC center is recomputed for every nonlinear evaluation.

Command (deterministic):

```sh
.venv/bin/python scripts/experiment_alt_geometry.py
```

Step 2 results: all 55 cases finished in roughly 11 seconds. Negative first offsets are worse continuously, before any rounding: -23 microradians gives radius 2.948528568635, -15 gives 2.948527879922. Useful overlooked positive offsets remain: 0 gives 2.948526810621, 15 gives 2.948526563640, 40 gives 2.948526584937, versus the optimum near +23 at 2.948526550329. All cited cases have maximum normalized constraint residual below 1.5e-12. Optimizer iteration-limit messages were retained explicitly; feasibility and actual native MEC, rather than success flags, determine usefulness.

Step 3 results: the original rigid-rotation hypothesis was incomplete. Among the sixteen largest saved basis moves, only a few are mostly rigid. Rotation curvature alone reaches 390 nm, while total normalized area linearization errors range from 290 nm to 5190 nm. The available leader headroom above the continuous optimum is 374.34 nm. Thus second-order deformation, not merely rigid rotation, can swamp the remaining slack. The positive contact stress has minimum weight 0.0234. Full diagnostics are in `alternative/geometry.json`.

A follow-up Python command inspected flat-angle increments for those sixteen moves; all have zero increments in both opposed-flat differences. It also grouped `branches/summary.json` by the first offset and printed each minimum-radius second offset. This confirms that previous long steps already preserve the flat differences; changing branches needs new seeds.

## Step 4 — create overlooked, grid-compatible branch seeds

Command (deterministic):

```sh
.venv/bin/python scripts/experiment_alt_branches.py --seconds 60 --first '7.346410206884,12.346410206884,17.346410206884,27.346410206884,32.346410206884,37.346410206884,47.346410206884,57.346410206884' --second '-0.653589793116,0.346410206884' --output experiments/2026-11-04/alternative/grid-branches
```

The corresponding raw first differences are 3.141600, 3.141605, 3.141610, 3.141620, 3.141625, 3.141630, 3.141640, and 3.141650 radians. Second differences are 3.141592 and 3.141593 radians. These are exactly compatible with differences of six-decimal input angles.

## Step 5 — install a distinct nonlinear integer solver

Hypothesis: a mixed-integer quadratic contact model can retain the curvature that linear cuts discard. This is distinct from the root agent's CP-SAT linear-cut search.

Documentation checked: [PySCIPOpt model API](https://pyscipopt.readthedocs.io/en/stable/api/model.html) and [nonlinear expressions](https://pyscipopt.readthedocs.io/en/stable/tutorials/expressions.html). Both describe integer variables with nonlinear constraints through `Model.addCons`.

Commands:

```sh
.venv/bin/python -c "import importlib.util; print(bool(importlib.util.find_spec('pyscipopt')))"
uv pip install --python .venv/bin/python pyscipopt
UV_CACHE_DIR=/private/tmp/shape-alt-uv-cache uv pip install --python .venv/bin/python pyscipopt
```

Results: package initially absent. The first installation could not write the default uv cache in the restricted environment; retrying with a temporary writable cache succeeded. Installed PySCIPOpt 6.3.0 in the existing venv. No dependency manifest was edited.

The first Step 4 invocation stopped at argument parsing: the comma-separated negative `--second` value required `--second=...`. No solver case ran. Corrected command:

```sh
.venv/bin/python scripts/experiment_alt_branches.py --seconds 60 --first '7.346410206884,12.346410206884,17.346410206884,27.346410206884,32.346410206884,37.346410206884,47.346410206884,57.346410206884' --second='-0.653589793116,0.346410206884' --output experiments/2026-11-04/alternative/grid-branches
```

## Step 6 — branch-locked mixed-integer quadratic search

Implementation: `scripts/experiment_alt_quadratic.py` removes two integer translations and the two flat-angle differences, leaving 41 integer placement coordinates plus two continuous MEC-center coordinates. The native normalized contact functions are expanded to second order using central differences of analytic Jacobians (0.1 microunit steps). SCIP minimizes the maximum violation at the leader radius. A warm incumbent is included. Each returned candidate is checked with the original polygon validator; only a valid improvement is sent to the arena API.

The script also checks the quadratic approximation on 30 random local directions before solving. These checks diagnose model error and do not assert global exactness. Seed: 110401.

Command:

```sh
.venv/bin/python scripts/experiment_alt_quadratic.py --input solution.json --seconds 60 --steps 50 --output experiments/2026-11-04/alternative/quadratic-current
```

Step 6 diagnostic: the raw-area quadratic model has 664 nonzero quadratic terms but randomized 20-microunit probes show maximum normalized error 411,558 nm (median 55,689 nm). This invalidates its use as a reliable local model over that box. The culprit is the overlap area's growth law and changing wedge topology: flat wedges grow approximately quadratically in penetration, while curved contacts grow approximately as penetration^(3/2). The first solve remains bounded at 60 seconds; no result will be promoted without original validation.

## Step 7 — use clearance-like coordinates before quadratic modeling

Hypothesis: applying the monotone powers area^(1/2) at the two flat wedges and area^(2/3) elsewhere converts overlap area into a quantity closer to signed penetration. The zero set stays exactly at the same allowed overlap area. First derivatives at the constraint boundary are scaled to match the original normalized contact Jacobian. This should suppress the stiff quadratic growth without relaxing feasibility.

The quadratic script now has `--clearance-power`. Its random-probe diagnostic will determine whether this is useful before longer solves.

Command, seed 110401:

```sh
.venv/bin/python scripts/experiment_alt_quadratic.py --input solution.json --seconds 20 --steps 50 --clearance-power --output experiments/2026-11-04/alternative/quadratic-clearance
```

Step 7 results: the power transform reduced maximum Hessian magnitude from 610.93 to 0.56 nm per square microunit. The 20-second solve retained the existing valid incumbent (radius 2.948526966725367); it did not improve it. A broader branch test was started:

```sh
.venv/bin/python scripts/experiment_alt_quadratic.py --input experiments/2026-11-04/alternative/grid-branches/f+027p35_s+00p35.json --difference 3.141620 3.141593 --seconds 60 --steps 50 --clearance-power --seed 110402 --output experiments/2026-11-04/alternative/quadratic-branch620
```

Its 30 random diagnostics showed zero false-feasible or false-infeasible classifications. Large absolute errors remained because positive clearance saturates when actual overlap becomes zero; that error does not automatically imply a wrong feasibility boundary. The solver outcome is recorded in its run log and summary.

## Step 8 — analytically invert the flat-wedge area constraint

The root agent independently suggested the same clearance coordinate and supplied a more exact route: invert the piecewise triangular/trapezoidal integral of the two opposed flat edges. `experiment_contact_csp.flat_areas` provides the segment geometry. A Python comparison on the current solution found flat-model versus native polygon areas differing by only 8.93e-15 for pair(1,5) and 2.05e-16 for pair(6,10).

Implementation: `--flat-inversion` now replaces those two area constraints by the maximum allowed intercept minus the actual intercept. For overlap length L and slope magnitude q, the maximum tip height is sqrt(2*q*tol) in the triangular case, and tol/L+q*L/2 in the trapezoidal case. Both relative angle differences remain locked, so only common angle and relative position vary. Analytic first derivatives are included; Hessians are obtained by central differencing those derivatives. Curved contacts retain the area^(2/3) transform.

Command, seed 110403:

```sh
.venv/bin/python scripts/experiment_alt_quadratic.py --input solution.json --seconds 120 --steps 50 --clearance-power --flat-inversion --seed 110403 --output experiments/2026-11-04/alternative/quadratic-flat-inverted
```

Step 8 diagnostic: analytic flat inversion reduced maximum 20-microunit probe error from 27,960 nm to 85.34 nm, median 25.39 nm; no false-feasible or false-infeasible classification occurred in the 30 probes. This is still a local approximation and does not establish exactness. The broader branch620 power-only solve ended invalid with maximum clearance residual about222 nm; no API call was made.

## Step 9 — keep curvature and change the integer branching coordinates

Hypothesis: Cartesian integer branching wastes the small contact slack; a reduced lattice basis may improve the same quadratic search. This is not the previous tempering algorithm: SCIP retains the second-order constraints and branches on integer reduced coefficients, while physical displacement variables are linked by exact linear equalities.

Implementation: `--reduced` embeds the 41 independent integer coordinates in the contact Jacobian after projecting out the continuous enclosing center, weighted by the near-null contact stress. LLL provides a unimodular integer basis. Four linear symmetries/flat differences have already been eliminated. Physical displacement is limited to the requested box and reduced coefficients to +/-4.

Command, seed 110404 (120-second bound):

```sh
.venv/bin/python scripts/experiment_alt_quadratic.py --input solution.json --seconds 120 --steps 300 --clearance-power --flat-inversion --reduced --coefficients 4 --basis-regularization .0001 --seed 110404 --output experiments/2026-11-04/alternative/quadratic-reduced
```

## Step 10 — quantify the nonrigid contact manifold

An exact current-branch continuation was generated with:

```sh
.venv/bin/python scripts/experiment_alt_branches.py --seconds 10 --first '23.346410206884' --second='-0.653589793116' --output experiments/2026-11-04/alternative/current-branch
```

Result: radius 2.948526554663789, maximum normalized residual 4.87e-13. Relative to this fixed branch, the current grid packing costs 412.06158 nm of radius, while the leader permits only 369.22182 nm. The shortfall is 42.83976 nm. Relative to the unconstrained continuous optimum, the grid penalty is 417.17495 nm and allowed headroom374.33519 nm.

Hypothesis: after fixing the two opposed-flat differences and the enclosing-circle center, three first-order contact modes remain: global rotation and two nonrigid flexes. The flexes are blocked at second order, rather than at first order. Exact contact continuation can quantify their radius cost and distinguish this mechanism from pure rounding noise.

Implementation: `experiment_alt_manifold.py` finds the null space of the 41 active contact rows plus the two fixed-flat equations, projects out global rotation, and solves all contact equations with Newton continuation on a 5x5 grid in the two remaining mode coordinates. Radius is an unknown during continuation. This is diagnostic, not a submission search.

Command (deterministic, 30-second bound):

```sh
.venv/bin/python scripts/experiment_alt_manifold.py --seconds 30
```

Step 10 results: the exact contact system has rank43, leaving two nonrigid flexes after removing rotation. The radius Hessian in orthonormal flex coordinates is [[0.11211086,0.00451125],[0.00451125,0.05906191]], with eigenvalues0.05868101 and0.11249176. Leader headroom allows flex amplitudes about3.547mm and2.562mm along the principal axes. All25 continued seeds have residuals at most1.512e-12 and radii from2.948526554663789 to2.948526916029802. Quadratic radius prediction error is at most1.223nm over the tested +/-2mm box. The complete diagnostic took0.021s.

Step 8/9 solve results: the120-second Cartesian inverted-flat solve visited7436 nodes and retained the current valid incumbent. The120-second reduced-basis solve visited4207 nodes and also retained it. Neither produced a valid improvement. Their dual bounds remained too weak to establish optimality or infeasibility; they are heuristic search results.

## Step 11 — final SCIP strengthening test

Hypothesis: summing the quadratic contact constraints with positive contact-stress weights exposes global second-order curvature to SCIP more directly than the separate local constraints. This weighted constraint is mathematically redundant because all weights are positive. Feasibility emphasis may also spend more work improving the incumbent.

A Python signature check confirmed that installed SCIP exposes `SCIP_PARAMEMPHASIS.FEASIBILITY` and `Model.setEmphasis(paraemphasis, quiet=True)`.

Command, seed110405,120-second bound:

```sh
.venv/bin/python scripts/experiment_alt_quadratic.py --input solution.json --seconds 120 --steps 300 --clearance-power --flat-inversion --reduced --coefficients 4 --basis-regularization .0001 --stress-aggregate --feasibility-emphasis --seed 110405 --output experiments/2026-11-04/alternative/quadratic-stress
```

## Step 12 — independently check the new flat-clearance derivatives

A finite-difference audit used80 perturbed pair placements (uniform position/common-angle perturbations within+/-50microunits) and checked400 directional derivatives: both x/y coordinates of each pair and their tied common angle. Central-difference step1e-7; random seed110406. Audit data is saved as `alternative/flat-derivative-audit.json`.

The same command recorded installed numpy/scipy/shapely/fpylll/pyscipopt versions and SHA256 hashes of all `experiment_alt_*.py` scripts in `alternative/environment.json`.

Step 12 execution note: the first inline audit import failed because `arena` lives under `scripts/` and that directory was not on Python's module path. Retried after `sys.path.insert(0,'scripts')`; the solver runs themselves are direct script invocations and were unaffected. The derivative audit and environment manifest were then written successfully.

Step 12 results: maximum absolute derivative error8.0445e-10; median2.3518e-10 over400 comparisons. Versions: numpy2.5.4, scipy1.18.1, shapely2.2.0, fpylll0.6.4, pyscipopt6.3.0. The independent audit agent also reviewed the branch centering, tied theta variables, center signs, and power-transform Jacobian; no blocking issue was found.

An interim `alternative/report.json` combines all completed solver summaries with contact/manifold diagnostics. It will be refreshed after the final stress-aggregate run. Every candidate retained by SCIP so far is either the existing valid incumbent or an invalid near-solution; no new arena API request has been necessary.

## Final result

The final stress-aggregate run also ended at its120-second limit without improving the valid incumbent. All six SCIP variants together visited23,816 nodes. No valid improvement was found and no candidate was promoted. All owned runner sessions exited normally; `solution.json` was never written by this agent. `alternative/report.json` includes the six completed solve summaries, the branch and manifold diagnostics, and the derivative audit.

The useful outcome is geometric and reproducible: negative-first branches are continuously inferior; wider positive first offsets were underexplored; the contact network has two nonrigid first-order flexes whose second-order radius cost is measurable; and analytic flat-area inversion removes most of the curvature-model error. These findings do not prove a rounding lower bound or global optimality.

For direct reproducibility, the inline derivative audit was saved as `scripts/experiment_alt_audit.py` and rerun with:

```sh
.venv/bin/python scripts/experiment_alt_audit.py
```

It reads the preserved original incumbent from `alternative/quadratic-current/candidate-0.json`, avoiding dependence on a later replacement of `solution.json`. The environment manifest now hashes that input and all five alternative scripts.

Final stress-run candidate detail: proxy maximum deficit improved from42.84 to35.68nm; actual native radius2.9485269554063613. Original validation still found overlaps in pairs(0,11),(2,9),(7,10). Candidate saved at `alternative/quadratic-stress/candidate-0.json` for possible independent repair; it is not a valid solution and was not sent to the API.
