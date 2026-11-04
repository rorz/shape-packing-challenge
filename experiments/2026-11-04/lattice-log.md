# Contact-lattice experiments — 2026-11-04

## Step 1 — inspect earlier search and choose a different mechanism

- Reviewed `experiment_lll.py`, `experiment_lll_hillclimb.py`, and `experiment_lll_milp.py`.
- Existing LLL search discards every conditional lattice completion except the single Babai rounding result. Hillclimbing then tests a few short sums of basis directions; MILP searches bounded coefficient boxes.
- Hypothesis: retaining multiple coherent conditional rounding decisions will expose feasible inequality points close to a known valid pose, without relying on another large random target sweep.
- Chosen experiment: k-best GSO beam decoding, with bounded beam width, bounded wall time, exact local geometry screening, and final reference validation.
- Shared core and `solution.json` will remain untouched. All outputs go under `experiments/2026-11-04/lattice/`.

Commands inspected:

```text
rg --files scripts
cat scripts/experiment_lll.py
cat scripts/experiment_lll_hillclimb.py
cat scripts/experiment_lll_milp.py
.venv/bin/python: inspect fpylll Enumeration and GSO APIs
```

Outcome: fpylll exposes GSO coefficients directly; implement a bounded conditional beam rather than potentially unbounded exact sphere enumeration.

## Bounded k-best run

Hypothesis: coherent alternative GSO roundings reveal feasible inequalities discarded by single Babai decoding.

```text
.venv/bin/python scripts/experiment_oblique_kbest.py --output experiments/2026-11-04/lattice/kbest-smoke.json --seconds 3 --width 64 --actual 16
```

- {"time": "2026-11-04T01:04:32.206582+00:00", "event": "model", "seconds": 0.40184566704556346, "active": 41, "minimum_stress": 0.023181119871277384, "initial_radius": 2.9485265546636814, "initial_info": {"success": true, "iterations": 172, "violation": 5.69962201459386e-14}, "seed": 2401104, "width": 64, "branches": 3}

- {"time": "2026-11-04T01:04:32.209600+00:00", "event": "nearest", "seconds": 0.40485858399188146, "violation": 3.566099391841858e-08, "targets": 1, "trials": 64, "actuals": 1, "linear_violation": 3.100780143048419e-08}

- {"time": "2026-11-04T01:04:32.240857+00:00", "event": "nearest", "seconds": 0.43611858400981873, "violation": 3.5660993871321516e-08, "targets": 13, "trials": 832, "actuals": 195, "linear_violation": 3.1007801430493294e-08}

- {"time": "2026-11-04T01:04:34.805986+00:00", "event": "done", "seconds": 3.0012283340329304, "targets": 1202, "trials": 76928, "actuals": 14377, "duplicates": 4855, "best_violation": 3.5660993871321516e-08, "best_score": 2.948526966725367}

## Step 2 — bounded decoder smoke test and target choice

- Implemented `experiment_oblique_kbest.py` with a conditional GSO beam and actual polygon-area screening.
- Smoke command: width 64, at most 16 actual checks per target, seed 2401104, 3 seconds, current valid `solution.json`.
- Outcome: 1,202 targets, 76,928 coherent completions, 14,377 actual geometry evaluations; best maximum violation 3.5661e-8. No valid improvement. Logs and events are in `lattice/kbest-smoke.*`.
- Inference: central contact-simplex targets pull away from the incumbent's useful uneven slack allocation. Keeping more completions alone did not fix that.
- Next hypothesis: project the known incumbent's weighted slack onto the feasible simplex, retaining its clearance pattern while lifting the violated circle rows. Test boundary clearance floors 0/15/30/60 billionths in alternation. This is a local conditional search around a known valid lattice point, rather than another independent target sweep.

## Bounded k-best run

Hypothesis: coherent alternative GSO roundings reveal feasible inequalities discarded by single Babai decoding.

```text
.venv/bin/python scripts/experiment_oblique_kbest.py --output experiments/2026-11-04/lattice/kbest-incumbent-512.json --seconds 90 --width 512 --actual 128 --targets incumbent --seed 2401105
```

- {"time": "2026-11-04T01:05:55.605178+00:00", "event": "model", "seconds": 0.40603279101196676, "active": 41, "minimum_stress": 0.023181119871277384, "initial_radius": 2.9485265546636814, "initial_info": {"success": true, "iterations": 172, "violation": 5.69962201459386e-14}, "seed": 2401105, "width": 512, "branches": 3}

- {"time": "2026-11-04T01:05:55.611010+00:00", "event": "nearest", "seconds": 0.4118603329989128, "violation": 1.7099207161308527e-08, "targets": 1, "trials": 512, "actuals": 1, "linear_violation": 1.6877994273043088e-08}

- {"time": "2026-11-04T01:05:56.821171+00:00", "event": "nearest", "seconds": 1.6220388329820707, "violation": 1.7015624687033437e-08, "targets": 37, "trials": 18944, "actuals": 4040, "linear_violation": 1.6794375494923067e-08}

- {"time": "2026-11-04T01:05:57.617924+00:00", "event": "nearest", "seconds": 2.4187990829814225, "violation": 1.6958747739437285e-08, "targets": 61, "trials": 31232, "actuals": 6686, "linear_violation": 1.673753084495408e-08}

- {"time": "2026-11-04T01:06:05.175869+00:00", "event": "nearest", "seconds": 9.976816666021477, "violation": 1.688689166279289e-08, "targets": 425, "trials": 217600, "actuals": 29515, "linear_violation": 1.6665717280406264e-08}

- {"time": "2026-11-04T01:07:25.202446+00:00", "event": "done", "seconds": 90.00391179102007, "targets": 15767, "trials": 8072704, "actuals": 84978, "duplicates": 1933198, "best_violation": 1.688689166279289e-08, "best_score": 2.948526966725367}

## Step 3 — diagnose the nearest-point beam's gauge failure

- Incumbent-slack projection, width 512: ~22,000 coherent completions/s, but the best radius violation remained near the known incumbent's 1.67e-8 excess over the search radius. The full 90-second run finished: 8,072,704 completions, 84,978 actual checks, 1,933,198 duplicate skips, no valid improvement.
- Inspected the reduced basis. Rows 0–1 are exact global x/y translations. Rows 2–8 are mostly single-angle moves, much cheaper in the contact metric than a coordinated positional change.
- Hypothesis: these cheap choices flood the k-nearest set. Retain one conditional completion for the low block, spend beam width on positional decisions, and conditionally repair the low angular block by minimizing inequality loss. Deduplicate exact translations before geometry calls.
- Implemented `--freeze-low` and `--angular-repair` in the new experiment only. Shared solver remains untouched.

## Bounded k-best run

Hypothesis: coherent alternative GSO roundings reveal feasible inequalities discarded by single Babai decoding.

```text
.venv/bin/python scripts/experiment_oblique_kbest.py --output experiments/2026-11-04/lattice/kbest-quotient-512.json --seconds 90 --width 512 --actual 128 --targets incumbent --freeze-low 9 --angular-repair --seed 2401106
```

- {"time": "2026-11-04T01:08:23.426390+00:00", "event": "model", "seconds": 0.39651033299742267, "active": 41, "minimum_stress": 0.023181119871277384, "initial_radius": 2.9485265546636814, "initial_info": {"success": true, "iterations": 172, "violation": 5.69962201459386e-14}, "seed": 2401106, "width": 512, "branches": 3}

- {"time": "2026-11-04T01:08:23.440765+00:00", "event": "nearest", "seconds": 0.4108884999877773, "violation": 1.7099207161308527e-08, "targets": 1, "trials": 512, "actuals": 1, "linear_violation": 1.687799427303356e-08}


## Step 4 — independent audit of wrapped angle guidance

- API/audit agent inspected the new experiment and found that nonzero global frame angles could leave the incumbent's angle difference unwrapped in `known_gaps`, even though the position prior wrapped it correctly.
- Fixed `known_delta[:,2]` with `atan2(sin(delta),cos(delta))` before applying the contact Jacobian.
- Existing runs used angle 0, so their guidance was unaffected. The fix is required before any rotated-frame test.
- Independent probe: `experiments/2026-11-04/oblique-angle-guidance-probe.npz`.

- {"time": "2026-11-04T01:09:53.034235+00:00", "event": "done", "seconds": 90.00488212495111, "targets": 6889, "trials": 3527168, "actuals": 769, "duplicates": 881023, "best_violation": 1.7099207161308527e-08, "best_score": 2.948526966725367}

## Step 5 — prevent duplicate starvation and widen the hard beam

- The quotient decoder removed translations and conditionally optimized the seven cheap angular rows. It still returned the incumbent's radius shortfall as the best point.
- Found an implementation flaw: limiting the ranked list to `actual` points before checking the deduplication set meant repeated leaders prevented lower-ranked unseen completions from ever reaching geometry evaluation.
- Fixed ordering: scan the full ranked beam, skip seen points, and stop only after the requested number of *new* geometry checks.
- Next bounded run: width 4,096, 256 new geometry evaluations per target, frozen low block 9 with angular inequality repair, seed 2401107, 120 seconds, target radius 2.94852695. Increase weighted-target noise from 0.001 to 0.005 so repeated targets cross actual contact-space cell boundaries.

## Bounded k-best run

Hypothesis: coherent alternative GSO roundings reveal feasible inequalities discarded by single Babai decoding.

```text
.venv/bin/python scripts/experiment_oblique_kbest.py --output experiments/2026-11-04/lattice/kbest-wide-4096.json --seconds 120 --width 4096 --actual 256 --targets incumbent --target-noise .005 --freeze-low 9 --angular-repair --seed 2401107
```

- {"time": "2026-11-04T01:10:14.734606+00:00", "event": "model", "seconds": 0.39487283397465944, "active": 41, "minimum_stress": 0.023181119871277384, "initial_radius": 2.9485265546636814, "initial_info": {"success": true, "iterations": 172, "violation": 5.69962201459386e-14}, "seed": 2401107, "width": 4096, "branches": 3}

- {"time": "2026-11-04T01:10:14.836045+00:00", "event": "nearest", "seconds": 0.4963136669830419, "violation": 1.7099207161308527e-08, "targets": 1, "trials": 4096, "actuals": 1, "linear_violation": 1.6877994273033453e-08}

## Step 6 — change the decoder objective, not merely the sample count

- Checked the decoder independently on 40 randomized bases: width 1 exactly matched fpylll Babai; all retained coefficient vectors were distinct; projected versus ambient squared distances differed only by a constant (spread below 1e-9).
- Wider quotient decoding reaches new points but still favors the valid incumbent's Voronoi neighbourhood.
- New hypothesis: symmetric nearest-target distance is the wrong objective for inequality feasibility. During descending GSO decisions, the lower unrounded coordinates have an affine continuous completion. Its contact gaps can be updated with a precomputed directional vector for each rounding residual.
- Added an asymmetric decoder score: normalized Euclidean distance plus a penalty on *negative* weighted contact gaps. This biases the tree itself toward the positive cone, without calling a generic optimizer at every node.
- Test first with width 512 and penalty 100; exact translations are fixed, angular decisions remain in the tree. Final nonlinear and reference validation are unchanged.

## Bounded k-best run

Hypothesis: coherent alternative GSO roundings reveal feasible inequalities discarded by single Babai decoding.

```text
.venv/bin/python scripts/experiment_oblique_kbest.py --output experiments/2026-11-04/lattice/kbest-asymmetric-100.json --seconds 90 --width 512 --actual 128 --targets incumbent --target-noise .005 --freeze-low 2 --asymmetry 100 --seed 2401108
```

- {"time": "2026-11-04T01:11:53.076545+00:00", "event": "model", "seconds": 0.43226275005145, "active": 41, "minimum_stress": 0.023181119871277384, "initial_radius": 2.9485265546636814, "initial_info": {"success": true, "iterations": 172, "violation": 5.69962201459386e-14}, "seed": 2401108, "width": 512, "branches": 3}

- {"time": "2026-11-04T01:11:53.091243+00:00", "event": "nearest", "seconds": 0.44695662503363565, "violation": 4.927666972776701e-06, "targets": 1, "trials": 512, "actuals": 1, "linear_violation": 9.458979988218128e-09}

- {"time": "2026-11-04T01:11:53.092706+00:00", "event": "nearest", "seconds": 0.4484180830186233, "violation": 1.7099207161308527e-08, "targets": 1, "trials": 512, "actuals": 7, "linear_violation": 1.687799427303271e-08}

- {"time": "2026-11-04T01:12:14.342824+00:00", "event": "done", "seconds": 120.00379537499975, "targets": 917, "trials": 3756032, "actuals": 162704, "duplicates": 3007863, "best_violation": 1.7099207161308527e-08, "best_score": 2.948526966725367}

- {"time": "2026-11-04T01:13:22.647263+00:00", "event": "done", "seconds": 90.00350529100979, "targets": 6185, "trials": 3166720, "actuals": 2328, "duplicates": 3163588, "best_violation": 1.7099207161308527e-08, "best_score": 2.948526966725367}

## Step 7 — retain nonlinear branch information in the decoder

- The one-sided beam reached better linear points, but a predicted ~9.5e-9 violation became ~4.9e-6 under actual geometry. This is the opposed-flat wedge changing branch, plus nonlinear curvature; it is not a valid packing improvement.
- Added a bounded bank of tangent cuts from such nonlinear witnesses. The cuts retain actual contact derivatives and eliminate the free circle center through the same affine center fit as the base model.
- Cuts participate both in partial GSO beam scoring and in final completion ranking. Limit: 12 distinct cuts. These are search heuristics; reference validation remains authoritative.
- Next bounded test: 90 seconds, width 512, one-sided penalty 100, frozen translations only, seed 2401109.

## Bounded k-best run

Hypothesis: coherent alternative GSO roundings reveal feasible inequalities discarded by single Babai decoding.

```text
.venv/bin/python scripts/experiment_oblique_kbest.py --output experiments/2026-11-04/lattice/kbest-asymmetric-cuts.json --seconds 90 --width 512 --actual 128 --targets incumbent --target-noise .005 --freeze-low 2 --asymmetry 100 --cuts 12 --seed 2401109
```

- {"time": "2026-11-04T01:13:48.978513+00:00", "event": "model", "seconds": 0.40044454199960455, "active": 41, "minimum_stress": 0.023181119871277384, "initial_radius": 2.9485265546636814, "initial_info": {"success": true, "iterations": 172, "violation": 5.69962201459386e-14}, "seed": 2401109, "width": 512, "branches": 3}

- {"time": "2026-11-04T01:13:48.991532+00:00", "event": "cut", "seconds": 0.4134612919879146, "row": 77, "violation": 4.927666972776701e-06, "linear_violation": 9.458979988218128e-09, "count": 1}

- {"time": "2026-11-04T01:13:48.992485+00:00", "event": "nearest", "seconds": 0.4144117920077406, "violation": 4.927666972776701e-06, "targets": 1, "trials": 512, "actuals": 1, "linear_violation": 9.458979988218128e-09}

- {"time": "2026-11-04T01:13:48.993890+00:00", "event": "nearest", "seconds": 0.41581720899557695, "violation": 1.7099207161308527e-08, "targets": 1, "trials": 512, "actuals": 7, "linear_violation": 1.687799427303271e-08}

- {"time": "2026-11-04T01:13:49.051002+00:00", "event": "cut", "seconds": 0.4729341249912977, "row": 77, "violation": 1.0304002441959689e-07, "linear_violation": 5.2819016099170826e-08, "count": 2}

- {"time": "2026-11-04T01:13:49.117361+00:00", "event": "cut", "seconds": 0.5392928750370629, "row": 72, "violation": 1.1371271782393131e-07, "linear_violation": 2.5793657762851212e-08, "count": 3}

- {"time": "2026-11-04T01:13:49.118412+00:00", "event": "cut", "seconds": 0.540338792023249, "row": 90, "violation": 1.201912926998674e-07, "linear_violation": 6.879911777468231e-08, "count": 4}

- {"time": "2026-11-04T01:13:49.320455+00:00", "event": "cut", "seconds": 0.7423899589921348, "row": 90, "violation": 3.929672832351565e-07, "linear_violation": 4.6083127025959456e-08, "count": 5}

- {"time": "2026-11-04T01:13:49.709290+00:00", "event": "cut", "seconds": 1.1312256670207717, "row": 8, "violation": 2.098306362506304e-07, "linear_violation": 5.48828277633086e-08, "count": 6}

- {"time": "2026-11-04T01:13:49.800692+00:00", "event": "cut", "seconds": 1.2226287089870311, "row": 8, "violation": 1.0239781972571912e-07, "linear_violation": 5.311805544436303e-08, "count": 7}

- {"time": "2026-11-04T01:13:57.064121+00:00", "event": "cut", "seconds": 8.48610112501774, "row": 148, "violation": 7.798292287475306e-07, "linear_violation": 5.213371649316526e-08, "count": 8}

- {"time": "2026-11-04T01:13:57.066442+00:00", "event": "cut", "seconds": 8.48841629200615, "row": 84, "violation": 2.071601345295163e-07, "linear_violation": 6.644431660987742e-08, "count": 9}

- {"time": "2026-11-04T01:13:57.070627+00:00", "event": "cut", "seconds": 8.492602292040829, "row": 147, "violation": 1.6012832526612897e-07, "linear_violation": 6.591306809267985e-08, "count": 10}

- {"time": "2026-11-04T01:14:04.377148+00:00", "event": "cut", "seconds": 15.79917083401233, "row": 72, "violation": 1.7588411284948397e-07, "linear_violation": 7.084177850168312e-08, "count": 11}

- {"time": "2026-11-04T01:14:04.378718+00:00", "event": "cut", "seconds": 15.800735542026814, "row": 128, "violation": 1.0252524740295144e-07, "linear_violation": 7.084177850168312e-08, "count": 12}

- {"time": "2026-11-04T01:15:18.583934+00:00", "event": "done", "seconds": 90.0063931249897, "targets": 5788, "trials": 2963456, "actuals": 2321, "duplicates": 2958792, "best_violation": 1.7099207161308527e-08, "best_score": 2.948526966725367}

## Step 8 — one snapped orthogonal frame, targeting a real score improvement

- Root independently found that wrapping opposed pairs can change their raw angle difference by 2π. A physical integer-microradian difference need not remain an integer raw difference after the wrap.
- Added `--snap-frame`: rotate the incumbent first, canonicalize its angles, then optimize the continuous seed while holding those *raw* opposed-angle differences fixed. This tests one exact 90-degree positional lattice symmetry while changing the angular grid phase.
- This does not rely on angle periodicity to encode extra precision; canonicalization follows the audited server rules.
- Final bounded batch: width 1,024, one-sided penalty 30, up to 16 nonlinear cuts, frame π/2, and target 2.9485269666 (slightly better than the current valid incumbent), while retaining the first-place goal separately. Prioritize any reference-valid score improvement.

- Preflight correction: the first snapped-frame invocation exited with argparse status 2 because the option declaration was missed on a line containing two parser calls. No search ran. Added the declaration and reran the exact command; retained the failed invocation in `kbest-snapped-quadrant.log` below via the run manifest.

## Bounded k-best run

Hypothesis: coherent alternative GSO roundings reveal feasible inequalities discarded by single Babai decoding.

```text
.venv/bin/python scripts/experiment_oblique_kbest.py --output experiments/2026-11-04/lattice/kbest-snapped-quadrant.json --seconds 95 --width 1024 --actual 192 --targets incumbent --target-noise .005 --freeze-low 2 --asymmetry 30 --cuts 16 --snap-frame --angle 1.5707963267948966 --target 2.9485269666 --seed 2401110
```

- {"time": "2026-11-04T01:16:12.010081+00:00", "event": "model", "seconds": 0.5173758750315756, "active": 41, "minimum_stress": 0.023181121573869395, "initial_radius": 2.9485265546637724, "initial_info": {"success": false, "iterations": 250, "violation": 1.0223218671559403e-12}, "seed": 2401110, "width": 1024, "branches": 3}

- {"time": "2026-11-04T01:16:12.035661+00:00", "event": "cut", "seconds": 0.5429549579857849, "row": 90, "violation": 3.4737555251655878e-06, "linear_violation": 3.338711113433044e-08, "count": 1}

- {"time": "2026-11-04T01:16:12.036733+00:00", "event": "nearest", "seconds": 0.5440211250097491, "violation": 3.4737555251655878e-06, "targets": 1, "trials": 1024, "actuals": 1, "linear_violation": 3.338711113433044e-08}

- {"time": "2026-11-04T01:16:12.037512+00:00", "event": "cut", "seconds": 0.5448000410106033, "row": 90, "violation": 3.819253552138484e-06, "linear_violation": 3.338711113433044e-08, "count": 2}

- {"time": "2026-11-04T01:16:12.038189+00:00", "event": "cut", "seconds": 0.5454770409851335, "row": 84, "violation": 1.1177724766818546e-06, "linear_violation": 3.342517984584976e-08, "count": 3}

- {"time": "2026-11-04T01:16:12.038902+00:00", "event": "nearest", "seconds": 0.5461895830230787, "violation": 1.1177724766818546e-06, "targets": 1, "trials": 1024, "actuals": 11, "linear_violation": 3.342517984584976e-08}

- {"time": "2026-11-04T01:16:12.039264+00:00", "event": "cut", "seconds": 0.5465518750133924, "row": 84, "violation": 8.16468301517075e-07, "linear_violation": 3.342517984584976e-08, "count": 4}

- {"time": "2026-11-04T01:16:12.039894+00:00", "event": "nearest", "seconds": 0.5471816660137847, "violation": 8.16468301517075e-07, "targets": 1, "trials": 1024, "actuals": 13, "linear_violation": 3.342517984584976e-08}

- {"time": "2026-11-04T01:16:12.040262+00:00", "event": "cut", "seconds": 0.5475488330121152, "row": 84, "violation": 9.453857703470389e-07, "linear_violation": 4.1167304295243185e-08, "count": 5}

- {"time": "2026-11-04T01:16:12.041701+00:00", "event": "cut", "seconds": 0.548988291004207, "row": 143, "violation": 2.2868180149515638e-07, "linear_violation": 4.4746534187306944e-08, "count": 6}

- {"time": "2026-11-04T01:16:12.042307+00:00", "event": "nearest", "seconds": 0.5495940000400878, "violation": 2.2868180149515638e-07, "targets": 1, "trials": 1024, "actuals": 27, "linear_violation": 4.4746534187306944e-08}

- {"time": "2026-11-04T01:16:12.043982+00:00", "event": "cut", "seconds": 0.5512697910307907, "row": 90, "violation": 4.174833399741148e-06, "linear_violation": 4.282552441629903e-08, "count": 7}

- {"time": "2026-11-04T01:16:12.044346+00:00", "event": "cut", "seconds": 0.5516335410065949, "row": 131, "violation": 2.560309675381234e-07, "linear_violation": 4.590582765059593e-08, "count": 8}

- {"time": "2026-11-04T01:16:12.045003+00:00", "event": "cut", "seconds": 0.5522904160316102, "row": 84, "violation": 2.8171998191788207e-06, "linear_violation": 4.555580080739422e-08, "count": 9}

- {"time": "2026-11-04T01:16:12.045242+00:00", "event": "cut", "seconds": 0.5525287500349805, "row": 84, "violation": 2.511276766510904e-06, "linear_violation": 4.555580080739422e-08, "count": 10}

- {"time": "2026-11-04T01:16:12.045847+00:00", "event": "cut", "seconds": 0.5531429160037078, "row": 84, "violation": 3.1240849162901945e-06, "linear_violation": 4.555580080739422e-08, "count": 11}

- {"time": "2026-11-04T01:16:12.047718+00:00", "event": "cut", "seconds": 0.5550060410168953, "row": 8, "violation": 3.2623397268339877e-07, "linear_violation": 4.58589898098286e-08, "count": 12}

- {"time": "2026-11-04T01:16:12.047976+00:00", "event": "cut", "seconds": 0.5552637910004705, "row": 84, "violation": 4.182775459909949e-06, "linear_violation": 4.802329318344478e-08, "count": 13}

- {"time": "2026-11-04T01:16:12.048539+00:00", "event": "cut", "seconds": 0.5558257080265321, "row": 84, "violation": 4.4935547063719435e-06, "linear_violation": 4.802329318344478e-08, "count": 14}

- {"time": "2026-11-04T01:16:12.050962+00:00", "event": "cut", "seconds": 0.5582515000132844, "row": 90, "violation": 1.7113026583200151e-06, "linear_violation": 4.9466784862426176e-08, "count": 15}

- {"time": "2026-11-04T01:16:12.052081+00:00", "event": "cut", "seconds": 0.5593687500222586, "row": 90, "violation": 2.1106996971858924e-06, "linear_violation": 4.856364298940226e-08, "count": 16}

- {"time": "2026-11-04T01:16:12.053609+00:00", "event": "nearest", "seconds": 0.5608976660296321, "violation": 2.1380723991060226e-07, "targets": 1, "trials": 1024, "actuals": 89, "linear_violation": 4.855124202050265e-08}

- {"time": "2026-11-04T01:16:12.099352+00:00", "event": "nearest", "seconds": 0.6066434580134228, "violation": 6.441266023333774e-08, "targets": 2, "trials": 2048, "actuals": 246, "linear_violation": 5.5558040572202294e-08}

- {"time": "2026-11-04T01:16:12.517448+00:00", "event": "nearest", "seconds": 1.0247442910331301, "violation": 5.539036071284911e-08, "targets": 11, "trials": 11264, "actuals": 1927, "linear_violation": 5.597165541767733e-08}

- {"time": "2026-11-04T01:17:46.505832+00:00", "event": "done", "seconds": 95.0136832910357, "targets": 2951, "trials": 3021824, "actuals": 16751, "duplicates": 2999290, "best_violation": 5.539036071284911e-08, "best_score": 2.948526966725367}

## Step 9 — eliminate the entire angular integer block with 2-SAT

- Coordinated with root: search 28 integer position coordinates after fixing translation, treat 15 orientations and two enclosing-center coordinates as continuous during proposal projection.
- For a fixed position proposal, each orientation has two adjacent six-decimal choices. Every forbidden pair assignment is a 2-clause, so angular feasibility is exactly 2-SAT at a fixed enclosing center; no 32,768-case exhaustive search is needed.
- Implemented `scripts/angular_rounder.py`: one 30-shape batch computes all 2x2 pair tables, 90 analytic boundary rows give unary containment, and Tarjan SCC returns a globally compatible assignment or None.
- API: `round_angles(continuous_p, target, center=None, return_info=False)`. Optional `area_guard` defaults to 1e-12; final reference validation belongs to the caller. No shared geometry or solution was changed.
- Verification command: `.venv/bin/python scripts/experiment_oblique_angle_rounder_check.py` (seed 41104).
- Outcome: 350 randomized binary systems matched brute force; the valid incumbent was recovered at its exact score; impossible radius 2 was rejected. Timing and exact score are saved in `lattice/angle-rounder-check.json`.

- Frozen starting incumbent for replay: `lattice/incumbent-start.json`, SHA-256 `088950a795745e2757fd462e6a9085220777ca00422bca66fb3d9c6bba0e475e`. All current beam runs report its verified score 2.948526966725367; replace `--input solution.json` with this snapshot when replaying after future promotions.

## Final bounded-run outcomes

| Run | Seconds | Coherent completions | Actual checks | Best violation | Outcome |
|---|---:|---:|---:|---:|---|
| kbest-asymmetric-100 | 90.00 | 3166720 | 2328 | 1.70992e-08 | done; no valid score improvement |
| kbest-asymmetric-cuts | 90.01 | 2963456 | 2321 | 1.70992e-08 | done; no valid score improvement |
| kbest-incumbent-512 | 90.00 | 8072704 | 84978 | 1.68869e-08 | done; no valid score improvement |
| kbest-quotient-512 | 90.00 | 3527168 | 769 | 1.70992e-08 | done; no valid score improvement |
| kbest-smoke | 3.00 | 76928 | 14377 | 3.5661e-08 | done; no valid score improvement |
| kbest-snapped-quadrant | 95.01 | 3021824 | 16751 | 5.53904e-08 | done; no valid score improvement |
| kbest-wide-4096 | 120.00 | 3756032 | 162704 | 1.70992e-08 | done; no valid score improvement |

No shared core or solution edits were made in this experiment. The current valid score remains 2.948526966725367. New reusable result: `scripts/angular_rounder.py` eliminates the angular binary search with 2-SAT in about 0.23 ms at a supplied circle center. Its independent verification is saved in `lattice/angle-rounder-check.json`. Every owned bounded runner has finished; no unattended search is intended.
