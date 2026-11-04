# Adversarial search, 2026-11-04

**Latest continuation:** [round3](round3/README.md) improved the verified
incumbent to **2.948526962953499**, added native batch decoding and exact
finite-domain proofs, and is pursuing an active first-place goal. The outcome
below describes this earlier campaign only. Its whole-turn conclusion also
needs a narrow qualification: round3 tests two exceptional scored seam angles
reachable via raw ±3π, without finding a useful result.

**Follow-up correction:** the [second campaign](round2/README.md) found a
different contact topology at the same continuous radius. The old radius-only
basin archive would discard it. Earlier references to repeated "same basin"
results below should therefore be read as repeated scores, not established
geometric equivalence. The follow-up also repairs two false-rejection filters.

## Outcome

**No new verified improvement.** The saved valid packing remains
**2.948526966725367**, **4.28397557e-8** above first. `solution.json` was not
changed by this research session. Lower-radius diagnostics with continuous
angles or residual overlaps are not submission files. Final API evidence:
`api-final.json` and `leaderboard-final.json`.

All bounded searches have finished. The full suite passes **78 tests**; native
angular enumeration also has independent brute-force comparison evidence.
`summary.json` indexes all 30 root run records, including smoke checks, expected
timeout, failed implementations, corrected replays, and source changes during
execution. Agent-owned runs are indexed in their individual reports below.

| Route | Result and evidence |
| --- | --- |
| Scoring freedoms | Order/translation effects are roundoff, too small to bridge the gap; `scoring-audit.md` |
| Broader contact branches | Negative-first branches worse; 16 useful positive seeds retained; `alternative-log.md` |
| Curved manifold | Two physical flex modes, 25 accurate continuation seeds; `alternative/manifold/` |
| K-best / inequality-aware decoding | 24.6 million completions, 284,228 geometry checks, no improvement; `lattice-log.md` |
| CP-SAT | Four 120-second batches, no improvement; `cpsat-log.md` |
| SCIP quadratic models | Six batches, 23,816 nodes, no improvement; `alternative/report.json` |
| Residual projections and family search | Different frames, branches, curved seeds, nonlinear corrections; no verified improvement; `projection-log.md` |
| Integer centers plus angular solving | Found continuous witnesses below the leader, then exposed integer-angle obstructions; `positions-log.md`, `angular-repair-log.md` |

Reusable results:

- `angular_rounder.py`: adjacent-angle compatibility as 2-SAT, about 0.23 ms
  with a supplied center. Independently checked against brute force.
- `angular_min_cost.py`: minimum-violation completion even when no adjacent-angle
  assignment is feasible; all 32,768 choices in about 0.52 ms native / 0.71 ms
  including cost construction at a fixed center. Independent tests cover the
  full 15-bit space, tiny costs, and the saved valid packing.
- `research_run.py`: bounded execution, command/input/source records,
  content-addressed source snapshots, environment versions, and failures.
  Successful-completion and process-group-timeout checks are recorded too.

The piece-14 endpoint angle and piece-13 contact angle must be handled jointly
with positions. Merely improving continuously optimized centers is insufficient.
The strongest untested direction is to derive and enforce the angular-grid
conflict constraints directly during center search. Do not rerun the logged
configurations assuming that more random targets resolve those conflicts.

Starting incumbent: **2.948526966725367**, API-verified in `../api_best.json`.
Live leader confirmed unchanged: **2.9485269238856113**.
The user's public entry is now the starting incumbent above. This session has
not submitted an entry. Never promote an invalid near-solution.

## What we already tried

The previous session's artifacts remain under `experiments/`. These findings are
historical evidence, not fresh measurements:

| Approach | Evidence / result | Avoid repeating without a new reason |
| --- | --- | --- |
| Global perturbation, compression, symmetry, continuation | 12,274 optimizations repeatedly reached R≈2.94868077462; radius-only deduplication did not establish identical geometry | Repeating the score-only archive; see round2 correction |
| Continuous tolerance-aware optimization | R≈2.94852654944, but six-decimal rounding invalid | Treating the continuous score as attainable |
| Random rigid frames / rounding | About 200 million frames, no better valid result | Another undirected phase sweep |
| LLL/BKZ contact-simplex search | Produced current incumbent; `../best_run.json` | Repeating the same seed/configuration |
| Contact CSP / profile / crossover | Thousands of bounded searches, no leader-beating result | Same domains and fixed-center assumptions |
| HiGHS linear cuts in reduced basis | Lower-radius candidates remain invalid | Reporting linear feasibility as geometric feasibility |
| Parallel tempering / quenching | 838,314 proposals, no valid improvement | Same energy and moves |
| Angles plus whole turns | API round→wrap→round removes extra physical precision | This is ruled out, not an open hypothesis |
| Public leader coordinates | API deliberately returns null | They are unavailable anonymously |

## New hypotheses and ownership

1. **Scoring audit** (`scoring-audit.md`): equivalent representations might expose
   numerical/order effects. Measure honestly against containment and API.
2. **Lattice enumeration** (`lattice-log.md`): Babai nearest-plane samples may
   miss useful neighboring lattice points; test a different proposal mechanism.
3. **Curved contact family** (`alternative-log.md`): linear tangent directions
   may lose feasibility at second order; inspect curvature and branch coverage.
4. **Learned discrete feasibility** (`cpsat-log.md`): CP-SAT may solve contact
   inequalities better than nearest-point sampling or the previous MILP.
5. **Residual-feedback projection** (`projection-log.md`): alternating lattice
   and contact-halfspace projections retain each failed step's residual.
6. **Integer positions and angular compatibility** (`positions-log.md`):
   conditionally solve angular grids, then promote demonstrated obstructions
   into the integer master problem.

Each log must record hypothesis, exact command/configuration/seed, duration,
result, validation status, and the inference warranted by that result.
Bounded failure is not a proof that no better packing exists.

Use `scripts/research_run.py` for automatic command/source-hash/output records.
Retain failed runs. Candidate files are experimental until independently checked
with `run.py` and the remote validator. `solution.json` is updated only after a
strictly better server-verified result.
