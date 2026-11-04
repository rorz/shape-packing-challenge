# Simultaneous two-disk fission gate

The prize-deadline strategy review authorized one new structural experiment:
fuse both opposed-flat pairs, optimize eleven semicircles plus two full disks,
then choose both diameter partitions and release all four halves together.
Earlier experiments split one disk at a time. This finite schedule changes that
transition mechanism; it is not another single-contact release campaign.

The runner detects the two disjoint active flat pairs from each input. It uses
the original continuous branch and the frozen mirror seed; their labels differ.
It interleaves both macro starts across at most eight cut choices per disk and
three tangential staggers per disk (0, +0.3, −0.3), at most 1,152 trials. The
structural gate has a 180-second deadline including recompression and continuous
area refinement. No candidate is accepted on optimizer status alone.

Strict feasible states within 0.0001 of the input strict radii receive area
refinement unless geometrically congruent to a known strict state. Competitive
area-refined seeds must have minimum normalized constraint gap ≥−2e-9 and radius
at least 8e-8 below the recorded leader. Congruence checks include positions,
normals, permutations, rotation, translation and reflection, with a 1e-5
position/angle tolerance. Contact graphs are
recorded as a separate diagnostic; matching radius or graph alone never removes
a geometrically distinct seed. Novelty is relative to the two inputs and this
run's accepted states, not an exhaustive comparison with all historical seeds.

Only new competitive seeds receive a generic 45-coordinate native probe, at
most four seeds and ten seconds each. The run is offline. Rounded candidates
need reference validation; the root agent separately handles any API check and
promotion. The runner does not read or write the global incumbent.

Five focused tests passed in 0.62 seconds: dynamic pair detection under
permutation, joint release support equivalence, congruence including normals,
complete interleaved scheduling, and the two-disk Jacobian/partition limits.

```bash
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/experiment_joint_fission.py \
  --input experiments/branch-3.141616-3.141592.json \
  --input tests/fixtures/recovery/mirror-continuous.json \
  --output experiments/pantheon-pilot/deadline-fission-v1 \
  --seconds 180 --seed 3111501 --probe-seconds 10 --max-probes 4
```

Full source snapshots, input hashes, macro cut sets, individual trials and any
seeds/probes are preserved in that ignored output directory. Do not repeat this
exact schedule if it produces no new competitive geometry.

## Result

The full **1,152-trial schedule completed in 70.01 seconds**, before the
180-second deadline. Of these, 1,151 were strict-feasible; 382 entered the broad
continuous gate, 328 matched already seen strict geometry, and 54 received area
refinement. Eight refined outputs were competitive, but all eight matched known
continuous geometry within the stated tolerance. **No new competitive seed or
valid score improvement was found.** No native probe was launched, because its
new-geometry prerequisite never passed.

Both macro starts produced eight diameter choices for each disk, without
omitting a detected contact partition. The detected input pairs were (1,5) and
(6,10) for the original branch, and (1,14) and (4,8) for the mirror seed. Their
optimized full-disk macro radii were 3.035813454803334 and 3.0272720529517696.

This closes the tested two-macro joint schedule. It does not rule out arbitrary
two-disk macro packings, other stagger magnitudes, larger continuous
deformations, or rounding improvements within a known continuous geometry.
No extension or repeat was launched after the negative gate.

The best refined radius was 2.948526549416175, matching the existing continuous
minimum; it is not a valid six-decimal submission. The unchanged runner source
matches its frozen run snapshot. SHA-256 evidence:

- Runner: `eac8b29e8fab622606fc23ba2b4c8f240b1471518d444520bb4722ff1f6d7cab`.
- Run report: `9e6856550e6f0f019b34f71ba9fa91ea8557157faeb3b72b9234f69d823d1fd5`.
- Complete event log: `0eaf8b11180eeb2e27addc9ef0fbb2a640209c86af3670d0772f5a333140b346`.
