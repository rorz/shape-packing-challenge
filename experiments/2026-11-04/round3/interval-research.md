# Exact finite-CSP interval compression

## Initial hypothesis and immutable inputs

Large flat-contact compatibility relations survive identical-row/column quotienting. Reorder their column classes, represent every true run as an inclusive interval, and project the relation through a `(row_class, low, high)` table plus `rank_B >= low` and `rank_B <= high`. No geometry is approximated or recomputed. Every source tuple is retained exactly.

Arena identified immutable `conflicts/network-angle3.{domains,tables}.npz` as the latest completed width-three mirror domain. Root's corresponding original domain is `conflicts-original/wide.{domains,tables}.npz`. Domains contain complete canonical six-decimal poses; tables contain categorical allowed pairs. Domain digests are checked when present.

## Geometry and mask orders

`experiment_interval_tables.py` compares class identity, degree, angular grouping, geometric normal/flat-clearance orders, and a bounded spectral heuristic. The first large-relation pass found:

- Mirror4/8: 424,890 allowed class tuples become12,274 intervals under a flat-clearance order; identity gives15,405.
- Mirror5/8: 250,278 allowed tuples become707 intervals, exactly one per row.
- Original6/10: 646,759 allowed tuples become7,124 intervals under identity order.

The analyzer then checked all saved relations, including the actual categorical row and column element maps, under `intervals-verified/`. Exact commands are in the tool transcript; the principal invocation is:

```text
.venv/bin/python scripts/experiment_interval_tables.py --prefix experiments/2026-11-04/round3/conflicts/network-angle3 --prefix experiments/2026-11-04/round3/conflicts-original/wide --output experiments/2026-11-04/round3/intervals-verified --minimum-cells 1
```

## Stronger Hamming-tour ordering

If an all-zero mask closes the column tour, its Hamming length is exactly twice the total number of true intervals across all rows. Therefore a finite TSP ordering heuristic directly minimizes the encoding size. Native deterministic best-improvement2-opt preserves the columns and all compatibility values.

- Mirror4/8: identity15,405 →3,817 after400 improving swaps →3,570 at a2-opt local optimum after143 additional swaps. This is79.3× fewer scalar table entries than the binary allowed table.
- Original6/10: 7,124 →3,134 after400 swaps →2,930 at a local optimum after141 further swaps;147.2× scalar reduction.
- Mirror5/8 remains707, already minimal for its nonempty row classes;236× scalar reduction.

The extra convergence pass took approximately0.2seconds per difficult relation after compilation. This is a heuristic ordering optimum, with no claim of a globally shortest tour. The categorical relation itself remains exact. Artifacts and reconstruction checks are in `intervals-hamming/`.

## Tests and process isolation

Twenty-one focused tests pass. They exhaustively enumerate CP-SAT projections of synthetic matrices, including duplicate row/column classes, nontrivial rank permutations, all-allowed and all-forbidden cases. Actual matrix tests exercise every original categorical tuple through the element maps for the three critical relations and instantiate positive CP witnesses. Hamming-tour objective checks and the separate Hamming builder also pass exhaustive synthetic CP projections.

```text
.venv/bin/python -m pytest -q tests/test_interval_pair_relation.py
```

The active master and area helper are untouched. Arena's combined runner imports the stable identity helper using a process-local hook. The optional stronger helper is separate: `interval_order_optimization.add_hamming_interval_pair`. A standalone CLI composes it with the same support-bundle runner:

```text
.venv/bin/python scripts/experiment_hamming_support_bundle.py --domains DOMAINS.npz --tables TABLES.npz --output OUTPUT --seconds 180
```

It enables interval encoding inside its own process and restores the imported helper afterward. This task establishes an exact, smaller finite model; it does not establish first place or alter the shared solution.
