# Packing search records

`original.json` preserves the input packing. `best.json` is the best candidate
accepted by the server. Its current validation and promotion records are
[`pantheon-pilot/candidate-api.json`](pantheon-pilot/candidate-api.json) and
[`pantheon-pilot/promotion.json`](pantheon-pilot/promotion.json).
`api_best.json` and `best_run.json` describe the earlier incumbent. No entry was
submitted to the public leaderboard by these research runs.

The `branch-*.json` files are full precision continuous seeds with fixed angle
differences at the two nearly coincident flat contacts. They are search inputs,
not submission files. `best_phase*_seed.json` also preserve successful global
orientations. `benchmark.json` records local throughput.

Git retains the source in `scripts/`, tests and their small immutable fixtures,
research Markdown, selected continuous seeds, and the curated evidence listed
in each campaign's `.gitignore`. The dated research overview starts at
[`2026-11-04/README.md`](2026-11-04/README.md).

Bulk run directories, source snapshots, per-step logs, candidate populations,
NumPy tables and generated plots remain on disk but are ignored. Historical
logs can reference those local artifacts; a fresh clone does not contain the
complete run archive. Existing tests that replay large archived tables skip
when those optional files are absent. Required regression and benchmark inputs
belong in `tests/fixtures/`, not in an ignored run directory.

To retain a new result, add its small input or validation record explicitly to
the campaign's allowlist. Do not unignore an entire run tree. Ignored diagnostics
can include invalid near-solutions; use `run.py` and the API checker before
promoting any candidate.
