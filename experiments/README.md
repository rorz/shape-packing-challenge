# Packing search records

`original.json` preserves the input packing. `best.json` is the best candidate
accepted by the server; `api_best.json` records its exact score and file hash.
`best_run.json` records the configuration that produced it. No entry was
submitted to the public leaderboard.

The `branch-*.json` files are full precision continuous seeds with fixed angle
differences at the two nearly coincident flat contacts. They are search inputs,
not submission files. `best_phase*_seed.json` also preserve successful global
orientations. `benchmark.json` records local throughput.

Other local files are ignored diagnostics from the bounded experimental runs.
They include invalid near-solutions. Use `run.py` and the API checker before
promoting any candidate.
