# Frozen radius/overlap frontier and candidate rescue

## Audit hypothesis and scope

Earlier min-max rankings used a search radius2.94852698, which can favour a higher-radius state with tiny overlaps over a repairable state already below first place. Separate true minimum-enclosing-circle radius from overlap deficit, freeze source bytes before evaluating, and use full GEOS checks for every low-radius state.

`experiment_rescue_frontier.py` inspected4,035JSON paths, froze3,447 distinct submitted geometries, and completed full105-pair GEOS checks on1,331 states. Every frozen state at or below leader+5nm was checked exactly. All3,447 MECs received native boundary/support certificates; the selected witness also passed an independent reference MEC calculation. Higher-radius frontier entries are from the explicitly checked near controls, not a claim that every high-radius input received GEOS evaluation.

Overlap deficits use `(GEOS area -1e-6) / max(norm(dArea/dPlacement),1e-3)`, clipped at zero. These are normalized equivalent-displacement values, not a proof of how far a feasible repair lies away. Source bytes, hashes and aliases are retained in `snapshots/` and `sources.json`; the full machine-readable audit is `summary.json`.

## Stronger saved witness

Only one frozen state met both radius≤leader+5nm and normalized maximum overlap<15nm:

- Source: `tied/winning-section/nearest-raw.json`; frozen SHA256 `4db98d5d40f55447233dbe94f88b349928757fab3bd26e9fc44fffeea76bf2a8`.
- Native radius2.948526904838365; independent reference2.9485269048382277. It is19.047nm below first place.
- Three exact GEOS overlaps:0/9=4.265nm,1/2=9.154nm,2/10=6.720nm.
- The saved43-row tied basis matches the raw witness exactly. After quotienting only integer translations,41 physical directions remain. All angular directions are retained.

The earlier five-overlap witness therefore was not the strongest saved low-radius state. The audit also found `mirror/campaign-a/nearest-raw.json` at50.105nm below leader with five overlaps/max18.969nm.

## Bounded exact rescue

`experiment_rescue_csp.py` ran a300-second total budget. It first passed a separate known-valid55-support engine control; the invalid witness was never represented as a valid control.

- XYZ±1/theta±3 local box:63 full poses per piece, exact pair tables,55 support cuts, Hamming intervals. Exactly INFEASIBLE at target2.9485269238 in6.27seconds. The overlap-only CSP is feasible; enclosure constraints cause this finite rejection.
- Full signed41-direction union, then XYZ±1/theta±3:2,457–3,843 poses per piece. Exact preparation hit its deadline after17.5million native pair evaluations and at least706,475 completed-pair GEOS evaluations. First pair alone took244seconds. Result UNKNOWN due to incomplete tables; no infeasibility claim. The completed pair cache remains available.
- Total runtime300.86seconds. No new candidate or API improvement. Root separately owns a wider625-pose positional/angular box around the same witness.

## Later live witness

Tempered reported a new broad-L point after the initial frozen audit. Its raw input and matching state were frozen under `broad-l/`: radius16.076nm below leader, four GEOS overlaps, max normalized deficit9.013nm. It trades3nm of radius headroom for a slightly smaller maximum deficit than the selected three-overlap point. It has not been silently merged into the earlier audit snapshot.
