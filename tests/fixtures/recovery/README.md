# Recovery fixtures

Four independently reference-validated raw submissions and their shared continuous mirror seed. The incumbent is calibration; the other three cases are held out from width calibration and span several exact flat-angle branches. These are historical solutions, not new searches.

`manifest.json` records SHA-256 hashes, reference scores, original paths and sidecar provenance. Fixture paths are relative to this directory. The source experiment directories are not required to run the benchmark.

The `*-origin.json` files are optional diagnostics. Three retain the full-precision posed origin from the target's own sidecar. The incumbent retains its documented unposed section because the old campaign did not preserve the exact winning frame. Its first recorded frame must not be substituted for that missing checkpoint.

Target coordinates, saved integer move coefficients and target-derived displacement directions must never be injected into generated search domains. Fitting five continuous family/rigid parameters to a target is explicitly an optimistic coverage test, not independent rediscovery. Every benchmark report must distinguish membership, unhinted finite solving, and blind global rediscovery.
