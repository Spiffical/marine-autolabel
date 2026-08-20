# Fixtures

Synthetic, with the exact schema a real run emits.

`firstpass/empty_*` has ZERO objects on purpose. A first pass finding nothing is
a real production case, not a degenerate one -- the click engine then has to do
all the work -- and several loaders and scorers used to crash on it.

`run/summary.json` carries the full key set a run records. `test_config.py`
asserts `RunConfig` covers every one, so a field dropped or renamed during a
refactor fails here rather than silently reverting to a default mid-run.

Real captured run data is kept out of this repository: it identifies datasets
and third-party annotators.
