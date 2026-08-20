# Running the pipeline

## Install

SAM3 is an optional extra, so the pure-python stages install and test without CUDA:

```bash
pip install -e '.[dev]'          # config, rle, postprocess, eval, prompts
pip install -e '.[dev,sam3]'     # + SAM3 and torch, on the GPU machine
```

## Configure

Copy `.env.example` to `.env` and fill it in. Nothing machine-specific belongs
in a tracked file — data roots, the GPU host and the API key all live here.

A run is described by a config file, not by flags. See
`examples/run_example.json`. Unknown keys are **rejected**: a
typo would otherwise leave the default silently in place, and these runs are too
expensive to discover that from the results.

## Check before you spend

```bash
mal-pipeline --config examples/run_example.json --dry-run
```

Resolves the config and manifest and reports what would run, without loading
SAM3 or making a single API call.

## Run

```bash
mal-pipeline --config examples/run_example.json
```

Runs are **resumable**. A frame that completed cleanly is reused; a frame whose
model calls failed is retried, because its results are wrong rather than merely
incomplete. Use `--no-resume` to force reprocessing, `--frame-id` to narrow.

A frame that raises is recorded and the run continues — losing a whole fan-out
to one bad frame is the expensive outcome.

## On the GPU host

```bash
infra/sync_to_remote.sh            # preview
infra/sync_to_remote.sh --apply    # transfer source only
infra/remote_python.sh -m pytest -q
```

The sync never carries `.env`, data, weights or run outputs, and never deletes
remote files — remote outputs and datasets are persistent state.

## Testing

```bash
pytest -q          # pure-python stages: no GPU, no API key, no network
pytest -m gpu      # needs CUDA and the sam3 extra
pytest -m api      # needs a live ANTHROPIC_API_KEY
```

The default set makes **no API calls**: every model- and SAM3-dependent step is
an injected callable driven by fakes. Keep it that way — spend the key on
results, not on tests.

To check a port against the original repo:

```bash
MAL_LEGACY_REPO=~/path/to/sam3-autolabeling pytest tests/test_parity_parsers.py
```

## Reading a run

`summary.json` in the output root carries the config, provenance and per-frame
results. Each pass records where its proposals went:

```
n_proposed -> n_run -> n_generated -> n_verify_kept -> n_recovered
```

with `n_repeat_skipped`, `n_dedup_removed`, `n_verify_dropped`,
`n_nms_removed`, `n_low_confidence` and `hit_round_cap` alongside. If a pass
proposed a lot and kept little, that line says which stage took them.

**Look at the composites.** Do not report a number you have not visually
spot-checked; it catches wrong conclusions that the summary alone does not.

## Evaluating a change

Any MLLM-in-the-loop change needs **repeats >= 3**, reported as mean ± std. A
single run is a plumbing check, not a result. `eval/repeats.py` enforces this and
also refuses to average runs whose model or generator differed — those are
different experiments.
