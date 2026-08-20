# Working agreement

## Read first

1. [docs/architecture.md](docs/architecture.md) — how the pipeline works
2. [docs/running.md](docs/running.md) — how to run and read a run

Machine-specific detail (hosts, paths, credentials) lives in `.env` and
`docs/LOCAL_NOTES.md`, both untracked. Do not put any of it in a tracked file.

## Spending the API key

The pipeline's calls bill a real key, and a runaway loop bills it fast.

- The default test suite makes **no API calls**. Every model- and SAM3-dependent
  step is an injected callable; drive them with fakes.
- Before a live run, `--dry-run` costs nothing and catches config mistakes.
- Any loop that ends when "the model decides it is done" must carry a hard cap.
  Two already do (`max_repair_rounds`, `max_convergence_passes`) because both
  were found unbounded.

## Evaluating changes

- **repeats >= 3**, reported mean ± std. A single run is a plumbing check.
- Do not average runs whose model or generator differed. `eval/repeats.py`
  refuses to.
- Look at the rendered composites. Write honest conclusions, failures included.
- Ground truth is the first pass and it is **incomplete**. Recall is a lower
  bound; stray counts are an upper bound; more masks is not automatically better.
- Results, presentation material and captured run data belong in `private/`,
  which is gitignored. This repository ships the pipeline, not the findings.

## Changing behaviour

Several things read like bugs and are load-bearing. Each has a test saying so,
and the test name explains why. Before "fixing" one, read it:

- a missing verdict KEEPS a mask (incomplete ground truth)
- the generator's area cap is deliberately loose at 0.60
- the whole-frame MLLM click-review pass stays OFF by default; enabling it
  reduced recall in testing
- consolidation preserves originals when either gate is unsure

## Repository rules

- No hostnames, usernames or absolute machine paths in tracked files. Use
  `${MAL_*}` placeholders and `.env`.
- SAM3 stays an optional extra so the pure stages test without CUDA.
- The prompt text is tuned. If you must restructure how a prompt is assembled,
  prove the output is byte-identical — see `tests/test_parity_parsers.py`.
- Record provenance with every scored run. The git SHA alone is not enough;
  scored runs have been made from dirty trees, so `RunConfig.write()` also
  stamps per-file sha256.

## Related repositories

The SAM3 fork ([Spiffical/sam3](https://github.com/Spiffical/sam3)) holds the
Meta-derived model patches and agent loop, pinned by SHA in the `sam3` extra.
Model changes belong there plus a pin bump here, never as a local patch.
