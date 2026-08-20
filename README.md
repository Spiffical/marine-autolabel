# marine-autolabel

MLLM-guided exhaustive instance segmentation of marine life in underwater video.

Per ~10 s clip: pick frames, screen them for quality, run a SAM3 text-prompt agent
for a first pass, then run a **click engine** in which a vision MLLM places point
clicks on the organisms the first pass missed, SAM3 masks each click, and a
verification pass accepts or rejects each mask.

The design goal is *recall* — finding every organism in the frame — because the
first pass reliably misses camouflaged, small and partially occluded animals.

See [docs/architecture.md](docs/architecture.md) for how it works and
[docs/running.md](docs/running.md) to run it.

## Layout

| Path | What |
|---|---|
| `src/marine_autolabel/config.py` | `RunConfig` — a run is a file, not 40 CLI flags |
| `src/marine_autolabel/prompts/` | system prompts, `_general` / `_underwater` pairs |
| `src/marine_autolabel/sam3svc/` | SAM3 point + text services |
| `src/marine_autolabel/firstpass/` | text-prompt agent driver |
| `src/marine_autolabel/clickengine/` | discovery, mask generation, verification |
| `src/marine_autolabel/postprocess/` | dedup, NMS, overlap consolidation, matching |
| `src/marine_autolabel/pipeline/` | stage graph, resumable runs |
| `src/marine_autolabel/eval/` | scoring, SeaTube annotation matching, repeats |
| `configs/` | benchmark frame manifests |
| `benchmarks/` | FathomNet and model-matrix baselines |
| `infra/` | GPU sync, SLURM |

## Install

SAM3 is an optional extra, so the pure-python stages install and test without CUDA:

```bash
pip install -e '.[dev]'            # config, rle, postprocess, eval, prompts
pip install -e '.[dev,sam3]'       # + SAM3, torch (GPU box)
```

The `sam3` extra pins [Spiffical/sam3](https://github.com/Spiffical/sam3), a fork of
`facebookresearch/sam3` carrying the model patches this pipeline needs — notably
`add_mask_prompt()` for injecting click-derived masks into the video tracker, and a
RoPE fix in `vitdet.py` that makes non-native `image_size` work. That fork stays
under the Meta SAM License; the code in *this* repo does not.

## Run

```bash
cp .env.example .env    # ANTHROPIC_API_KEY + data roots
mal-pipeline --config examples/run_example.json
```

## Testing

```bash
pytest -q                       # pure-python stages
pytest -m gpu                   # needs CUDA + the sam3 extra
pytest -m api                   # needs a live ANTHROPIC_API_KEY
```

CI runs the unmarked set on Python 3.10, matching the GPU box.
