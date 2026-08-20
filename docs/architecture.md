# How the pipeline works

The goal is **recall**: finding every organism in a frame. The first pass
reliably misses camouflaged, small, thin and partially occluded animals, so a
second stage goes looking for what it missed.

In practice the binding constraint is **click placement**, not SAM3's masking:
effort spent on finding the organism returns more than effort spent refining the
mask around it.

## The two stages

**First pass — text prompts.** A cheaper model proposes short phrases and SAM3
grounds them into instance masks. Phrases are *retrieval handles*, not taxonomy;
they exist only to make SAM3 ground something. An empty first pass is normal --
on a dense scene the text stage can ground nothing at all.

**Click engine — the recovery stage.** A stronger vision model looks at what is
already covered and clicks what is not. Each click group becomes a SAM3 mask,
which is then verified, possibly repaired, and either accepted or dropped.

## Per-frame sequence

`pipeline/frame.py` is the whole sequence, and each model- or SAM3-dependent
step is injected — which is why the test suite needs no GPU and no API key.

```
quality screen        drop corrupted or black frames before spending anything
      |               (rank01's chosen frames are all black)
load known masks      first-pass output + any accepted masks carried in
      |
  +---+ for each discovery pass ------------------------------+
  |  render views     raw / grid / strong / outline (+ focus)  |
  |  discover         model proposes click groups for a region |
  |  suppress repeats drop same-description same-location      |
  |  dedup clicks     40 px, deterministic                     |
  |  generate         SAM3 in the loop -> one mask per group   |
  |  verify           strict complete-and-single identity      |
  |  repair           feed repair clicks back, BOUNDED         |
  |  NMS + confidence                                          |
  |  accepted masks join the known set  <---------------------+
  +------------------------------------------------------------+
      |
consolidate           merge only masks judged to be one organism
```

Accepted masks re-enter the known set between passes. That is what stops a later
pass re-finding what an earlier one already got.

## The three coverage views

Each hides something the others show, which is why all three are sent:

| View | Shows | Hides |
|---|---|---|
| `grid` (light fill + coordinates) | overall coverage | faint life under the tint |
| `strong` (heavy fill) | whether one structure is covered at all | everything beneath it |
| `outline` (contours only) | colonies visible *through* an accepted silhouette | at-a-glance coverage |

The outline view exists because in a crowded thicket the filled inter-branch
space of one accepted mask makes the colonies behind it look covered. It is the
decisive view for branching scenes.

## Mask generation

| Generator | Behaviour |
|---|---|
| full-frame | tight masks, but abandons thin or low-contrast organisms |
| zoom | crops and upscales so a small organism fills the view; **never abandons** |
| **hybrid** | full-frame first, zoom only on abandon or a degenerate mask — the recommended default |

The zoom path runs only on a seed the full-frame pass already worked from, so
"there is nothing here" is not an answer it may give.

## Things that look like bugs and are not

- **A missing or unparseable verdict KEEPS the mask.** Ground truth is
  incomplete; an unjudged mask is more likely a real organism than a false
  positive, and dropping it costs recall.
- **Strays are mostly real.** A mask matching nothing in the first pass usually
  means the first pass was wrong, not that the click engine hallucinated.
- **An empty proposal is a stop signal**, in convergence mode. It means the
  model has run out of things to find.
- **Everything is bounded.** The repair loop and convergence mode both end on a
  model behaving well, so both carry hard caps. Each repair round costs one SAM3
  generation plus one verification call per pending mask.

## Where the numbers come from

`eval/score.py` scores against the first pass, which is **incomplete** — so
recall is a lower bound and stray counts an upper bound. `eval/seatube.py`
matches against external annotations by majority consensus over repeats, and
never invents taxonomy it was not given.

Evaluate any change at repeats >= 3, mean ± std. A single run is a plumbing
check, not a result.
