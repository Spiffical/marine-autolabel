"""Matching click-engine masks against the first pass.

Ported from `run_presentation_custom_flow._match_firstpass`. Decides, for each
mask the click engine produced, whether it re-found something the first pass
already had or is a new candidate.

Two passes, in order:

  1. Greedy IoU. Highest-IoU pairs first, each mask and each first-pass object
     used at most once.
  2. Containment rescue. A click refiner can return a high-quality *part* of an
     organism -- a fish's head, say -- whose union IoU with the whole first-pass
     mask falls below threshold even though it is almost entirely inside it.
     Rescued when the seed click lands inside a known mask, or >=80% of the
     smaller mask overlaps. This is safer than lowering the global IoU
     threshold, which would start merging nearby distinct animals.

Anything still unmatched is a stray, which the verifier then splits into
new candidates and false positives.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..geometry import iou as mask_iou

CONTAINMENT_THRESHOLD = 0.8


def match_firstpass(
    click_results: list[dict[str, Any]],
    firstpass: list[dict[str, Any]],
    threshold: float = 0.5,
) -> dict[int, dict[str, Any]]:
    """Map click-result index -> match info. Unmatched indices are absent."""
    pairs: list[tuple[float, int, int]] = []
    for click_index, result in enumerate(click_results):
        mask = np.asarray(result["mask"]).astype(bool)
        if not mask.any():
            continue
        for first_index, existing in enumerate(firstpass):
            pairs.append((mask_iou(mask, existing["mask"]), click_index, first_index))

    used_clicks: set[int] = set()
    used_firstpass: set[int] = set()
    matches: dict[int, dict[str, Any]] = {}
    for score, click_index, first_index in sorted(pairs, reverse=True):
        if score < threshold or click_index in used_clicks or first_index in used_firstpass:
            continue
        used_clicks.add(click_index)
        used_firstpass.add(first_index)
        matches[click_index] = {
            "firstpass_index": first_index,
            "iou": float(score),
            "method": "iou",
        }

    for click_index, result in enumerate(click_results):
        if click_index in matches:
            continue
        click_mask = np.asarray(result["mask"]).astype(bool)
        if not click_mask.any():
            continue

        click = result.get("seed_click") or {}
        px = int(round(float(click.get("x", -1.0)) * (click_mask.shape[1] - 1)))
        py = int(round(float(click.get("y", -1.0)) * (click_mask.shape[0] - 1)))

        candidates = []
        for first_index, existing in enumerate(firstpass):
            known = existing["mask"]
            intersection = int(np.logical_and(click_mask, known).sum())
            union = int(np.logical_or(click_mask, known).sum())
            score = intersection / union if union else 0.0
            smaller = min(int(click_mask.sum()), int(known.sum()))
            overlap = intersection / smaller if smaller else 0.0
            seed_inside = (
                0 <= px < known.shape[1] and 0 <= py < known.shape[0] and bool(known[py, px])
            )
            if seed_inside or overlap >= CONTAINMENT_THRESHOLD:
                candidates.append(
                    (
                        seed_inside,
                        overlap,
                        score,
                        first_index,
                        "seed_inside_firstpass" if seed_inside else "mask_containment",
                    )
                )

        if candidates:
            _, overlap, score, first_index, method = max(candidates)
            matches[click_index] = {
                "firstpass_index": first_index,
                "iou": float(score),
                "overlap_coefficient": float(overlap),
                "method": method,
            }
    return matches
