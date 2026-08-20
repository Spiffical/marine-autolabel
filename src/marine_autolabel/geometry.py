"""Pure mask and click geometry.

Everything here is numpy/OpenCV only -- no SAM3, no model calls -- which makes
the algorithmic core of the click engine testable without a GPU.

Gathered from `click_engine_probe.py`, `run_presentation_custom_flow.py` and
`tile_stray_bakeoff.py`, where these helpers were duplicated or near-duplicated.
Behaviour is preserved exactly; the one consolidation is `norm_to_pixel`, which
was open-coded as `int(round(v * (W - 1)))` with clamping in several places.

Click coordinates are normalised to [0, 1] against the FULL frame everywhere in
this pipeline, never against a crop.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def norm_to_pixel(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    """Normalised (x, y) -> clamped integer pixel coordinates."""
    px = min(width - 1, max(0, int(round(float(x) * (width - 1)))))
    py = min(height - 1, max(0, int(round(float(y) * (height - 1)))))
    return px, py


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two binary masks. 0.0 when both are empty."""
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def overlap_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over the area of the *smaller* mask.

    Unlike IoU this stays high when a small mask sits wholly inside a large one,
    which is the containment case that IoU misses.
    """
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    smaller = min(int(a.sum()), int(b.sum()))
    if smaller == 0:
        return 0.0
    return float(np.logical_and(a, b).sum()) / float(smaller)


def keep_positive_seed_components(mask: np.ndarray, clicks: list[dict[str, Any]]) -> np.ndarray:
    """Drop disconnected SAM3 spill while retaining every positively seeded component.

    Fine branch gaps are left untouched. With multiple positive clicks, target
    pieces separated by occlusion are retained deliberately by seeding each piece.
    If rounding puts every positive seed just outside the mask, the largest
    component is kept as a safe fallback rather than returning nothing.
    """
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return mask
    positive = [c for c in clicks if int(c.get("label", 1)) == 1]
    if not positive:
        return mask

    n_labels, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if n_labels <= 2:
        return mask

    height, width = mask.shape
    seeded: set[int] = set()
    for click in positive:
        x, y = norm_to_pixel(click["x"], click["y"], width, height)
        label = int(labels[y, x])
        if label:
            seeded.add(label)

    if not seeded:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        seeded.add(int(np.argmax(sizes)))
    return np.isin(labels, list(seeded))


def clean_candidate_components(
    masks: np.ndarray, clicks: list[dict[str, Any]]
) -> np.ndarray:
    return np.stack([keep_positive_seed_components(mask, clicks) for mask in masks])


def select_in_band(masks: np.ndarray, scores: np.ndarray) -> tuple[int, str]:
    """Pick the smallest candidate whose area is plausible for one organism.

    SAM3's multimask output for a click typically spans part-of-object,
    whole-object and whole-scene. The smallest mask inside the plausible band is
    the individual; larger ones have merged neighbours or grabbed the substrate.

    Returns `(index, reason)`.
    """
    height, width = masks.shape[1], masks.shape[2]
    total = float(height * width)
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    min_area = max(200, int(0.001 * total))
    max_area = 0.5 * total

    eligible = sorted((int(a), i) for i, a in enumerate(areas) if min_area <= a <= max_area)
    if eligible:
        return eligible[0][1], "smallest_in_band"
    above = sorted((int(a), i) for i, a in enumerate(areas) if a >= min_area)
    if above:
        return above[0][1], "smallest_above_min"
    return int(np.argmax(scores)), "fallback_highest_score"


def smallest_valid(masks: np.ndarray, valid: Any) -> int:
    """Index of the smallest-area candidate among those passing `valid`.

    The caller must ensure at least one entry is True.
    """
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    return sorted((int(areas[i]), i) for i in range(len(areas)) if valid[i])[0][1]


def mask_level_nms(
    results: list[dict[str, Any]],
    threshold: float = 0.5,
    containment_threshold: float = 0.9,
) -> tuple[list[dict[str, Any]], int]:
    """Remove duplicate or over-broad masks, preferring tighter ones.

    IoU alone misses a common crowded-coral failure: a clean individual mask sits
    wholly inside a much larger mask that merged a neighbouring colony. Smaller
    masks are processed first, so strict containment rejects the broad container
    only when the broad proposal's own seed also falls inside the tight mask.
    Distinct seeds preserve interlaced colonies whose silhouettes legitimately
    overlap.

    Returns `(kept_in_original_order, removed_count)`.
    """

    def seed_inside(result: dict[str, Any], mask: np.ndarray) -> bool:
        click = result.get("seed_click") or {}
        if not isinstance(click.get("x"), (int, float)) or not isinstance(
            click.get("y"), (int, float)
        ):
            return False
        height, width = mask.shape
        x, y = norm_to_pixel(click["x"], click["y"], width, height)
        return bool(mask[y, x])

    kept: list[tuple[int, dict[str, Any]]] = []
    removed = 0
    order = sorted(range(len(results)), key=lambda i: int(np.asarray(results[i]["mask"]).sum()))

    for index in order:
        result = results[index]
        mask = np.asarray(result["mask"]).astype(bool)
        if mask.any() and any(
            iou(mask, np.asarray(old["mask"]).astype(bool)) >= threshold
            or (
                overlap_coefficient(mask, old["mask"]) >= containment_threshold
                and seed_inside(result, np.asarray(old["mask"]).astype(bool))
            )
            for _, old in kept
        ):
            removed += 1
            continue
        kept.append((index, result))

    kept.sort(key=lambda item: item[0])
    return [result for _, result in kept], removed


def geom_dedup(
    groups: list[dict[str, Any]], width: int, height: int, px: int = 40
) -> tuple[list[dict[str, Any]], int]:
    """Merge click groups whose first click is within `px` of an earlier one.

    A deterministic duplicate remover for two markers landing on the same animal.
    Preferred over an MLLM review pass, which costs recall. Surviving groups are
    renumbered from 1.
    """
    kept: list[dict[str, Any]] = []
    removed = 0
    for group in groups:
        click = group["clicks"][0]
        if any(
            abs(click["x"] - k["clicks"][0]["x"]) * width < px
            and abs(click["y"] - k["clicks"][0]["y"]) * height < px
            for k in kept
        ):
            removed += 1
            continue
        kept.append(group)
    for index, group in enumerate(kept, 1):
        group["id"] = index
    return kept, removed


def duplicate_click(
    clicks: list[dict[str, Any]], candidate: dict[str, Any], tolerance: float = 1e-3
) -> bool:
    """Detect a no-progress click: same label, effectively the same point."""
    return any(
        int(click.get("label", -1)) == int(candidate.get("label", -2))
        and (
            (float(click.get("x", -10.0)) - float(candidate.get("x", 10.0))) ** 2
            + (float(click.get("y", -10.0)) - float(candidate.get("y", 10.0))) ** 2
        )
        ** 0.5
        <= tolerance
        for click in clicks
    )


def merge_corrected_positive_click(
    clicks: list[dict[str, Any]], corrected: dict[str, Any]
) -> list[dict[str, Any]]:
    """Replace the nearest positive seed with `corrected`, keeping all others."""
    merged = [dict(click) for click in clicks]
    positives = [i for i, click in enumerate(merged) if int(click.get("label", 1)) == 1]
    if not positives:
        return [dict(corrected)] + merged
    nearest = min(
        positives,
        key=lambda i: (float(merged[i]["x"]) - float(corrected["x"])) ** 2
        + (float(merged[i]["y"]) - float(corrected["y"])) ** 2,
    )
    merged[nearest] = dict(corrected)
    return merged
