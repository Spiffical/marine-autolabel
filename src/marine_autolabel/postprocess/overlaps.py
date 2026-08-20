"""Finding masks that may share one identity.

Ported from `run_presentation_custom_flow._ambiguous_overlap_components` and
`_duplicates_known_mask`.

In crowded branching scenes -- coral thickets especially -- one organism is
often segmented as several masks, while genuinely distinct colonies interlace
and overlap. Geometry alone cannot tell those apart, so this module only
*selects candidates* for identity review. It never merges anything: temporal and
visual identity evidence plus a strict union-mask verifier make the actual
consolidation decision downstream.

Two triggers put a pair up for review:
  overlap       the smaller mask is >=40% inside the larger
  near_contact  the masks come within a hair of touching, scaled to the frame
                diagonal so the threshold is resolution independent
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..geometry import iou as mask_iou
from ..geometry import overlap_coefficient


def ambiguous_overlap_components(
    masks: list[np.ndarray],
    *,
    containment_threshold: float = 0.40,
    max_gap_fraction: float = 0.006,
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """Group mask indices that may share an identity.

    Returns `(components, pairs)`. Components are connected groups of indices,
    each sorted, and only groups of size >= 2 appear. Pairs carry the evidence
    for each flagged relationship, which the review stage shows the model.
    """
    normalized = [np.asarray(mask).astype(bool) for mask in masks]
    adjacency: dict[int, set[int]] = {index: set() for index in range(len(masks))}
    pairs: list[dict[str, Any]] = []

    if normalized:
        height, width = normalized[0].shape
        max_gap_px = float(max_gap_fraction) * float(np.hypot(width, height))
        # Distance from every pixel to the nearest pixel of each mask.
        distance_maps = [
            cv2.distanceTransform(np.logical_not(mask).astype(np.uint8), cv2.DIST_L2, 3)
            for mask in normalized
        ]
    else:
        max_gap_px = 0.0
        distance_maps = []

    for left in range(len(masks)):
        for right in range(left + 1, len(masks)):
            if not normalized[left].any() or not normalized[right].any():
                # An empty mask has no nearest pixel; there is nothing to relate.
                continue
            overlap = overlap_coefficient(masks[left], masks[right])
            gap_px = float(
                min(
                    distance_maps[left][normalized[right]].min(),
                    distance_maps[right][normalized[left]].min(),
                )
            )
            trigger = (
                "overlap"
                if overlap >= containment_threshold
                else "near_contact"
                if gap_px <= max_gap_px
                else None
            )
            if trigger is None:
                continue

            adjacency[left].add(right)
            adjacency[right].add(left)
            pairs.append(
                {
                    "left_index": left,
                    "right_index": right,
                    "smaller_mask_overlap": float(overlap),
                    "iou": float(mask_iou(normalized[left], normalized[right])),
                    "gap_px": gap_px,
                    "trigger": trigger,
                }
            )

    components: list[list[int]] = []
    unseen = {index for index, neighbours in adjacency.items() if neighbours}
    while unseen:
        stack = [min(unseen)]
        component: set[int] = set()
        while stack:
            index = stack.pop()
            if index in component:
                continue
            component.add(index)
            stack.extend(adjacency[index] - component)
        unseen -= component
        components.append(sorted(component))
    return components, pairs


def duplicates_known_mask(
    mask: np.ndarray,
    known: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.5,
    containment_threshold: float = 0.8,
) -> bool:
    """Is this mask a repeat of something already accepted?

    Uses containment as well as IoU so a re-segmentation at a slightly different
    extent is still caught, without merging nearby distinct organisms.
    """
    return any(
        mask_iou(mask, np.asarray(item["mask"]).astype(bool)) >= iou_threshold
        or overlap_coefficient(mask, item["mask"]) >= containment_threshold
        for item in known
    )
