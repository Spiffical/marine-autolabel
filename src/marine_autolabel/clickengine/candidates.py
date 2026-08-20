"""Filtering candidate masks before they reach the model.

Ported from `som_missed_creatures.py`. Cheap geometric rejections run before any
API call, so obvious non-organisms never cost tokens.

Order matters and is preserved: size, then edge clipping, then duplication
against what is already accepted, then the multi-component test. The last is the
most expensive and the most specific, so it runs last.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..geometry import iou as mask_iou

DROP_TOO_SMALL = "too_small"
DROP_TOO_LARGE = "too_large"
DROP_MULTI_EDGE = "multi_edge_clipped"
DROP_DUPLICATE = "duplicate_of_existing"
DROP_MULTI_COMPONENT = "multi_component_blob"


def touches_edges(mask: np.ndarray, edge_tol_px: int) -> int:
    """How many frame edges the mask comes within `edge_tol_px` of.

    A mask touching two or more edges is usually substrate or a lighting
    artefact spanning the frame rather than an organism.

    `edge_tol_px=0` disables the test: every slice becomes empty, so the count
    is always 0.
    """
    height, width = mask.shape
    tol = edge_tol_px
    return sum(
        [
            bool(mask[:tol, :].any()),
            bool(mask[height - tol :, :].any()),
            bool(mask[:, :tol].any()),
            bool(mask[:, width - tol :].any()),
        ]
    )


def count_significant_components(mask: np.ndarray, *, min_component_frac: float = 0.15) -> int:
    """Connected components holding at least `min_component_frac` of the area.

    One organism yields 1. A blob that merged several distinct objects yields 2
    or more. The fraction threshold keeps stray specks from counting.
    """
    arr = np.asarray(mask, dtype=np.uint8)
    if arr.sum() == 0:
        return 0
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(arr, connectivity=8)
    threshold = max(1, int(min_component_frac * int(arr.sum())))
    return sum(1 for i in range(1, n_labels) if int(stats[i, 4]) >= threshold)


def filter_candidates_with_reasons(
    candidates: list[dict[str, Any]],
    existing_masks: list[dict[str, Any]],
    *,
    iou_dedup: float,
    min_area_px: float,
    max_area_frac: float,
    edge_tol_px: int,
) -> list[tuple[dict[str, Any], str | None]]:
    """Return `(candidate, drop_reason)` for every input; None means kept."""
    existing = [np.asarray(entry["mask"], dtype=bool) for entry in existing_masks]
    results: list[tuple[dict[str, Any], str | None]] = []

    for candidate in candidates:
        mask = np.asarray(candidate["mask"], dtype=bool)
        height, width = mask.shape
        area = int(mask.sum())

        if area < min_area_px:
            results.append((candidate, DROP_TOO_SMALL))
        elif area > max_area_frac * height * width:
            results.append((candidate, DROP_TOO_LARGE))
        elif touches_edges(mask, edge_tol_px) > 1:
            results.append((candidate, DROP_MULTI_EDGE))
        elif any(mask_iou(mask, known) > iou_dedup for known in existing):
            results.append((candidate, DROP_DUPLICATE))
        elif count_significant_components(mask, min_component_frac=0.15) > 1:
            results.append((candidate, DROP_MULTI_COMPONENT))
        else:
            results.append((candidate, None))
    return results


def filter_candidates(
    candidates: list[dict[str, Any]], existing_masks: list[dict[str, Any]], **kwargs: Any
) -> list[dict[str, Any]]:
    """Survivors only. See `filter_candidates_with_reasons` for the reasons."""
    return [
        candidate
        for candidate, reason in filter_candidates_with_reasons(
            candidates, existing_masks, **kwargs
        )
        if reason is None
    ]
