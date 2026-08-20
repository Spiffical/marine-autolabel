"""Scoring click placement and detections against ground truth.

Ported from `click_engine_probe.py`.

"Ground truth" here is the first-pass output, which is *incomplete* -- that is
the central caveat of every number this module produces. A stray click is
usually a real organism the first pass missed, not a false positive, so treat
recall as a lower bound and stray counts as an upper bound.

Empty ground truth is a real case, not a degenerate one -- a first pass can
legitimately find nothing. Every rate here is defined for that case rather than
dividing by zero.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..geometry import norm_to_pixel

COVERAGE_TOLERANCES = (0, 8, 16, 24, 40)


def recall(n_detected: int, n_ground_truth: int) -> float:
    """Detected over ground truth; 0.0 when there is no ground truth.

    Dividing directly would raise ZeroDivisionError on an empty GT, which is
    reachable whenever the first pass finds nothing.
    """
    if n_ground_truth <= 0:
        return 0.0
    return n_detected / n_ground_truth


def score_clicks(
    groups: list[dict[str, Any]],
    gt: list[dict[str, Any]],
    width: int,
    height: int,
    tol_px: int = 8,
) -> dict[str, Any]:
    """How well did the model's clicks cover the known organisms?

    A ground-truth creature counts as hit when a positive click lands inside its
    mask dilated by `tol_px`. The tolerance reflects that a click a few pixels
    off a thin body is still on target for SAM3 point mode.

    Reports nearest-click distance per creature as well as hits, which separates
    a precision miss (clicked near the body, just outside) from a detection miss
    (never looked there at all).
    """
    fg_clicks: list[dict[str, Any]] = []
    for group in groups:
        for click in group["clicks"]:
            if click["label"] == 1:
                px, py = norm_to_pixel(click["x"], click["y"], width, height)
                fg_clicks.append(
                    {
                        "px": px,
                        "py": py,
                        "desc": group.get("description", ""),
                        "gid": group.get("id"),
                    }
                )

    if tol_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * tol_px + 1, 2 * tol_px + 1)
        )
        gt_test = [cv2.dilate(g["mask"].astype(np.uint8), kernel).astype(bool) for g in gt]
    else:
        gt_test = [g["mask"] for g in gt]

    nearest: dict[Any, float | None] = {}
    for entry in gt:
        distance = cv2.distanceTransform(1 - entry["mask"].astype(np.uint8), cv2.DIST_L2, 3)
        best: float | None = None
        for click in fg_clicks:
            if 0 <= click["py"] < height and 0 <= click["px"] < width:
                d = float(distance[click["py"], click["px"]])
                best = d if best is None else min(best, d)
        nearest[entry["id"]] = best

    hits: dict[Any, int | None] = {}
    used_clicks: set[int] = set()
    for index, entry in enumerate(gt):
        hit_click = None
        for click_index, click in enumerate(fg_clicks):
            inside = (
                0 <= click["py"] < height
                and 0 <= click["px"] < width
                and gt_test[index][click["py"], click["px"]]
            )
            if inside:
                hit_click = click_index
                used_clicks.add(click_index)
                break
        hits[entry["id"]] = hit_click

    stray = [c for i, c in enumerate(fg_clicks) if i not in used_clicks]
    n_hit = sum(1 for v in hits.values() if v is not None)

    return {
        "n_gt": len(gt),
        "n_hit": n_hit,
        "coverage": recall(n_hit, len(gt)),
        "hits": hits,
        "nearest_px": nearest,
        "cov_at": {
            tol: sum(1 for d in nearest.values() if d is not None and d <= tol)
            for tol in COVERAGE_TOLERANCES
        },
        "n_fg_clicks": len(fg_clicks),
        "n_stray": len(stray),
        "fg_clicks": fg_clicks,
        "stray": stray,
    }


def mean_or_zero(values: list[float]) -> float:
    """Mean of a possibly-empty list. Empty gives 0.0, never nan."""
    return float(np.mean(values)) if values else 0.0
