"""Crop geometry for zoomed views.

Several stages show the model a zoomed crop rather than the full frame: mask
verification, click localisation, and the zoom fallback generator. All of them
need the same window calculation, ported here from
`click_engine_probe._mask_crop_geom`.

Coordinates returned are pixels in the full frame. Clicks passed in are
normalised to the full frame, as everywhere else in the pipeline.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def mask_crop_geom(
    mask: np.ndarray,
    clicks: list[dict[str, Any]],
    width: int,
    height: int,
    region_frac: float,
) -> tuple[int, int, int, int]:
    """Crop window `(left, top, crop_w, crop_h)` around a mask, or its clicks.

    The window contains the whole mask with padding (1.8x its bounding box) and
    is never smaller than `region_frac` of the frame, so a tiny organism still
    gets enough surrounding context to judge. When the mask is empty the window
    centres on the mean click position instead, falling back to frame centre.

    The result is always clamped inside the frame.
    """
    if mask.any():
        ys, xs = np.where(mask)
        bx0, bx1 = int(xs.min()), int(xs.max())
        by0, by1 = int(ys.min()), int(ys.max())
        cx, cy = (bx0 + bx1) // 2, (by0 + by1) // 2
        crop_w = int(max(width * region_frac, (bx1 - bx0) * 1.8))
        crop_h = int(max(height * region_frac, (by1 - by0) * 1.8))
    else:
        points = [(c["x"] * width, c["y"] * height) for c in clicks] or [(width / 2, height / 2)]
        cx = int(sum(p[0] for p in points) / len(points))
        cy = int(sum(p[1] for p in points) / len(points))
        crop_w, crop_h = int(width * region_frac), int(height * region_frac)

    crop_w, crop_h = min(crop_w, width), min(crop_h, height)
    left = max(0, min(width - crop_w, cx - crop_w // 2))
    top = max(0, min(height - crop_h, cy - crop_h // 2))
    return left, top, crop_w, crop_h
