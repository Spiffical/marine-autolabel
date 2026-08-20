"""Segmenting on a zoomed crop, mapped back to full-frame coordinates.

Ported from `click_engine_probe._sam3_raw_crop`.

A small or thin organism occupies too few pixels at full frame for SAM3 to
resolve; cropping around the seed and upscaling makes it fill the field of view.
The masks come back in crop space and must be mapped home, which is the part
that is easy to get subtly wrong.

Downscaling uses INTER_AREA with a 0.5 threshold rather than INTER_NEAREST:
area-averaging preserves the edges of fine branches, which nearest-neighbour
erodes away -- and fine branches are exactly what this path exists to capture.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

PredictFn = Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def crop_and_upscale(
    frame: np.ndarray, geom: tuple[int, int, int, int], upscale: int
) -> np.ndarray:
    left, top, crop_w, crop_h = geom
    patch = frame[top : top + crop_h, left : left + crop_w]
    return cv2.resize(patch, (crop_w * upscale, crop_h * upscale), interpolation=cv2.INTER_CUBIC)


def clicks_to_crop_pixels(
    clicks: list[dict[str, Any]],
    geom: tuple[int, int, int, int],
    upscale: int,
    frame_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Full-frame normalised clicks -> pixel coordinates in the upscaled crop."""
    left, top, _, _ = geom
    height, width = frame_shape
    coords = np.array(
        [
            [(float(c["x"]) * width - left) * upscale, (float(c["y"]) * height - top) * upscale]
            for c in clicks
        ],
        dtype=np.float32,
    )
    labels = np.array([int(c["label"]) for c in clicks], dtype=np.int64)
    return coords, labels


def masks_to_full_frame(
    masks: np.ndarray, geom: tuple[int, int, int, int], frame_shape: tuple[int, int]
) -> np.ndarray:
    """Map crop-space masks back into full-frame canvases."""
    left, top, crop_w, crop_h = geom
    height, width = frame_shape
    full = np.zeros((masks.shape[0], height, width), dtype=bool)
    for index in range(masks.shape[0]):
        small = (
            cv2.resize(
                masks[index].astype(np.float32), (crop_w, crop_h), interpolation=cv2.INTER_AREA
            )
            >= 0.5
        )
        full[index, top : top + crop_h, left : left + crop_w] = small
    return full


def predict_on_crop(
    frame: np.ndarray,
    clicks: list[dict[str, Any]],
    geom: tuple[int, int, int, int],
    upscale: int,
    predict: PredictFn,
) -> tuple[np.ndarray, np.ndarray]:
    """Run `predict` on an upscaled crop; return full-frame masks and scores."""
    zoomed = crop_and_upscale(frame, geom, upscale)
    coords, labels = clicks_to_crop_pixels(clicks, geom, upscale, frame.shape[:2])
    masks, scores = predict(zoomed, coords, labels)
    return masks_to_full_frame(np.asarray(masks).astype(bool), geom, frame.shape[:2]), scores
