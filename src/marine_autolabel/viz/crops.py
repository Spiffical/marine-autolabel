"""Zoomed crops of a mask, for the model to judge.

Ported from `click_engine_probe._render_mask_crop`, `_render_binary_mask_crop`
and `_render_candidates`.

Every overlay here ships with a BINARY companion, and that is not redundancy: a
translucent mask lets the underlying branches show through, and a vision model
reads those visible branches as the selected pixels. The binary view --
white inside, black outside -- is the only unambiguous statement of what the
mask actually contains.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

CYAN = (255, 255, 0)  # BGR
FG_DOT = (0, 255, 0)
BG_MARK = (0, 0, 255)
TARGET_PANEL_PX = 360


def default_upscale(crop_w: int, crop_h: int) -> int:
    """Enough magnification for a small organism to fill the panel."""
    return max(2, int(round(TARGET_PANEL_PX / max(crop_w, crop_h))))


def _crop(image: np.ndarray, geom: tuple[int, int, int, int], upscale: int,
          nearest: bool = False) -> np.ndarray:
    left, top, crop_w, crop_h = geom
    patch = image[top : top + crop_h, left : left + crop_w]
    return cv2.resize(
        patch,
        (crop_w * upscale, crop_h * upscale),
        interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_CUBIC,
    )


def _draw_clicks(panel: np.ndarray, clicks: list[dict[str, Any]],
                 geom: tuple[int, int, int, int], upscale: int,
                 frame_shape: tuple[int, int]) -> None:
    left, top, _, _ = geom
    height, width = frame_shape
    for click in clicks:
        px = int((float(click["x"]) * width - left) * upscale)
        py = int((float(click["y"]) * height - top) * upscale)
        if int(click.get("label", 1)) == 1:
            cv2.circle(panel, (px, py), 7, (0, 0, 0), -1)
            cv2.circle(panel, (px, py), 5, FG_DOT, -1)
        else:
            cv2.drawMarker(panel, (px, py), BG_MARK, cv2.MARKER_TILTED_CROSS, 18, 3)


def render_mask_crop(
    frame: np.ndarray, mask: np.ndarray, clicks: list[dict[str, Any]],
    geom: tuple[int, int, int, int], path: Path, upscale: int,
) -> str:
    """Zoomed crop with the mask filled and outlined in cyan, clicks drawn."""
    panel = _crop(frame, geom, upscale)
    mask_zoom = _crop(mask.astype(np.uint8), geom, upscale, nearest=True).astype(bool)

    overlay = panel.copy()
    overlay[mask_zoom] = CYAN
    panel = cv2.addWeighted(overlay, 0.40, panel, 0.60, 0)
    contours, _ = cv2.findContours(
        mask_zoom.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(panel, contours, -1, CYAN, 2)
    _draw_clicks(panel, clicks, geom, upscale, frame.shape[:2])

    cv2.imwrite(str(path), panel)
    return str(path)


def render_binary_mask_crop(
    mask: np.ndarray, geom: tuple[int, int, int, int], path: Path, upscale: int
) -> str:
    """Exact mask membership: white inside, black outside, nothing else."""
    mask_zoom = _crop(mask.astype(np.uint8), geom, upscale, nearest=True)
    cv2.imwrite(str(path), (mask_zoom > 0).astype(np.uint8) * 255)
    return str(path)


def render_candidate_sheet(
    frame: np.ndarray, masks: np.ndarray, clicks: list[dict[str, Any]],
    geom: tuple[int, int, int, int], path: Path, upscale: int | None = None,
) -> str:
    """Side-by-side panels of every SAM3 candidate, overlay above binary.

    The model picks one by index, so panel order is the candidate order and must
    not be rearranged.
    """
    left, top, crop_w, crop_h = geom
    upscale = upscale or default_upscale(crop_w, crop_h)

    overlays, binaries = [], []
    for index in range(masks.shape[0]):
        panel = _crop(frame, geom, upscale)
        mask_zoom = _crop(masks[index].astype(np.uint8), geom, upscale, nearest=True).astype(bool)
        tinted = panel.copy()
        tinted[mask_zoom] = CYAN
        panel = cv2.addWeighted(tinted, 0.45, panel, 0.55, 0)
        _draw_clicks(panel, clicks, geom, upscale, frame.shape[:2])
        cv2.putText(panel, f"#{index}", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(panel, f"#{index}", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        overlays.append(panel)
        binaries.append(
            cv2.cvtColor((mask_zoom.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
        )

    sheet = np.vstack([np.hstack(overlays), np.hstack(binaries)])
    cv2.imwrite(str(path), sheet)
    return str(path)
