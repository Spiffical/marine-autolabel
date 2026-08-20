"""Overlays shown to the vision model.

Ported from `som_missed_creatures.py` and the outline view added to
`run_presentation_custom_flow.py` on 2026-08-18.

These are not debug art -- they are the model's input, and what they make
visible determines what gets found. The three coverage views exist because each
hides something the others show:

  filled    accepted masks as translucent green. Reads coverage at a glance,
            but in a crowded thicket the filled inter-branch space of one
            silhouette swallows the colonies visible behind it.
  strong    the same at higher alpha, for judging whether a specific organism
            is covered at all.
  outline   contours only, raw interior left visible. The decisive view for
            branching scenes: a separate colony seen through an accepted
            outer silhouette is NOT segmented, and only this view shows it.

All click coordinates are normalised to the full frame.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

GREEN_FILL = np.array([0, 255, 0], dtype=np.float32)  # BGR
GREEN_OUTLINE = (0, 200, 0)
GRID_LINE = (200, 200, 200)

CLICK_PALETTE = [
    (0, 0, 255),    # red
    (255, 200, 0),  # cyan-ish
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (0, 200, 255),  # orange
    (200, 0, 255),  # purple
]


def render_existing_masks_overlay(
    frame_bgr: np.ndarray, existing_masks: list[dict[str, Any]], alpha: float = 0.4
) -> np.ndarray:
    """Accepted masks as translucent green fill with a crisp outline."""
    out = frame_bgr.copy()
    for entry in existing_masks:
        mask = np.asarray(entry["mask"], dtype=bool)
        if not mask.any():
            continue
        out_f = out.astype(np.float32)
        out_f[mask] = (1 - alpha) * out_f[mask] + alpha * GREEN_FILL
        out = out_f.astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, GREEN_OUTLINE, 1)
    return out


def render_outline_overlay(
    frame_bgr: np.ndarray, existing_masks: list[dict[str, Any]], thickness: int = 3
) -> np.ndarray:
    """Accepted masks as contours only, leaving the raw scene visible inside.

    Added 2026-08-18 for dense coral: with filled overlays, a colony visible
    through the inter-branch space of an accepted silhouette looks covered when
    it is not. Outlines make the distinction legible.
    """
    out = frame_bgr.copy()
    for entry in existing_masks:
        mask = np.asarray(entry["mask"], dtype=bool)
        if not mask.any():
            continue
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, (0, 255, 0), thickness, cv2.LINE_AA)
    return out


def render_grid_overlay(frame_bgr: np.ndarray, *, grid_n: int = 10) -> np.ndarray:
    """An N x N grid with `row,col` labels at each cell's top-left corner.

    Gives the model coarse coordinates that map back to pixels reliably, which
    is what the `{"cell": [row, col]}` click alternative refers to. Labels are
    drawn twice, dark then light, so they stay readable over any background.
    """
    out = frame_bgr.copy()
    height, width = out.shape[:2]
    step_y, step_x = height / grid_n, width / grid_n

    for col in range(1, grid_n):
        x = int(round(col * step_x))
        cv2.line(out, (x, 0), (x, height), GRID_LINE, 1, cv2.LINE_AA)
    for row in range(1, grid_n):
        y = int(round(row * step_y))
        cv2.line(out, (0, y), (width, y), GRID_LINE, 1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for row in range(grid_n):
        for col in range(grid_n):
            origin = (int(round(col * step_x)) + 4, int(round(row * step_y)) + 14)
            label = f"{row},{col}"
            cv2.putText(out, label, origin, font, 0.36, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(out, label, origin, font, 0.36, (240, 240, 240), 1, cv2.LINE_AA)
    return out


def render_proposed_click_groups_overlay(
    frame_bgr: np.ndarray,
    groups: list[dict[str, Any]],
    existing_masks: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Proposed click groups, colour-coded per creature id.

    Positive clicks are a circle with a white centre dot, the familiar SAM
    positive marker; negative clicks are a tilted cross. Each is labelled
    `<creature_id>.<click_index>` so a reviewer can tie clicks to creatures.
    """
    out = (
        frame_bgr.copy()
        if existing_masks is None
        else render_existing_masks_overlay(frame_bgr, existing_masks, alpha=0.2)
    )
    height, width = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    for group in groups:
        creature_id = int(group.get("id", -1))
        colour = (
            CLICK_PALETTE[(creature_id - 1) % len(CLICK_PALETTE)]
            if creature_id > 0
            else (200, 200, 200)
        )
        for click_index, click in enumerate(group.get("clicks") or [], start=1):
            # NB: scaled by width, not width - 1, unlike geometry.norm_to_pixel.
            # A one-pixel difference that only affects where a marker is drawn.
            cx = int(round(click["x"] * width))
            cy = int(round(click["y"] * height))
            label = f"{creature_id}.{click_index}"
            if int(click.get("label", 1)) == 1:
                cv2.circle(out, (cx, cy), 12, colour, 2)
                cv2.circle(out, (cx, cy), 2, (255, 255, 255), -1)
            else:
                cv2.drawMarker(out, (cx, cy), colour, cv2.MARKER_TILTED_CROSS, 20, 2)
            cv2.putText(out, label, (cx + 14, cy + 6), font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, label, (cx + 14, cy + 6), font, 0.5, colour, 1, cv2.LINE_AA)
    return out
