"""Rendering the standard discovery view set to disk.

Pairs with `clickengine.discovery`, which names the views and describes them to
the model. This module produces exactly those keys, so the two stay in step: if
a view is added there, `build_content` refuses the set produced here until this
module supplies it.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .overlays import (
    render_existing_masks_overlay,
    render_grid_overlay,
    render_outline_overlay,
)

LIGHT_ALPHA = 0.20
STRONG_ALPHA = 0.48


def _write(path: Path, image: np.ndarray) -> str:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"could not write view: {path}")
    return str(path)


def render_discovery_views(
    frame: np.ndarray,
    known_masks: list[dict],
    out_dir: Path,
    *,
    focus_region: tuple[float, float, float, float] | None = None,
) -> dict[str, str]:
    """Write the discovery views and return `{view_key: path}`.

    The light view carries the coordinate grid; the strong view is for judging
    whether a structure is covered at all; the outline view leaves interiors raw
    so colonies behind an accepted silhouette stay visible.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    light = render_existing_masks_overlay(frame, known_masks, alpha=LIGHT_ALPHA)
    strong = render_existing_masks_overlay(frame, known_masks, alpha=STRONG_ALPHA)
    outline = render_outline_overlay(frame, known_masks)

    views = {
        "raw": _write(out_dir / "raw.png", frame),
        "grid": _write(out_dir / "grid.png", render_grid_overlay(light)),
        "strong": _write(out_dir / "strong.png", strong),
        "outline": _write(out_dir / "outline.png", render_grid_overlay(outline)),
    }

    if focus_region is not None:
        left, right, top, bottom = focus_region
        height, width = frame.shape[:2]
        x0, x1 = int(left * width), max(int(left * width) + 1, int(right * width))
        y0, y1 = int(top * height), max(int(top * height) + 1, int(bottom * height))

        def enlarge(image: np.ndarray) -> np.ndarray:
            crop = image[y0:y1, x0:x1]
            return cv2.resize(crop, (width, height), interpolation=cv2.INTER_CUBIC)

        views["focus_raw"] = _write(out_dir / "focus_raw.png", enlarge(frame))
        views["focus_strong"] = _write(
            out_dir / "focus_strong.png", render_grid_overlay(enlarge(strong))
        )
    return views


def extract_reference_frames(
    video_path: Path, target_index: int, offsets: list[int], out_dir: Path
) -> list[str]:
    """Write nearby frames as temporal context, nearest first.

    A persistent organism is visible across these; marine snow and glare are
    not. Offsets that fall outside the clip are skipped rather than erroring --
    a frame near the start or end legitimately has fewer neighbours.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    paths: list[str] = []
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for offset in offsets:
            for index in (target_index - offset, target_index + offset):
                if index < 0 or (total > 0 and index >= total):
                    continue
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, neighbour = capture.read()
                if not ok:
                    continue
                paths.append(_write(out_dir / f"ref_{index:06d}.png", neighbour))
    finally:
        capture.release()
    return paths
