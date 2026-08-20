"""Turning a click group into a mask.

Three generators, ported from `click_engine_probe.py`:

  full-frame  tight masks, IoU(det) ~0.91, but abandons thin or low-contrast
              organisms such as brittle-stars
  zoom        recovers those, at roughly 0.07 IoU(det) cost on every mask
  hybrid      full-frame first, zoom only when it abandons or returns a
              degenerate mask -- the recommended default

`hybrid_policy` takes the two generators as arguments so the decision rule can
be tested without SAM3. Changing the generator materially changes results, so
re-evaluate at repeats >= 3 before swapping it.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

DEFAULT_MIN_AREA_PX = 200
"""Below this a mask is degenerate: SAM3 returned a speck, not an organism."""


class Generator(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Return `(result, trace)`."""


def mask_area(result: dict[str, Any]) -> int:
    return int(np.asarray(result["mask"]).sum())


def is_degenerate(result: dict[str, Any], min_area_px: int = DEFAULT_MIN_AREA_PX) -> bool:
    """A result the full-frame pass failed to produce a usable mask for."""
    return result.get("status") == "abandoned" or mask_area(result) < min_area_px


def hybrid_policy(
    full_frame: Callable[[], tuple[dict[str, Any], list[dict[str, Any]]]],
    zoom: Callable[[], tuple[dict[str, Any], list[dict[str, Any]]]],
    *,
    min_area_px: int = DEFAULT_MIN_AREA_PX,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Full-frame first; fall back to zoom only when it fails.

    The zoom result is taken only if it clears `min_area_px` itself -- otherwise
    the full-frame result is returned even though it was degenerate, because a
    degenerate zoom result is no better and the full-frame trace is the more
    informative one to keep.

    Returns `(result, trace)`. On recovery the result's `select_reason` is
    prefixed `hybrid_zoom/` and the trace records why the fallback fired.
    """
    result, trace = full_frame()
    area = mask_area(result)
    if not is_degenerate(result, min_area_px):
        return result, trace

    zoom_result, zoom_trace = zoom()
    if mask_area(zoom_result) < min_area_px:
        return result, trace

    zoom_result["select_reason"] = "hybrid_zoom/" + zoom_result.get("select_reason", "")
    return zoom_result, trace + [{"hybrid": "zoom_recover", "fullframe_area": area}] + zoom_trace
