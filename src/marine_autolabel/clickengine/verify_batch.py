"""Verifying a batch of generated masks.

Ported from `click_engine_probe.verify_masks`. The per-verdict rules live in
`verify.py`; this is the batch loop that applies them and records the outcome on
each result.

The judge call is injected, so the accounting -- which fields land on a result,
how a missing confidence is handled, which bucket a mask goes to -- is testable
without an API key.

Empty masks bypass the judge entirely. There is nothing to look at, and asking
costs a call per empty mask; they are kept with confidence 0.0 and handled by
the caller, matching the original.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .verify import accept_mask_verdict, mask_quality_repair_click

KNOWN_FAILURES = {"fragment", "merge", "wrong", "background"}

CONFIDENCE_WHEN_KEPT = 0.75
CONFIDENCE_WHEN_DROPPED = 0.1
"""Coarse fallbacks for a model that omitted an explicit confidence."""


def coerce_confidence(value: Any) -> float | None:
    """Clamp a reported confidence to [0, 1], or None if it was not a number."""
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return None


def verify_masks(
    results: list[dict[str, Any]],
    *,
    judge: Callable[[dict[str, Any]], dict[str, Any]],
    strict_identity: bool = False,
    second_opinion: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split results into `(kept, dropped)`, annotating each in place.

    `judge(result)` returns the parsed answer dict for one mask.

    Each result gains `creature_confidence`, `mask_complete_identity`,
    `mask_single_identity`, `mask_quality_failure` and
    `mask_quality_repair_click`. The repair click is what feeds the bounded
    repair loop in `recovery.py`.

    `second_opinion`, when provided, is consulted before a `background`
    rejection with no repair click becomes final. A background verdict is the
    one rejection that permanently blacklists a location -- the prior-attempt
    filter skips any same-description re-proposal -- and it is the verdict
    caught being wrong on a real organism (a corner anemone two discovery
    passes and a judge had each confirmed). If the second answer keeps, the
    mask is kept with that answer's confidence, marked
    `mask_second_opinion="kept"`; if it also rejects, the drop stands, marked
    `"confirmed_reject"`. Rejections carrying a repair click skip the second
    opinion: the repair loop is the cheaper path, and by proposing a fix the
    verifier already treated the organism as real.
    """
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    for result in results:
        mask = np.asarray(result["mask"]).astype(bool)
        if not mask.any():
            result["creature_confidence"] = 0.0
            kept.append(result)
            continue

        answer = judge(result) or {}
        confidence = coerce_confidence(answer.get("confidence"))
        keep = accept_mask_verdict(answer, strict_identity=strict_identity)
        failure = str(answer.get("failure", "")).strip().lower()
        repair = mask_quality_repair_click(answer)

        result["mask_complete_identity"] = answer.get("complete_identity") is True
        result["mask_single_identity"] = answer.get("single_identity") is True
        result["mask_quality_failure"] = failure if failure in KNOWN_FAILURES else None
        result["mask_quality_repair_click"] = repair

        if (
            not keep
            and failure == "background"
            and repair is None
            and second_opinion is not None
        ):
            second = second_opinion(result) or {}
            if accept_mask_verdict(second, strict_identity=strict_identity):
                keep = True
                confidence = coerce_confidence(second.get("confidence"))
                result["mask_quality_failure"] = None
                result["mask_second_opinion"] = "kept"
            else:
                result["mask_second_opinion"] = "confirmed_reject"

        result["creature_confidence"] = (
            confidence
            if confidence is not None
            else (CONFIDENCE_WHEN_KEPT if keep else CONFIDENCE_WHEN_DROPPED)
        )
        (kept if keep else dropped).append(result)

    return kept, dropped


def filter_by_confidence(
    results: list[dict[str, Any]], minimum: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split on `creature_confidence`, returning `(above, below)`.

    A result the judge never saw has no confidence; it is treated as 0.0 and
    falls below any positive threshold.
    """
    above: list[dict[str, Any]] = []
    below: list[dict[str, Any]] = []
    for result in results:
        value = result.get("creature_confidence")
        (above if (value if value is not None else 0.0) >= minimum else below).append(result)
    return above, below
