"""Control rules for the click refinement loop.

Ported from `click_engine_probe.py`. Small functions, but each encodes a
failure this pipeline actually hit.
"""
from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import Any

from ..geometry import merge_corrected_positive_click

MAX_CORRECTION_DISPLACEMENT = 0.10
"""Normalised full-frame distance a localisation correction may move a seed."""


def iteration_indices(max_clicks: int) -> Iterable[int]:
    """Refiner iterations. `max_clicks <= 0` means the agent decides when to stop."""
    if max_clicks <= 0:
        return itertools.count()
    return range(max_clicks)


def click_budget_reached(clicks: list[Any], max_clicks: int) -> bool:
    """Has a positive, finite click ceiling been reached?"""
    return max_clicks > 0 and len(clicks) >= max_clicks


REASONING_MODEL_MINIMUM = 4096
"""Completion floor for models that reason before emitting the answer tag."""

_REASONING_MODEL_MARKERS = ("sonnet-5", "opus-5", "fable-5")


def is_reasoning_model(model: str) -> bool:
    """Does this model spend completion budget on reasoning before answering?

    Matches the Claude 5 family. The original listed only sonnet-5 and opus-5,
    which left fable-5 unfloored even while it was the click model.
    """
    normalized = str(model).lower()
    return any(marker in normalized for marker in _REASONING_MODEL_MARKERS)


def response_token_budget(model: str, default: int, *, minimum: int | None = None) -> int:
    """Floor the completion budget for models that reason before answering.

    A Claude 5 model can spend a small budget entirely on internal reasoning and
    return either no text block or a JSON tag truncated mid-object. The parsers
    are lenient by design, so a truncated tag reads as "nothing found" -- the
    failure presents as poor recall rather than an error, which is why it needs
    to be prevented here and detected in the transport.

    Older models keep their cheaper caps.
    """
    floor = REASONING_MODEL_MINIMUM if minimum is None else int(minimum)
    return max(int(default), floor) if is_reasoning_model(model) else int(default)


def bounded_corrected_positive_click(
    clicks: list[dict[str, Any]],
    corrected: dict[str, Any],
    *,
    max_displacement: float = MAX_CORRECTION_DISPLACEMENT,
) -> tuple[list[dict[str, Any]], float | None, bool]:
    """Apply a localisation correction only if it stays near its seed.

    The localiser may nudge a slightly misplaced foreground point onto a solid
    part of the same target. A large move is far more likely to have jumped to a
    *different* organism, which in a dense scene silently changes which animal is
    being segmented. Beyond `max_displacement` the correction is refused and the
    discovery agent's original clicks are kept.

    Returns `(clicks, displacement, applied)`; displacement is measured from the
    nearest positive seed in normalised full-frame coordinates.
    """
    positives = [click for click in clicks if int(click.get("label", 1)) == 1]
    if not positives:
        return [dict(corrected)] + [dict(click) for click in clicks], None, True

    displacement = min(
        (
            (float(click["x"]) - float(corrected["x"])) ** 2
            + (float(click["y"]) - float(corrected["y"])) ** 2
        )
        ** 0.5
        for click in positives
    )
    if displacement > max_displacement:
        return [dict(click) for click in clicks], displacement, False
    return merge_corrected_positive_click(clicks, corrected), displacement, True
