"""The post-verification repair loop.

Ported from `run_presentation_custom_flow._recover_group_batch`.

A strict verifier sometimes diagnoses a target better than the candidate picker
did -- an incomplete branch, or a mask merged into a neighbour. Rather than
discard the target, its repair click is fed back into mask generation and the
result re-verified. That is genuinely valuable: it converts rejects into
accepted masks.

BOUNDED, unlike the original. The original's only stop conditions were the
verifier accepting, the verifier omitting a repair click, or a repair click
landing within 2.5% of an earlier same-label one. Nothing bounded the round
count, and each round costs one SAM3 generation plus one verification call per
pending mask. Measured across the 2026-08-18 runs: 54 batches finished in 1
round and 9 in 2, but single batches reached 3, 4, 5, 6, 7, 8 and 9 rounds, and
the 9-round case (dense coral, pass 2) had *more* pending work in its last round
than its first -- churning rather than converging.

`max_repair_rounds` caps it. When the cap bites, the result records it so a run
that hit the ceiling is visible rather than merely expensive.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

DEFAULT_MAX_REPAIR_ROUNDS = 4
"""Covers 63 of the 69 observed batches; bounds the tail."""

MIN_REPAIR_CLICK_SEPARATION = 0.025
"""A repair click closer than this to an earlier same-label click makes no
progress, so it terminates that mask's repair chain."""


def is_actionable_repair_click(
    repair_click: Any,
    prior_clicks: list[dict[str, Any]],
    *,
    min_separation: float = MIN_REPAIR_CLICK_SEPARATION,
) -> bool:
    """Would this repair click move the mask anywhere new?"""
    if not isinstance(repair_click, dict):
        return False
    if repair_click.get("label") not in (0, 1):
        return False
    x, y = repair_click.get("x"), repair_click.get("y")
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return False
    if isinstance(x, bool) or isinstance(y, bool):
        return False

    label = int(repair_click["label"])
    for prior in prior_clicks:
        if int(prior.get("label", -1)) != label:
            continue
        distance = (
            (float(prior["x"]) - float(x)) ** 2 + (float(prior["y"]) - float(y)) ** 2
        ) ** 0.5
        if distance < min_separation:
            return False
    return True


def run_repair_rounds(
    rejected: list[dict[str, Any]],
    *,
    regenerate: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any] | None],
    verify: Callable[
        [list[dict[str, Any]], int],
        tuple[list[dict[str, Any]], list[dict[str, Any]]],
    ],
    max_repair_rounds: int = DEFAULT_MAX_REPAIR_ROUNDS,
) -> dict[str, Any]:
    """Feed verifier repair clicks back into generation until it settles.

    `regenerate(rejected, repair_click)` returns a new result, or None if it
    could not produce one. `verify(results, round_number)` returns
    `(kept, still_rejected)`.

    Returns the recovered masks, the ones that ended terminally dropped, and
    counters describing how much work it took.
    """
    recovered: list[dict[str, Any]] = []
    terminal_dropped: list[dict[str, Any]] = []
    pending = list(rejected)
    attempts = 0
    rounds_run = 0
    hit_cap = False

    while pending:
        if rounds_run >= max_repair_rounds:
            # Everything still pending is dropped, but the caller can see why.
            hit_cap = True
            terminal_dropped.extend(pending)
            break

        rounds_run += 1
        generated: list[dict[str, Any]] = []
        for item in pending:
            repair_click = item.get("mask_quality_repair_click")
            prior_clicks = [
                dict(click)
                for click in item.get("clicks_used") or []
                if isinstance(click, dict) and click.get("label") in (0, 1)
            ]
            if not is_actionable_repair_click(repair_click, prior_clicks):
                terminal_dropped.append(item)
                continue

            attempts += 1
            repaired = regenerate(item, repair_click)
            if repaired is None or not np.asarray(repaired["mask"]).astype(bool).any():
                terminal_dropped.append(item if repaired is None else repaired)
                continue

            repaired["postverify_repair_round"] = rounds_run
            repaired["postverify_repair_history"] = [
                *list(item.get("postverify_repair_history") or []),
                {"round": rounds_run, "click": dict(repair_click)},
            ]
            generated.append(repaired)

        if not generated:
            break

        kept, still_rejected = verify(generated, rounds_run)
        recovered.extend(kept)
        pending = still_rejected

    return {
        "recovered": recovered,
        "terminal_dropped": terminal_dropped,
        "attempts": attempts,
        "repaired": len(recovered),
        "rounds_run": rounds_run,
        "hit_round_cap": hit_cap,
    }
