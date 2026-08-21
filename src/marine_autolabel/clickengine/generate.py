"""Turning one click group into a mask, with the model in the loop.

Ported from `click_engine_probe.refine_group_mm`.

Each iteration runs SAM3 in multimask mode, shows the model all candidates, and
the model either picks one, adds a click, rejects the attempt, or abandons the
target. `predict` and `judge` are injected so the control flow -- which is the
part that goes wrong -- is testable without SAM3 or an API key.

`max_area_frac` rejects only DEGENERATE near-whole-frame candidates: a click on
busy substrate can make SAM3 return a frame-spanning region. The cap is loose
(0.60) on purpose, because a creature close to the camera can legitimately fill
much of the frame and area alone cannot separate the two. The content-based
verify pass is the real stray filter; this only kills the pathological case.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..geometry import clean_candidate_components, duplicate_click, smallest_valid
from .loop import click_budget_reached, iteration_indices

MAX_DUPLICATE_RETRIES = 2
DEFAULT_MAX_AREA_FRAC = 0.60


def _result(
    group: dict[str, Any],
    mask: np.ndarray,
    score: float,
    reason: str,
    status: str,
    clicks: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "creature_id": int(group.get("id", -1)),
        "description": group.get("description", ""),
        "mask": mask,
        "score": float(score),
        "area_px": int(np.asarray(mask).sum()),
        "select_reason": reason,
        "spatial_match": "click_mode",
        "clicks_used": clicks,
        "status": status,
    }


def _abandoned(group: dict[str, Any], height: int, width: int) -> dict[str, Any]:
    return _result(
        group, np.zeros((height, width), dtype=bool), 0.0, "abandoned", "abandoned", []
    )


def refine_group(
    group: dict[str, Any],
    *,
    predict: Callable[[list[dict[str, Any]]], tuple[np.ndarray, np.ndarray]],
    judge: Callable[..., dict[str, Any]],
    width: int,
    height: int,
    reseed: Callable[[int], dict[str, Any] | None] | None = None,
    max_clicks: int = 5,
    max_attempts: int = 3,
    max_area_frac: float = DEFAULT_MAX_AREA_FRAC,
    strict_quality: bool = False,
    never_abandon: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Refine one group into a mask. Returns `(result, trace)`.

    `predict(clicks)` returns `(masks, scores)`.
    `judge(masks=..., scores=..., clicks=..., attempt=..., iteration=..., budget_reached=...)`
    returns a verdict: `{"verdict": "good", "index": int}`, `{"verdict": "add",
    "click": {...}}`, `{"verdict": "reject"}` or `{"verdict": "abandon"}`.
    `reseed(attempt)` optionally supplies a fresh seed click after a rejection.

    With `strict_quality`, only a mask the model explicitly picked is returned;
    otherwise the best-scoring candidate seen is kept as a fallback.

    With `never_abandon`, an abandon verdict or an all-oversized candidate set
    still returns the best mask seen so far. The zoom generator uses this: it
    runs only on a seed already verified to be a creature, so "there is nothing
    here" is not an answer it is allowed to give.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1; got {max_attempts}")

    seed = [dict(click) for click in group["clicks"]]
    trace: list[dict[str, Any]] = []
    best: tuple[float, dict[str, Any]] | None = None
    area_cap = max_area_frac * float(height * width)

    for attempt in range(max_attempts):
        clicks = [dict(click) for click in seed]
        verdict_name: str | None = None
        duplicate_retries = 0

        for iteration in iteration_indices(max_clicks):
            masks, scores = predict(clicks)
            masks = clean_candidate_components(masks, clicks)
            areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
            valid = areas <= area_cap

            if not valid.any():
                # Every candidate spans implausibly much frame: a substrate click.
                trace.append(
                    {
                        "attempt": attempt,
                        "iteration": iteration,
                        "verdict": "no_valid_band",
                        "areas": [int(a) for a in areas],
                    }
                )
                if never_abandon and best is not None:
                    return best[1], trace
                if never_abandon:
                    index = int(np.argmax(scores))
                    return (
                        _result(group, masks[index], scores[index], "no_valid_band_best",
                                "unverified", [dict(c) for c in clicks]),
                        trace,
                    )
                return _abandoned(group, height, width), trace

            best_index = int(np.argmax(np.where(valid, scores, -np.inf)))
            if best is None or float(scores[best_index]) > best[0]:
                best = (
                    float(scores[best_index]),
                    _result(
                        group, masks[best_index], scores[best_index],
                        "mm_best_score", "exhausted", [dict(c) for c in clicks],
                    ),
                )

            budget_reached = click_budget_reached(clicks, max_clicks)
            answer = judge(
                masks=masks, scores=scores, clicks=clicks, attempt=attempt,
                iteration=iteration, budget_reached=budget_reached,
                duplicate_retry=duplicate_retries > 0,
            )
            step: dict[str, Any] = {"attempt": attempt, "iteration": iteration}
            verdict_name = str((answer or {}).get("verdict", ""))

            if verdict_name == "good":
                # The original contract names this "choice"; "index" is kept as
                # an alias for callers written against the first port.
                index = (answer or {}).get("choice", (answer or {}).get("index"))
                if not isinstance(index, int) or not (0 <= index < len(masks)) or not valid[index]:
                    # Snap to the SMALLEST plausible candidate, as the original
                    # did -- the highest score is often the over-merged blob.
                    index = smallest_valid(masks, valid)
                step["verdict"] = f"good#{index}"
                trace.append(step)
                return (
                    _result(group, masks[index], scores[index], "mllm_pick", "verified",
                            [dict(c) for c in clicks]),
                    trace,
                )

            if verdict_name == "abandon":
                step["verdict"] = "abandon"
                trace.append(step)
                if never_abandon:
                    return best[1], trace
                return _abandoned(group, height, width), trace

            if verdict_name == "reject":
                step["verdict"] = "reject"
                trace.append(step)
                break

            candidate = (answer or {}).get("click")
            if verdict_name == "add" and not budget_reached and isinstance(candidate, dict):
                if duplicate_click(clicks, candidate):
                    duplicate_retries += 1
                    step["verdict"] = "duplicate_click_retry"
                    trace.append(step)
                    if duplicate_retries >= MAX_DUPLICATE_RETRIES:
                        trace.append({"attempt": attempt, "verdict": "duplicate_click_no_progress"})
                        verdict_name = "reject"
                        break
                    continue
                clicks.append(dict(candidate))
                step["verdict"] = f"add_label{int(candidate.get('label', 1))}"
                trace.append(step)
                continue

            # No usable verdict: either out of click budget or unparseable.
            if strict_quality:
                step["verdict"] = (
                    "click_budget_reject" if budget_reached else "parse_fail_reject"
                )
                trace.append(step)
                verdict_name = "reject"
                break

            step["verdict"] = "parse_fail_band"
            trace.append(step)
            return (
                _result(group, masks[best_index], scores[best_index], "parse_fail_band",
                        "unverified", [dict(c) for c in clicks]),
                trace,
            )

        if verdict_name == "reject" and attempt + 1 < max_attempts and reseed is not None:
            fresh = reseed(attempt + 1)
            if fresh is None:
                continue
            if fresh.get("verdict") == "abandon":
                trace.append({"attempt": attempt, "verdict": "reseed_abandon"})
                return _abandoned(group, height, width), trace
            seed = [{"x": float(fresh["x"]), "y": float(fresh["y"]), "label": 1}]

    if best is not None and never_abandon:
        trace.append({"verdict": "exhausted_keep_best", "score": round(best[0], 3)})
        return best[1], trace

    if strict_quality or best is None:
        trace.append({"verdict": "exhausted_no_verified_mask"})
        return _abandoned(group, height, width), trace

    score, result = best
    trace.append({"verdict": "exhausted_keep_best", "score": round(score, 3)})
    return result, trace
