"""Scheduling the discovery sweep across passes.

Ported from `run_presentation_custom_flow.py`. Pure policy: which region a pass
looks at, whether it audits the frame border, how many passes to run, and which
proposals are cross-pass repeats.

Two modes throughout:

  fixed sweep   `pass_count` passes tile the frame, so coverage is guaranteed
                by construction and "last pass" is knowable
  convergence   `pass_count == 0`, the model keeps going until it explicitly
                reports no missed life. A crowded scene is easy to dismiss from
                one full-frame view, so the first four passes still walk a 2x2
                tiling before the frame is reviewed as a whole
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

DEFAULT_SPATIAL_TOLERANCE = 0.025
DEFAULT_DESCRIPTION_OVERLAP = 0.60
CONVERGENCE_MINIMUM_TILES = 4


def choose_mask_guided_pass_count(
    requested: int,
    *,
    initial_known_count: int,
    sparse_extra_pass: bool,
    finder_mode: str,
) -> int:
    """Grant one extra pass when starting from nothing.

    With no first-pass masks to guide it there is no coverage to reason about,
    so the sweep gets an additional look. A dense scene where the text stage
    grounds nothing is exactly this case.
    """
    if requested <= 0:
        return 0
    if (
        sparse_extra_pass
        and initial_known_count == 0
        and finder_mode in {"mask-guided", "hybrid"}
    ):
        return requested + 1
    return requested


def should_run_border_scan(
    mode: str, *, convergence_mode: bool, pass_index: int, requested_passes: int
) -> bool:
    """Schedule the border audit without blocking agent-controlled convergence.

    In a fixed sweep `every` and `last` mean what they say. In convergence mode
    there is no knowable last pass, so either enabled mode performs a single
    dedicated edge audit on pass 1 and leaves later passes unconstrained.
    """
    if mode == "off":
        return False
    if convergence_mode:
        return pass_index == 1
    if mode == "every":
        return True
    return mode == "last" and pass_index == requested_passes


def discovery_focus_region(
    pass_index: int, pass_count: int
) -> tuple[float, float, float, float, str]:
    """Normalised `(left, right, top, bottom, label)` for one pass.

    1 pass covers the frame; 2 splits left/right; 3 splits into thirds; more
    tiles into a near-square grid. `pass_count == 0` is convergence mode.
    """
    if pass_index < 1:
        raise ValueError(f"pass_index is 1-based; got {pass_index}")

    if pass_count == 0:
        if pass_index <= CONVERGENCE_MINIMUM_TILES:
            return discovery_focus_region(pass_index, CONVERGENCE_MINIMUM_TILES)
        return 0.0, 1.0, 0.0, 1.0, "the entire residual frame"

    if pass_count <= 1:
        return 0.0, 1.0, 0.0, 1.0, "the entire frame"

    if pass_count <= 3:
        if pass_index > pass_count:
            raise ValueError(f"pass_index {pass_index} exceeds pass_count {pass_count}")
        left = (pass_index - 1) / pass_count
        right = pass_index / pass_count
        label = (
            ("left half", "right half")[pass_index - 1]
            if pass_count == 2
            else ("left third", "middle third", "right third")[pass_index - 1]
        )
        return left, right, 0.0, 1.0, label

    columns = int(np.ceil(np.sqrt(pass_count)))
    rows = int(np.ceil(pass_count / columns))
    zero_based = pass_index - 1
    row, column = zero_based // columns, zero_based % columns
    left, right = column / columns, min(1.0, (column + 1) / columns)
    top, bottom = row / rows, min(1.0, (row + 1) / rows)
    return (
        left,
        right,
        top,
        bottom,
        f"grid tile x={left:.2f}..{right:.2f}, y={top:.2f}..{bottom:.2f}",
    )


def first_positive_click(group: dict[str, Any]) -> dict[str, Any] | None:
    """The foreground seed, or None. Negative clicks only constrain a seed."""
    for click in group.get("clicks") or []:
        if int(click.get("label", -1)) == 1:
            return dict(click)
    return None


def _description_tokens(text: Any) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", str(text or "").lower()) if len(token) > 2}


def filter_prior_attempt_groups(
    groups: list[dict[str, Any]],
    prior_attempts: list[dict[str, Any]],
    *,
    spatial_tolerance: float = DEFAULT_SPATIAL_TOLERANCE,
    description_overlap_threshold: float = DEFAULT_DESCRIPTION_OVERLAP,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split proposals into `(run_now, skipped)`.

    Only a same-description, same-location reattempt is skipped. This is not a
    click budget: a new identity, a materially different click, or an earlier
    failed identity with a genuinely corrected click all still run.

    A group with no foreground seed is routed to `skipped` rather than raising.
    The parser already drops those, so it should not happen -- but a mid-run
    exception would abort an expensive fan-out over one malformed proposal, and
    a seedless group cannot produce a mask anyway.
    """
    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for group in groups:
        positive = first_positive_click(group)
        if positive is None:
            skipped.append(group)
            continue

        tokens = _description_tokens(group.get("description"))
        is_repeat = False
        for attempt in prior_attempts:
            click = attempt.get("click") or {}
            distance = float(
                np.hypot(
                    float(positive.get("x", 0.0)) - float(click.get("x", 0.0)),
                    float(positive.get("y", 0.0)) - float(click.get("y", 0.0)),
                )
            )
            if distance > spatial_tolerance:
                continue
            prior_tokens = _description_tokens(attempt.get("description"))
            smaller = min(len(tokens), len(prior_tokens))
            overlap = len(tokens & prior_tokens) / smaller if smaller else 0.0
            if overlap >= description_overlap_threshold:
                is_repeat = True
                break
        (skipped if is_repeat else kept).append(group)
    return kept, skipped


def dedup_proposals(
    groups: list[dict[str, Any]],
    width: int,
    height: int,
    *,
    px: int = 40,
    description_overlap_threshold: float = DEFAULT_DESCRIPTION_OVERLAP,
) -> tuple[list[dict[str, Any]], int]:
    """Description-aware click dedup.

    The pure 40 px geometric dedup removed two REAL organisms in two dense
    passes (a cap-and-stem 39.7 px from a whip's seed; a translucent whip near
    a branching coral) -- in crowded scenes, distinct organisms routinely stand
    closer than 40 px. So proximity alone no longer collapses two proposals:
    they must ALSO describe the same thing, judged by the same token-overlap
    rule the cross-pass repeat filter uses.

    Two markers on the same animal still collapse, because their descriptions
    agree. Surviving groups are renumbered from 1.
    """
    kept: list[dict[str, Any]] = []
    removed = 0
    for group in groups:
        seed = first_positive_click(group)
        if seed is None:
            kept.append(group)
            continue
        tokens = _description_tokens(group.get("description"))
        is_dup = False
        for other in kept:
            other_seed = first_positive_click(other)
            if other_seed is None:
                continue
            near = (
                abs(seed["x"] - other_seed["x"]) * width < px
                and abs(seed["y"] - other_seed["y"]) * height < px
            )
            if not near:
                continue
            other_tokens = _description_tokens(other.get("description"))
            smaller = min(len(tokens), len(other_tokens))
            overlap = len(tokens & other_tokens) / smaller if smaller else 1.0
            if overlap >= description_overlap_threshold:
                is_dup = True
                break
        if is_dup:
            removed += 1
        else:
            kept.append(group)
    for index, group in enumerate(kept, 1):
        group["id"] = index
    return kept, removed


def click_counts(clicks: list[dict[str, Any]]) -> dict[str, int]:
    """How many foreground and background clicks a group carries."""
    return {
        "positive": sum(int(click.get("label", -1)) == 1 for click in clicks),
        "negative": sum(int(click.get("label", -1)) == 0 for click in clicks),
    }
