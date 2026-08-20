"""SAM3 text proposals, and choosing which phrases to try.

Ported from `run_presentation_custom_flow._adaptive_prompt_planner_prompt`,
`_select_adaptive_prompt_specs` and `_propose_text_masks`.

Phrases here are RETRIEVAL HANDLES, not taxonomic labels: their only job is to
get SAM3 to ground an instance mask. That distinction matters because the
temptation is to treat a selected phrase as an identification, and it is not one.

Selection is closed-vocabulary on purpose. The model may only return phrases
from the frame's own candidate list, matched case- and whitespace-insensitively;
anything else is discarded rather than passed to SAM3. A hallucinated phrase
would otherwise become a silent, unfindable query.

Selecting NOTHING is valid. Click recovery handles life that no phrase can
ground, so an empty selection is a legitimate answer rather than a failure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromptSpec:
    """One text query for SAM3."""

    text: str
    group: str = "adaptive_scene"


def _normalise(phrase: str) -> str:
    return " ".join(str(phrase).split())


def build_planner_prompt(
    frame_id: str, visual_note: str, candidates: list[str]
) -> str:
    """Ask the model which of this frame's candidate phrases are worth trying."""
    candidate_lines = "\n".join(f"- {phrase}" for phrase in candidates)
    return (
        "You are planning text queries for SAM3 on one underwater TARGET FRAME. "
        "Choose only phrases whose visual concept is credibly present and useful "
        "for proposing instance masks in this exact frame. Phrases are retrieval "
        "handles only, never taxonomic labels. Prefer the shortest complementary "
        "set; omit redundant, visually mismatched, or overly generic phrases. "
        "In particular, use 'small creatures' only when discrete small animal "
        "bodies are actually visible. Do not use it merely because a scene has "
        "fine coral branches, texture, or many sessile colonies. Dense coral, "
        "sea-fan, sea-whip, or sponge scenes should use the matching morphology "
        "or plausible provided taxon handle instead. On-screen logos, lettering, "
        "timestamps, and video/UI overlays are never targets. Click recovery will "
        "handle visible life that none of these phrases can ground, so it is valid "
        "to select no phrase.\n\n"
        f"Frame id: {frame_id}\n"
        f"Scene note: {visual_note or 'none'}\n"
        "Allowed candidates (copy selected strings exactly):\n"
        f"{candidate_lines or '- none'}\n\n"
        "Output brief reasoning followed by EXACTLY ONE trailing tag:\n"
        '<answer>{"phrases":["<exact allowed candidate>"]}</answer>'
    )


def select_prompt_specs(candidates: list[str], answer: dict[str, Any]) -> list[PromptSpec]:
    """Keep only phrases that appear in `candidates`, in the model's order.

    Matching ignores case and collapses whitespace, so a reply that differs only
    in formatting still counts. Duplicates are dropped, since running the same
    query twice costs a SAM3 pass for nothing.
    """
    allowed = {
        _normalise(c).casefold(): _normalise(c) for c in candidates if _normalise(c)
    }
    raw = answer.get("phrases") if isinstance(answer, dict) else None
    if not isinstance(raw, list):
        return []

    selected: list[PromptSpec] = []
    seen: set[str] = set()
    for value in raw:
        if not isinstance(value, str):
            continue
        key = _normalise(value).casefold()
        if key in allowed and key not in seen:
            selected.append(PromptSpec(text=allowed[key]))
            seen.add(key)
    return selected


def compact_prompt_specs(candidates: list[str]) -> list[PromptSpec]:
    """The non-adaptive mode: try every candidate the manifest supplies."""
    seen: set[str] = set()
    specs: list[PromptSpec] = []
    for candidate in candidates:
        text = _normalise(candidate)
        key = text.casefold()
        if text and key not in seen:
            specs.append(PromptSpec(text=text, group="compact"))
            seen.add(key)
    return specs


def in_exclusion_region(
    box_xyxy: tuple[float, float, float, float],
    regions: list[list[float]],
    *,
    overlap_threshold: float = 0.5,
) -> bool:
    """Is this proposal mostly inside an excluded region of the frame?

    Manifests mark overlay areas -- the NOAA logo, the timestamp banner -- which
    SAM3 will happily ground a phrase onto. Coordinates are normalised xyxy.
    """
    x0, y0, x1, y1 = box_xyxy
    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if area <= 0:
        return False
    for region in regions:
        rx0, ry0, rx1, ry1 = region
        inter = max(0.0, min(x1, rx1) - max(x0, rx0)) * max(0.0, min(y1, ry1) - max(y0, ry0))
        if inter / area >= overlap_threshold:
            return True
    return False
