"""Assembling the multi-image prompts sent to the vision model.

Ported from `som_missed_creatures.build_som_prompt_messages`. The message shape
matches what the LLM adapters expect, so the same client dispatches it unchanged.

Reference frames carry real weight here: a persistent biological subject stays
visible across nearby times, while transient artefacts -- marine snow, glare,
lighting flashes -- do not. The prompt is explicit that motion is NOT required,
because stationary animals were being rejected for holding still.
"""
from __future__ import annotations

from typing import Any


def build_som_prompt_messages(
    *,
    system_prompt: str,
    target_image_path: str,
    neighbour_image_paths: list[str],
    initial_text_prompt: str,
    num_marks: int,
) -> list[dict[str, Any]]:
    """Build the set-of-mark prompt: marked target, then reference frames."""
    if num_marks < 1:
        raise ValueError(
            f"num_marks must be >= 1; got {num_marks}. SoM has nothing to ask about."
        )

    target_blurb = (
        f"The first image is the target frame, annotated with numbered "
        f"marks 1..{num_marks} on candidate masks that SAM3 produced "
        f"and that the text-prompted agent did not already cover."
    )
    if neighbour_image_paths:
        target_blurb += (
            f" The following {len(neighbour_image_paths)} images are "
            "unmarked reference frames from times near the target. Use "
            "them as additional perspectives -- some creatures move and "
            "some are stationary; do not require motion to accept a "
            "mark. A persistent biological subject should be visible "
            "in the reference frames (perhaps with slight lighting or "
            "viewpoint shifts), while transient artefacts (floating "
            "debris, glare, lighting flashes) usually are not."
        )

    answer_instruction = (
        f"The original creature query is: '{initial_text_prompt}'. "
        "Decide which of the numbered marks correspond to real biological "
        "subjects matching that query. Respond with your reasoning in free "
        "text, then end your response with EXACTLY ONE tag of this form "
        "and nothing else after it:\n"
        '<answer>{"accepted_marks": [<int>, ...]}</answer>\n'
        f"Accepted-mark ids must be in the range 1..{num_marks}. An empty "
        "list is valid if you do not see any biological subjects."
    )

    user_content: list[dict[str, Any]] = [
        {"type": "image", "image": target_image_path},
        {"type": "text", "text": target_blurb},
    ]
    user_content.extend({"type": "image", "image": path} for path in neighbour_image_paths)
    user_content.append({"type": "text", "text": answer_instruction})

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
