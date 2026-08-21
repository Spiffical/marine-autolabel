"""The per-frame stage graph.

Replaces `run_presentation_custom_flow.process_frame`, which was ~640 lines
taking 31 keyword arguments and reading two module globals. Here the sequence is
explicit and each stage is injected, so the ordering can be tested without SAM3,
without an API key, and without a GPU.

The sequence, and why it is this order:

  1. quality screen     drop corrupted or black frames before spending anything
                        on them. rank01's chosen frames are all black and are
                        correctly dropped here.
  2. load known masks   first-pass output plus any accepted masks carried in
                        from a previous run, so residual discovery continues
                        rather than restarting.
  3. discovery passes   each pass looks at one region, proposes click groups,
                        and recovers masks from them. Accepted masks join the
                        known set, so later passes see current coverage --
                        this is what stops a pass re-finding what the previous
                        one already got.
  4. border scan        a dedicated edge audit, scheduled by `sweep`.
  5. consolidation      identity review over masks that may be one organism.

A pass that proposes nothing is the model's stop signal in convergence mode.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..clickengine.sweep import (
    dedup_proposals,
    discovery_focus_region,
    filter_prior_attempt_groups,
    first_positive_click,
    should_run_border_scan,
)

MAX_CONVERGENCE_PASSES = 12
"""Hard ceiling on convergence mode.

The model signals completion by proposing nothing, but that depends on it
behaving. Without a ceiling a model that keeps proposing marginal clicks bills
indefinitely -- the same failure the repair loop had.
"""


@dataclass
class FrameOutcome:
    """What a frame produced, and enough detail to explain it."""

    frame_id: str
    accepted: list[dict[str, Any]] = field(default_factory=list)
    skipped_reason: str | None = None
    passes: list[dict[str, Any]] = field(default_factory=list)
    hit_pass_ceiling: bool = False

    @property
    def n_accepted(self) -> int:
        return len(self.accepted)

    def summary(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "n_accepted": self.n_accepted,
            "skipped_reason": self.skipped_reason,
            "n_passes": len(self.passes),
            "hit_pass_ceiling": self.hit_pass_ceiling,
            "passes": self.passes,
        }


@dataclass
class FrameStages:
    """The model- and SAM3-dependent steps, injected so the graph is testable.

    `screen_quality` returns None to keep the frame, or a reason to drop it.
    `discover` proposes click groups for one focus region.
    `recover` turns groups into verified masks; its return follows
    `clickengine.recovery`, carrying at least `recovered`.
    `consolidate` optionally merges masks judged to be one organism.
    """

    screen_quality: Callable[[np.ndarray], str | None] | None = None
    discover: Callable[..., list[dict[str, Any]]] = lambda **_: []
    recover: Callable[..., dict[str, Any]] = lambda **_: {"recovered": []}
    consolidate: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None


def process_frame(
    frame_id: str,
    frame: np.ndarray,
    *,
    stages: FrameStages,
    known_masks: list[dict[str, Any]] | None = None,
    pass_count: int = 0,
    border_scan: str = "last",
    click_dedup_px: int = 40,
    max_convergence_passes: int = MAX_CONVERGENCE_PASSES,
) -> FrameOutcome:
    """Run the discovery sweep over one frame.

    `pass_count == 0` is convergence mode: passes continue until the model
    proposes nothing, or `max_convergence_passes` is reached.
    """
    outcome = FrameOutcome(frame_id=frame_id)

    if stages.screen_quality is not None:
        reason = stages.screen_quality(frame)
        if reason:
            outcome.skipped_reason = reason
            return outcome

    height, width = frame.shape[:2]
    accepted = list(known_masks or [])
    prior_attempts: list[dict[str, Any]] = []

    convergence = pass_count == 0
    ceiling = max_convergence_passes if convergence else pass_count

    pass_index = 0
    while pass_index < ceiling:
        pass_index += 1
        region = discovery_focus_region(pass_index, pass_count)
        border = should_run_border_scan(
            border_scan,
            convergence_mode=convergence,
            pass_index=pass_index,
            requested_passes=pass_count,
        )

        proposed = stages.discover(
            frame=frame,
            known_masks=accepted,
            region=region,
            pass_index=pass_index,
            border_scan=border,
        )

        record: dict[str, Any] = {
            "pass_index": pass_index,
            "region": region[4],
            "border_scan": border,
            "n_proposed": len(proposed),
        }

        if not proposed:
            # In convergence mode this is the model's explicit stop signal.
            record["stopped"] = convergence
            outcome.passes.append(record)
            if convergence:
                break
            continue

        fresh, skipped = filter_prior_attempt_groups(proposed, prior_attempts)
        deduped, dedup_removed = dedup_proposals(fresh, width, height, px=click_dedup_px)
        record.update(
            n_repeat_skipped=len(skipped), n_dedup_removed=dedup_removed, n_run=len(deduped)
        )

        for group in deduped:
            seed = first_positive_click(group)
            if seed is not None:
                prior_attempts.append(
                    {"click": seed, "description": group.get("description", "")}
                )

        if deduped:
            batch = stages.recover(groups=deduped, known_masks=accepted, pass_index=pass_index)
            recovered = list(batch.get("recovered") or [])
            accepted.extend(recovered)
            record["n_recovered"] = len(recovered)
            # Carry the attrition breakdown, so a pass that proposed a lot and
            # kept a little can be diagnosed without re-running it. Without
            # these, "10 proposed, 4 recovered" gives no way to tell whether
            # generation, verification, NMS or the confidence floor took them.
            record.update(
                {
                    "n_generated": batch.get("n_generated"),
                    "n_verify_kept": batch.get("n_verify_kept"),
                    "n_verify_dropped": batch.get("n_verify_dropped"),
                    "n_repair_recovered": batch.get("n_repair_recovered"),
                    "n_nms_removed": batch.get("mask_nms_removed"),
                    "n_low_confidence": len(batch.get("low_confidence") or []),
                    "repair_rounds": batch.get("repair_rounds"),
                    "rejection_reasons": batch.get("rejection_reasons"),
                    "hit_round_cap": bool(batch.get("hit_round_cap")),
                }
            )
        else:
            record["n_recovered"] = 0

        outcome.passes.append(record)

    if convergence and pass_index >= ceiling:
        outcome.hit_pass_ceiling = True

    if stages.consolidate is not None:
        accepted = stages.consolidate(accepted)

    outcome.accepted = accepted
    return outcome
