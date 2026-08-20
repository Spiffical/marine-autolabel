"""Interpreting the post-mask verification verdict.

Ported from `click_engine_probe._accept_mask_verdict` and
`_mask_quality_repair_click`. Both are pure: the transport that obtains the
verdict lives elsewhere.

The verifier grades a produced mask rather than a click, and its job is to split
"stray" masks -- ones that match nothing in the first pass -- into genuinely new
organisms and false positives. Most strays turn out to be real missed creatures,
so the default acceptance rule is deliberately permissive.
"""
from __future__ import annotations

from typing import Any

REPAIRABLE_FAILURES = {"fragment", "merge", "background"}
"""`wrong` is absent on purpose: a mask on the wrong object cannot be repaired
by one click, it has to be discarded."""


def accept_mask_verdict(answer: dict[str, Any], *, strict_identity: bool = False) -> bool:
    """Should this mask be kept?

    The default rule is `keep is not False` -- a missing or malformed `keep`
    counts as acceptance. That asymmetry is deliberate: ground truth is
    incomplete, so a mask the verifier failed to judge is more likely a real
    organism than a false positive, and dropping it costs recall.

    `strict_identity` additionally demands that the verifier positively
    confirmed the mask covers one whole organism and only one.
    """
    if strict_identity:
        return (
            answer.get("keep") is True
            and answer.get("complete_identity") is True
            and answer.get("single_identity") is True
        )
    return answer.get("keep") is not False


def mask_quality_repair_click(answer: dict[str, Any]) -> dict[str, Any] | None:
    """A validated repair click for a rejected mask, or None.

    The click's label must agree with the stated failure: a `fragment` needs a
    positive click on the missed continuation, while `merge` and `background`
    need a negative click inside the wrongly included pixels. A verdict whose
    label contradicts its own failure mode is discarded rather than trusted,
    since acting on it would make the mask worse.

    Coordinates are normalised to the full frame, never to a crop.
    """
    failure = str(answer.get("failure", "")).strip().lower()
    if failure not in REPAIRABLE_FAILURES:
        return None

    click = answer.get("repair_click") or {}
    expected_label = 1 if failure == "fragment" else 0
    if (
        not isinstance(click.get("x"), (int, float))
        or not isinstance(click.get("y"), (int, float))
        or int(click.get("label", -1)) != expected_label
        or not 0.0 <= float(click["x"]) <= 1.0
        or not 0.0 <= float(click["y"]) <= 1.0
    ):
        return None
    return {"x": float(click["x"]), "y": float(click["y"]), "label": expected_label}
