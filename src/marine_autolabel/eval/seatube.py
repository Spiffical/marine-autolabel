"""Matching SeaTube annotations to segmentation masks.

Ported from `scripts/match_seatube_annotations.py`. These annotations are the
only EXTERNAL ground truth in the project -- everything else scores against the
pipeline's own first pass, which is incomplete.

Two constraints hold the whole thing up:

**The matcher never invents taxonomy.** The model sees only the WoRMS
annotations already attached to that clip and must answer with their ids. An
id outside that set is discarded, not looked up. A taxonomic label is a claim
about the world, and this pipeline is not entitled to make one it was not given.

**Nothing is assigned on a single opinion.** Repeats are reduced by majority
consensus: a label needs support from more than half the configured repeats AND
a mean confidence clearing a floor. An object with no consensus stays unmatched,
which is an honest outcome rather than a gap to be filled.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

DEFAULT_MIN_MEAN_CONFIDENCE = 0.55


def annotation_name(annotation: dict[str, Any]) -> str:
    return str(annotation.get("taxon") or annotation.get("name") or "").strip()


def int_list(value: Any) -> list[int]:
    """Coerce a scalar, list, or list of digit-strings into ints."""
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [value]
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            out.append(item)
        elif isinstance(item, str) and item.isdigit():
            out.append(int(item))
    return out


def normalize_response(
    parsed: dict[str, Any] | None,
    *,
    object_count: int,
    annotations: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Validate one match response against the ids it was actually offered.

    Object ids outside 1..object_count and annotation ids outside the clip's own
    annotation set are dropped. A match left with no valid ids on either side is
    discarded entirely rather than half-kept.
    """
    if not isinstance(parsed, dict):
        return None

    allowed = {int(item["annotation_id"]): item for item in annotations}
    matches: list[dict[str, Any]] = []

    for raw in parsed.get("matches", []) or []:
        if not isinstance(raw, dict):
            continue
        object_ids = sorted(
            {
                value
                for value in int_list(raw.get("object_ids", raw.get("object_id")))
                if 1 <= value <= object_count
            }
        )
        annotation_ids = sorted(
            {
                value
                for value in int_list(raw.get("annotation_ids", raw.get("annotation_id")))
                if value in allowed
            }
        )
        if not object_ids or not annotation_ids:
            continue

        try:
            confidence = min(1.0, max(0.0, float(raw.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0

        matches.append(
            {
                "object_ids": object_ids,
                "annotation_ids": annotation_ids,
                "taxa": sorted({annotation_name(allowed[value]) for value in annotation_ids}),
                "confidence": confidence,
                "reason": str(raw.get("reason", "")),
            }
        )

    return {
        "matches": matches,
        "unmatched_object_ids": sorted(
            {
                value
                for value in int_list(parsed.get("unmatched_object_ids"))
                if 1 <= value <= object_count
            }
        ),
        "unmatched_annotation_ids": sorted(
            {
                value
                for value in int_list(parsed.get("unmatched_annotation_ids"))
                if value in allowed
            }
        ),
        "notes": [str(value) for value in parsed.get("notes", []) or [] if str(value).strip()],
    }


def response_object_choices(response: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """One choice per object: the highest-confidence match naming it."""
    choices: dict[int, dict[str, Any]] = {}
    for match in response.get("matches", []):
        for object_id in match["object_ids"]:
            if object_id not in choices or match["confidence"] > choices[object_id]["confidence"]:
                choices[object_id] = match
    return choices


def build_consensus(
    responses: list[dict[str, Any]],
    *,
    object_count: int,
    annotations: list[dict[str, Any]],
    configured_repeats: int,
    min_mean_confidence: float = DEFAULT_MIN_MEAN_CONFIDENCE,
) -> dict[str, Any]:
    """Reduce independent repeats to assignments by majority.

    A label is assigned only when it is supported by a strict majority of the
    CONFIGURED repeats -- not of the responses actually received. That matters:
    if two of three calls failed, the surviving one cannot carry a majority on
    its own, so an unreliable run yields no assignments rather than confident
    ones drawn from a single opinion.
    """
    min_support = configured_repeats // 2 + 1

    per_object: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for response in responses:
        for object_id, choice in response_object_choices(response).items():
            per_object[object_id].append(choice)

    assignments: list[dict[str, Any]] = []
    used_annotations: set[int] = set()

    for object_id in range(1, object_count + 1):
        grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in per_object.get(object_id, []):
            grouped[tuple(row["taxa"])].append(row)
        if not grouped:
            continue

        # Most votes wins; mean confidence breaks ties.
        taxa, supporting = sorted(
            grouped.items(),
            key=lambda item: (
                len(item[1]),
                statistics.fmean(row["confidence"] for row in item[1]),
            ),
            reverse=True,
        )[0]

        if len(supporting) < min_support:
            continue
        confidences = [row["confidence"] for row in supporting]
        confidence_mean = statistics.fmean(confidences)
        if confidence_mean < min_mean_confidence:
            continue

        annotation_ids = sorted(
            {aid for row in supporting for aid in row["annotation_ids"]}
        )
        used_annotations.update(annotation_ids)
        assignments.append(
            {
                "object_id": object_id,
                "label": " / ".join(taxa),
                "taxa": list(taxa),
                "annotation_ids": annotation_ids,
                "support": len(supporting),
                "configured_repeats": configured_repeats,
                "valid_repeats": len(responses),
                "confidence_mean": confidence_mean,
                "confidence_std": (
                    statistics.stdev(confidences) if len(confidences) > 1 else 0.0
                ),
                "reasons": [row["reason"] for row in supporting if row["reason"]],
            }
        )

    assigned = {item["object_id"] for item in assignments}
    all_annotation_ids = {int(item["annotation_id"]) for item in annotations}
    return {
        "consensus_threshold": min_support,
        "min_consensus_confidence": min_mean_confidence,
        "valid_response_count": len(responses),
        "assignments": assignments,
        "unmatched_object_ids": sorted(set(range(1, object_count + 1)) - assigned),
        "unmatched_annotation_ids": sorted(all_annotation_ids - used_annotations),
    }


def compress_id_ranges(values: list[int]) -> str:
    """`[1,2,3,7,9,10]` -> `"1-3, 7, 9-10"`, for readable reports."""
    ordered = sorted(set(values))
    if not ordered:
        return ""
    spans: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        spans.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    spans.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(spans)
