"""Consolidating masks that turn out to be one organism.

Ported from `run_presentation_custom_flow._consolidate_ambiguous_overlaps`.

`overlaps.ambiguous_overlap_components` only *nominates* groups: geometry cannot
tell a coral segmented into three pieces from three interlaced colonies. So a
nominated component goes through two independent gates before anything merges:

  1. an identity partition, which may split a geometric component into several
     same-identity subgroups, or none at all
  2. a strict verification of the UNION mask, which must pass the same
     complete-and-single-identity bar as any other accepted mask

Either gate failing preserves the originals. That asymmetry is deliberate: a
wrong merge destroys two correct masks and is invisible afterwards, while a
missed merge leaves two masks that are each individually defensible.

The lowest index in a subgroup becomes the anchor and carries the union; the
rest are removed. Anchoring on position keeps the result deterministic.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .overlaps import ambiguous_overlap_components


def consolidate_masks(
    known_masks: list[dict[str, Any]],
    *,
    partition: Callable[[list[int], list[list[int]]], list[list[int]]],
    verify_union: Callable[[dict[str, Any], list[int]], bool],
    containment_threshold: float = 0.40,
    max_gap_fraction: float = 0.006,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge only subgroups that pass both gates. Returns `(masks, audit)`.

    `partition(candidate_indices, components)` returns same-identity subgroups as
    lists of indices into `known_masks`. `verify_union(union_result, subgroup)`
    returns whether the merged mask is acceptable.
    """
    masks = [np.asarray(item["mask"]).astype(bool) for item in known_masks]
    components, pairs = ambiguous_overlap_components(
        masks,
        containment_threshold=containment_threshold,
        max_gap_fraction=max_gap_fraction,
    )

    audit: dict[str, Any] = {
        "candidate_pairs": pairs,
        "components": components,
        "same_identity_groups": [],
        "union_attempts": [],
        "n_consolidated_components": 0,
        "n_masks_removed": 0,
    }
    if not components:
        return known_masks, audit

    candidate_indices = sorted({index for group in components for index in group})
    subgroups = [sorted(set(g)) for g in partition(candidate_indices, components) if len(g) > 1]
    audit["same_identity_groups"] = subgroups

    grouped = {index for group in subgroups for index in group}
    for component in components:
        if not set(component).intersection(grouped):
            audit["union_attempts"].append(
                {"component": component, "union_verified": False, "reason": "no_same_identity"}
            )

    replacements: dict[int, dict[str, Any]] = {}
    removed: set[int] = set()

    for subgroup in subgroups:
        union_mask = np.logical_or.reduce([masks[index] for index in subgroup])
        union_result = {
            "creature_id": subgroup[0] + 1,
            "description": "candidate union of temporally reviewed overlapping masks",
            "mask": union_mask,
            "source": "overlap_identity_union",
            "clicks_used": [],
        }
        verified = bool(verify_union(union_result, subgroup))
        audit["union_attempts"].append(
            {"component": subgroup, "union_verified": verified,
             "union_area_px": int(union_mask.sum())}
        )
        if not verified:
            continue

        anchor = subgroup[0]
        consolidated = dict(known_masks[anchor])
        consolidated["mask"] = union_mask
        consolidated["source"] = "overlap_identity_union"
        consolidated["consolidated_from"] = list(subgroup)
        replacements[anchor] = consolidated
        removed.update(index for index in subgroup if index != anchor)

    audit["n_consolidated_components"] = len(replacements)
    audit["n_masks_removed"] = len(removed)

    return (
        [
            replacements.get(index, item)
            for index, item in enumerate(known_masks)
            if index not in removed
        ],
        audit,
    )
