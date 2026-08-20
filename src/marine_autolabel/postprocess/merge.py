"""Merging accepted click-engine masks back into a frame row.

Ported from `som_missed_creatures.merge_accepted_masks_into_row`.

Provenance is the point: the merged row records which objects came from the
text agent and which the click engine added, so a later analysis can attribute
recall to the stage that earned it. Without it, a frame where the text stage
contributed nothing looks the same as one where it did all the work.
"""
from __future__ import annotations

import copy
from typing import Any

from ..rle import encode_binary_mask_to_rle

SOURCE_TEXT_AGENT = "text_agent"
SOURCE_CLICK_ENGINE = "som"


def merge_accepted_masks_into_row(
    existing_row: dict[str, Any], accepted_candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    """Append accepted candidates to a frame row without mutating the input.

    New objects get ids above the current maximum, so existing ids keep their
    meaning across stages. Object ids are normalised to plain ints, since numpy
    integers are not JSON-serialisable and this row gets written to disk.
    """
    row = copy.deepcopy(existing_row)
    row["out_obj_ids"] = [int(obj_id) for obj_id in row.get("out_obj_ids", [])]
    next_id = (int(max(row["out_obj_ids"])) + 1) if row["out_obj_ids"] else 1

    added_ids: list[int] = []
    for candidate in accepted_candidates:
        row["out_obj_ids"].append(next_id)
        row["out_binary_masks_rle"].append(encode_binary_mask_to_rle(candidate["mask"]))
        row["out_boxes_xywh"].append(list(candidate["bbox_xywh"]))
        row["out_probs"].append(float(candidate.get("score", 0.0)))
        row["out_tracker_probs"].append(float(candidate.get("score", 0.0)))
        added_ids.append(next_id)
        next_id += 1

    added = set(added_ids)
    row["source"] = SOURCE_CLICK_ENGINE
    row["added_obj_ids"] = added_ids
    row["source_per_obj_id"] = {
        str(obj_id): (SOURCE_CLICK_ENGINE if obj_id in added else SOURCE_TEXT_AGENT)
        for obj_id in row["out_obj_ids"]
    }
    return row
