"""frame_outputs schema handling, against real first-pass output.

The two fixtures are deliberately different shapes:
  empty - first pass found 0 objects (the empty edge case)
  populated   - first pass found 2
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from marine_autolabel.runio import (
    decode_frame_row,
    frame_object_metadata,
    iter_output_masks_with_ids,
    load_frame_outputs,
    merge_frame_outputs_by_obj_ids,
)

FIXTURES = Path(__file__).parent / "fixtures"
EMPTY = FIXTURES / "firstpass" / "empty_frame_outputs_rle.json"
POPULATED = FIXTURES / "firstpass" / "populated_frame_outputs_rle.json"


def _row(path: Path) -> tuple[dict, int, int]:
    payload = json.loads(path.read_text())
    height, width = payload["frame_size_hw"]
    return payload["frames"][0], height, width


def test_loads_the_stored_format():
    payload = load_frame_outputs(str(POPULATED))
    assert payload["format_version"] == 2
    assert payload["source"] == "sam3_agent_every_frame"
    assert payload["frame_size_hw"] == [720, 1280]


def test_decodes_stored_rle_rows_with_their_object_ids():
    row, height, width = _row(POPULATED)
    decoded = decode_frame_row(row, height, width)

    assert len(decoded) == len(row["out_obj_ids"])
    assert [oid for oid, _ in decoded] == [int(x) for x in row["out_obj_ids"]]
    for _, mask in decoded:
        assert mask.shape == (height, width)
        assert mask.dtype == np.bool_
        assert mask.any()


def test_empty_first_pass_decodes_to_nothing_without_erroring():
    """empty's first pass found no objects at all."""
    row, height, width = _row(EMPTY)
    assert row["out_obj_ids"] == []
    assert decode_frame_row(row, height, width) == []


def test_decode_frame_row_falls_back_to_the_in_memory_array_form():
    mask = np.zeros((4, 6), dtype=bool)
    mask[1:3, 2:5] = True
    row = {"out_binary_masks": np.stack([mask]), "out_obj_ids": np.array([7])}
    decoded = decode_frame_row(row, 4, 6)
    assert [oid for oid, _ in decoded] == [7]
    assert np.array_equal(decoded[0][1], mask)


def test_metadata_projects_normalised_boxes_into_pixels():
    row, height, width = _row(POPULATED)
    meta = frame_object_metadata(row, height, width)

    assert set(meta) == {int(x) for x in row["out_obj_ids"]}
    for entry in meta.values():
        x1, y1, x2, y2 = entry["box_xyxy"]
        assert 0 <= x1 < x2 <= width - 1
        assert 0 <= y1 < y2 <= height - 1
        assert 0.0 <= entry["confidence"] <= 1.0


def test_metadata_of_an_empty_row_is_empty():
    row, height, width = _row(EMPTY)
    assert frame_object_metadata(row, height, width) == {}


def test_iter_output_masks_resizes_to_the_requested_frame_size():
    small = np.zeros((2, 3), dtype=bool)
    small[0, 0] = True
    row = {"out_binary_masks": np.stack([small]), "out_obj_ids": np.array([1])}
    (obj_id, mask), = iter_output_masks_with_ids(row, 4, 6)
    assert obj_id == 1
    assert mask.shape == (4, 6)


def test_iter_output_masks_numbers_ids_from_one_when_absent():
    masks = np.stack([np.ones((3, 3), dtype=bool), np.zeros((3, 3), dtype=bool)])
    assert [oid for oid, _ in iter_output_masks_with_ids({"pred_masks": masks}, 3, 3)] == [1, 2]


class TestMerge:
    """merge_frame_outputs_by_obj_ids replaces only the listed object ids."""

    @staticmethod
    def _outputs(ids, value):
        n = len(ids)
        masks = np.zeros((n, 4, 4), dtype=bool)
        for i in range(n):
            masks[i, i, :] = value
        return {
            "out_obj_ids": np.asarray(ids, dtype=np.int64),
            "out_binary_masks": masks,
            "out_probs": np.asarray([0.5] * n, dtype=np.float32),
        }

    def test_replaces_only_the_named_ids(self):
        base = self._outputs([1, 2, 3], True)
        patch = self._outputs([2], False)
        merged = merge_frame_outputs_by_obj_ids(base, patch, {2})

        assert list(merged["out_obj_ids"]) == [1, 2, 3]
        assert merged["out_binary_masks"].shape == (3, 4, 4)
        assert merged["out_binary_masks"][0].any()      # id 1 kept from base
        assert not merged["out_binary_masks"][1].any()  # id 2 taken from patch

    def test_merging_into_nothing_yields_empty_arrays_not_a_crash(self):
        merged = merge_frame_outputs_by_obj_ids(self._outputs([1], True), {}, {1})
        assert merged["out_obj_ids"].shape == (0,)
        assert merged["out_binary_masks"].shape[0] == 0
        assert merged["out_probs"].shape == (0,)

    def test_patch_frame_stats_win(self):
        merged = merge_frame_outputs_by_obj_ids(
            self._outputs([1], True), {"frame_stats": {"n": 9}}, set()
        )
        assert merged["frame_stats"] == {"n": 9}


@pytest.mark.parametrize("path", [EMPTY, POPULATED], ids=["empty", "populated"])
def test_decoded_mask_count_matches_the_stored_rle_count(path):
    row, height, width = _row(path)
    assert len(decode_frame_row(row, height, width)) == len(row["out_binary_masks_rle"])
