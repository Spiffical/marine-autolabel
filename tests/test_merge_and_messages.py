"""Prompt assembly and merging accepted masks back into a frame row."""
from __future__ import annotations

import numpy as np
import pytest

from marine_autolabel.clickengine.messages import build_som_prompt_messages
from marine_autolabel.postprocess.merge import merge_accepted_masks_into_row
from marine_autolabel.rle import decode_rle_to_mask


class TestPromptMessages:
    def test_structure_is_system_then_one_user_turn(self):
        messages = build_som_prompt_messages(
            system_prompt="SYS", target_image_path="t.png", neighbour_image_paths=[],
            initial_text_prompt="all life", num_marks=3,
        )
        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[0]["content"] == "SYS"

    def test_target_image_comes_first(self):
        messages = build_som_prompt_messages(
            system_prompt="SYS", target_image_path="target.png",
            neighbour_image_paths=["n1.png", "n2.png"],
            initial_text_prompt="all life", num_marks=2,
        )
        content = messages[1]["content"]
        assert content[0] == {"type": "image", "image": "target.png"}

    def test_reference_frames_follow_in_order(self):
        messages = build_som_prompt_messages(
            system_prompt="SYS", target_image_path="t.png",
            neighbour_image_paths=["n1.png", "n2.png", "n3.png"],
            initial_text_prompt="all life", num_marks=1,
        )
        images = [b["image"] for b in messages[1]["content"] if b["type"] == "image"]
        assert images == ["t.png", "n1.png", "n2.png", "n3.png"]

    def test_the_answer_instruction_is_last(self):
        messages = build_som_prompt_messages(
            system_prompt="SYS", target_image_path="t.png",
            neighbour_image_paths=["n.png"], initial_text_prompt="all life", num_marks=4,
        )
        last = messages[1]["content"][-1]
        assert last["type"] == "text"
        assert "accepted_marks" in last["text"]

    def test_the_mark_range_is_stated(self):
        messages = build_som_prompt_messages(
            system_prompt="SYS", target_image_path="t.png", neighbour_image_paths=[],
            initial_text_prompt="all life", num_marks=7,
        )
        text = " ".join(b.get("text", "") for b in messages[1]["content"])
        assert "1..7" in text

    def test_the_query_is_echoed_back(self):
        messages = build_som_prompt_messages(
            system_prompt="SYS", target_image_path="t.png", neighbour_image_paths=[],
            initial_text_prompt="small benthic creatures", num_marks=1,
        )
        text = " ".join(b.get("text", "") for b in messages[1]["content"])
        assert "small benthic creatures" in text

    def test_reference_framing_only_appears_when_there_are_references(self):
        without = build_som_prompt_messages(
            system_prompt="S", target_image_path="t.png", neighbour_image_paths=[],
            initial_text_prompt="q", num_marks=1,
        )
        with_refs = build_som_prompt_messages(
            system_prompt="S", target_image_path="t.png", neighbour_image_paths=["n.png"],
            initial_text_prompt="q", num_marks=1,
        )
        assert "do not require motion" not in without[1]["content"][1]["text"]
        assert "do not require motion" in with_refs[1]["content"][1]["text"]

    def test_zero_marks_is_refused(self):
        """Nothing to ask about; calling the model would waste a request."""
        with pytest.raises(ValueError, match="num_marks must be >= 1"):
            build_som_prompt_messages(
                system_prompt="S", target_image_path="t.png", neighbour_image_paths=[],
                initial_text_prompt="q", num_marks=0,
            )


def row(n_existing=2):
    return {
        "frame_index": 5,
        "out_obj_ids": [1, 2][:n_existing],
        "out_binary_masks_rle": [{"size": [10, 10], "counts": "0"}] * n_existing,
        "out_boxes_xywh": [[0, 0, 1, 1]] * n_existing,
        "out_probs": [0.9] * n_existing,
        "out_tracker_probs": [0.9] * n_existing,
    }


def new_candidate(score=0.75):
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    return {"mask": mask, "bbox_xywh": [0.2, 0.2, 0.3, 0.3], "score": score}


class TestMerge:
    def test_new_ids_continue_above_the_existing_maximum(self):
        merged = merge_accepted_masks_into_row(row(2), [new_candidate(), new_candidate()])
        assert merged["out_obj_ids"] == [1, 2, 3, 4]
        assert merged["added_obj_ids"] == [3, 4]

    def test_ids_start_at_one_when_the_first_pass_found_nothing(self):
        """empty: first pass contributed 0 of 12."""
        merged = merge_accepted_masks_into_row(row(0), [new_candidate()])
        assert merged["out_obj_ids"] == [1]
        assert merged["added_obj_ids"] == [1]

    def test_provenance_is_recorded_per_object(self):
        merged = merge_accepted_masks_into_row(row(2), [new_candidate()])
        assert merged["source_per_obj_id"] == {"1": "text_agent", "2": "text_agent", "3": "som"}
        assert merged["source"] == "som"

    def test_the_input_row_is_not_mutated(self):
        original = row(2)
        before = {k: list(v) if isinstance(v, list) else v for k, v in original.items()}
        merge_accepted_masks_into_row(original, [new_candidate()])
        assert original["out_obj_ids"] == before["out_obj_ids"]
        assert len(original["out_binary_masks_rle"]) == 2

    def test_the_added_mask_round_trips(self):
        candidate = new_candidate()
        merged = merge_accepted_masks_into_row(row(0), [candidate])
        decoded = decode_rle_to_mask(merged["out_binary_masks_rle"][0], 10, 10)
        assert np.array_equal(decoded.astype(bool), candidate["mask"])

    def test_numpy_ids_are_normalised_for_json(self):
        source = row(2)
        source["out_obj_ids"] = [np.int64(1), np.int64(2)]
        merged = merge_accepted_masks_into_row(source, [new_candidate()])
        assert all(type(i) is int for i in merged["out_obj_ids"])

    def test_score_populates_both_probability_fields(self):
        merged = merge_accepted_masks_into_row(row(0), [new_candidate(score=0.42)])
        assert merged["out_probs"] == [0.42]
        assert merged["out_tracker_probs"] == [0.42]

    def test_merging_nothing_still_stamps_provenance(self):
        merged = merge_accepted_masks_into_row(row(2), [])
        assert merged["added_obj_ids"] == []
        assert set(merged["source_per_obj_id"].values()) == {"text_agent"}
