"""Discovery views: the prompt and the images it describes cannot drift apart."""
from __future__ import annotations

import pytest

from marine_autolabel.clickengine.discovery import (
    BASE_VIEWS,
    build_content,
    build_discovery_prompt,
    discovery_views,
)

BASE_PATHS = {
    "raw": "raw.png",
    "grid": "grid.png",
    "strong": "strong.png",
    "outline": "outline.png",
}
FOCUS_PATHS = {**BASE_PATHS, "focus_raw": "fr.png", "focus_strong": "fs.png"}


class TestViews:
    def test_the_base_set_is_four_views(self):
        assert [v.key for v in discovery_views(has_focus_crops=False)] == [
            "raw", "grid", "strong", "outline"
        ]

    def test_focus_crops_append_two_more(self):
        assert [v.key for v in discovery_views(has_focus_crops=True)][-2:] == [
            "focus_raw", "focus_strong"
        ]


class TestPrompt:
    def test_ordinals_follow_the_view_order(self):
        prompt = build_discovery_prompt(pass_instruction="", has_focus_crops=True)
        for ordinal, phrase in [
            ("first", "untouched raw TARGET FRAME"),
            ("second", "lightly translucent green"),
            ("THIRD", "strong green coverage map"),
            ("FOURTH", "outline-only map"),
            ("FIFTH", "enlarged untouched crop"),
            ("SIXTH", "matching enlarged strong-mask crop"),
        ]:
            assert f"The {ordinal} image is" in prompt
            assert phrase in prompt

    def test_focus_crop_sentences_are_absent_without_crops(self):
        prompt = build_discovery_prompt(pass_instruction="", has_focus_crops=False)
        assert "FIFTH" not in prompt
        assert "enlarged untouched crop" not in prompt

    def test_the_outline_view_keeps_its_decisive_framing(self):
        """The 2026-08-18 fix; its wording is why background colonies get found."""
        prompt = build_discovery_prompt(pass_instruction="", has_focus_crops=False)
        assert "decisive view for crowded branching scenes" in prompt
        assert "is NOT thereby segmented" in prompt

    def test_the_pass_instruction_is_placed_before_the_contract(self):
        prompt = build_discovery_prompt(
            pass_instruction="SWEEP THE LEFT HALF. ", has_focus_crops=False
        )
        assert prompt.index("SWEEP THE LEFT HALF") < prompt.index("Find every real")

    def test_reference_frames_are_announced_after_the_named_views(self):
        prompt = build_discovery_prompt(pass_instruction="", has_focus_crops=True)
        assert prompt.index("SIXTH") < prompt.index("Subsequent images are raw REFERENCE")

    def test_the_answer_contract_is_last(self):
        prompt = build_discovery_prompt(pass_instruction="", has_focus_crops=False)
        assert prompt.rstrip().endswith("</answer>")


class TestContentAssembly:
    def test_images_come_in_the_order_the_prompt_describes(self):
        content = build_content(
            view_paths=BASE_PATHS, neighbour_paths=["n1.png"], pass_instruction=""
        )
        images = [b["image"] for b in content if b["type"] == "image"]
        assert images == ["raw.png", "grid.png", "strong.png", "outline.png", "n1.png"]

    def test_focus_crops_slot_in_before_the_reference_frames(self):
        content = build_content(
            view_paths=FOCUS_PATHS, neighbour_paths=["n1.png"], pass_instruction=""
        )
        images = [b["image"] for b in content if b["type"] == "image"]
        assert images[4:6] == ["fr.png", "fs.png"]
        assert images[-1] == "n1.png"

    def test_the_prompt_is_the_final_block(self):
        content = build_content(view_paths=BASE_PATHS, neighbour_paths=[], pass_instruction="")
        assert content[-1]["type"] == "text"
        assert "TARGET FRAME" in content[-1]["text"]

    def test_focus_crops_switch_the_prompt_automatically(self):
        content = build_content(view_paths=FOCUS_PATHS, neighbour_paths=[], pass_instruction="")
        assert "SIXTH" in content[-1]["text"]

    def test_a_missing_view_is_refused(self):
        """Silent corruption otherwise: the model is told image 4 is something
        it is not, and the only symptom is worse clicks."""
        incomplete = {k: v for k, v in BASE_PATHS.items() if k != "outline"}
        with pytest.raises(ValueError, match="missing=\\['outline'\\]"):
            build_content(view_paths=incomplete, neighbour_paths=[], pass_instruction="")

    def test_an_unknown_view_is_refused(self):
        with pytest.raises(ValueError, match="unexpected=\\['bogus'\\]"):
            build_content(
                view_paths={**BASE_PATHS, "bogus": "b.png"},
                neighbour_paths=[], pass_instruction="",
            )

    def test_only_one_focus_crop_is_refused(self):
        half = {**BASE_PATHS, "focus_raw": "fr.png"}
        with pytest.raises(ValueError, match="focus_strong"):
            build_content(view_paths=half, neighbour_paths=[], pass_instruction="")

    def test_the_error_names_the_expected_order(self):
        with pytest.raises(ValueError, match="expected order="):
            build_content(view_paths={}, neighbour_paths=[], pass_instruction="")

    def test_no_reference_frames_is_fine(self):
        content = build_content(view_paths=BASE_PATHS, neighbour_paths=[], pass_instruction="")
        assert sum(1 for b in content if b["type"] == "image") == len(BASE_VIEWS)
