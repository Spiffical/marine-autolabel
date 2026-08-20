"""Text-proposal planning: closed vocabulary, and overlay exclusion."""
from __future__ import annotations

import pytest

from marine_autolabel.sam3svc.text import (
    PromptSpec,
    build_planner_prompt,
    compact_prompt_specs,
    in_exclusion_region,
    select_prompt_specs,
)

CANDIDATES = ["small creatures", "coral", "branching coral", "sea fan", "sponge"]


class TestPlannerPrompt:
    def test_lists_every_candidate(self):
        prompt = build_planner_prompt("05_wide", "a coral garden", CANDIDATES)
        for phrase in CANDIDATES:
            assert f"- {phrase}" in prompt

    def test_carries_the_frame_id_and_scene_note(self):
        prompt = build_planner_prompt("05_wide", "crowded coral garden", CANDIDATES)
        assert "05_wide" in prompt and "crowded coral garden" in prompt

    def test_an_absent_note_reads_as_none(self):
        assert "Scene note: none" in build_planner_prompt("f", "", CANDIDATES)

    def test_no_candidates_still_produces_a_valid_prompt(self):
        assert "- none" in build_planner_prompt("f", "note", [])

    def test_states_that_phrases_are_not_taxonomy(self):
        prompt = build_planner_prompt("f", "n", CANDIDATES)
        assert "retrieval handles only, never taxonomic labels" in prompt

    def test_states_that_selecting_nothing_is_valid(self):
        assert "valid to select no phrase" in build_planner_prompt("f", "n", CANDIDATES)


class TestSelection:
    def test_selects_the_named_candidates_in_order(self):
        specs = select_prompt_specs(CANDIDATES, {"phrases": ["sea fan", "coral"]})
        assert [s.text for s in specs] == ["sea fan", "coral"]
        assert all(isinstance(s, PromptSpec) for s in specs)

    def test_a_hallucinated_phrase_is_discarded(self):
        """It would become a silent, unfindable SAM3 query."""
        specs = select_prompt_specs(CANDIDATES, {"phrases": ["Chrysogorgia", "coral"]})
        assert [s.text for s in specs] == ["coral"]

    def test_matching_ignores_case_and_extra_whitespace(self):
        specs = select_prompt_specs(CANDIDATES, {"phrases": ["  SEA   FAN "]})
        assert [s.text for s in specs] == ["sea fan"]

    def test_duplicates_are_dropped(self):
        specs = select_prompt_specs(CANDIDATES, {"phrases": ["coral", "Coral", "coral"]})
        assert len(specs) == 1

    def test_non_string_entries_are_ignored(self):
        specs = select_prompt_specs(CANDIDATES, {"phrases": ["coral", 7, None, {}]})
        assert [s.text for s in specs] == ["coral"]

    @pytest.mark.parametrize("answer", [{}, {"phrases": "coral"}, {"other": []}, None])
    def test_a_malformed_answer_selects_nothing(self, answer):
        assert select_prompt_specs(CANDIDATES, answer) == []

    def test_an_empty_selection_is_legitimate(self):
        """Click recovery handles what no phrase can ground."""
        assert select_prompt_specs(CANDIDATES, {"phrases": []}) == []


class TestCompactMode:
    def test_uses_every_candidate(self):
        assert [s.text for s in compact_prompt_specs(CANDIDATES)] == CANDIDATES

    def test_deduplicates(self):
        assert len(compact_prompt_specs(["coral", "Coral", " coral "])) == 1

    def test_is_labelled_distinctly_from_adaptive(self):
        assert compact_prompt_specs(["coral"])[0].group == "compact"
        assert select_prompt_specs(["coral"], {"phrases": ["coral"]})[0].group == "adaptive_scene"


class TestExclusionRegions:
    LOGO = [[0.0, 0.0, 0.36, 0.12]]  # the NOAA banner on these clips

    def test_a_proposal_inside_the_banner_is_excluded(self):
        assert in_exclusion_region((0.05, 0.02, 0.20, 0.09), self.LOGO)

    def test_a_proposal_on_the_seafloor_is_kept(self):
        assert not in_exclusion_region((0.5, 0.6, 0.6, 0.7), self.LOGO)

    def test_a_mostly_outside_proposal_is_kept(self):
        """Only a proposal MOSTLY inside the overlay is dropped."""
        assert not in_exclusion_region((0.30, 0.08, 0.90, 0.60), self.LOGO)

    def test_the_threshold_is_configurable(self):
        box = (0.20, 0.06, 0.52, 0.18)  # roughly a quarter inside
        assert not in_exclusion_region(box, self.LOGO, overlap_threshold=0.5)
        assert in_exclusion_region(box, self.LOGO, overlap_threshold=0.2)

    def test_no_regions_excludes_nothing(self):
        assert not in_exclusion_region((0.0, 0.0, 0.1, 0.1), [])

    def test_a_degenerate_box_is_not_excluded(self):
        assert not in_exclusion_region((0.1, 0.1, 0.1, 0.1), self.LOGO)
