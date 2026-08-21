"""The judgement prompts: verbatim fidelity is the requirement.

Byte-parity against the archived original is checked in
tests/test_parity_parsers.py when MAL_LEGACY_REPO is set. These tests pin the
load-bearing clauses so an edit that drops one fails immediately, everywhere.
"""
from __future__ import annotations

import numpy as np

from marine_autolabel.clickengine.judgement import (
    DUPLICATE_FEEDBACK,
    MASK_VERIFY_FMT,
    MM_JUDGE_FMT,
    build_judge_prompt,
    build_verify_prompt,
    geometry_fact,
    repair_fact,
)


def mask(y0=100, y1=300, x0=200, x1=500, h=720, w=1280):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


class TestJudgePrompt:
    def test_carries_the_untrusted_hypothesis_framing(self):
        prompt = build_judge_prompt("tan crab")
        assert "'tan crab' is an UNTRUSTED HYPOTHESIS" in prompt

    def test_answer_key_is_choice_and_panel_relative(self):
        prompt = build_judge_prompt("x")
        assert '"choice": <0|1|2>' in prompt
        assert "NORMALIZED [0,1] within ONE panel" in prompt

    def test_good_is_non_comparative(self):
        assert "GOOD IS A STRICT, NON-COMPARATIVE VERDICT" in build_judge_prompt("x")

    def test_budget_clause_appears_only_when_spent(self):
        assert "NO CLICK BUDGET REMAINS" not in build_judge_prompt("x")
        assert "NO CLICK BUDGET REMAINS" in build_judge_prompt("x", budget_reached=True)

    def test_duplicate_feedback_is_threaded_through(self):
        prompt = build_judge_prompt("x", duplicate_feedback=DUPLICATE_FEEDBACK)
        assert "duplicated an existing" in prompt

    def test_ends_with_the_contract(self):
        assert build_judge_prompt("x").endswith(MM_JUDGE_FMT)


class TestGeometryFact:
    def test_bbox_is_normalised_to_three_places(self):
        fact = geometry_fact(mask(), 1280, 720)
        assert "x=0.156..0.390" in fact
        assert "y=0.139..0.416" in fact

    def test_interior_mask_touches_nothing(self):
        assert "touches frame edge(s)=none" in geometry_fact(mask(), 1280, 720)

    def test_edge_contact_is_named(self):
        m = mask(0, 50, 1200, 1280)
        fact = geometry_fact(m, 1280, 720)
        # The original enumerates left, right, top, bottom -- in that order.
        assert "touches frame edge(s)=right,top" in fact

    def test_facts_override_visual_guessing(self):
        assert "override any visual guess" in geometry_fact(mask(), 1280, 720)

    def test_an_empty_mask_does_not_crash(self):
        assert "EMPTY" in geometry_fact(np.zeros((720, 1280), bool), 1280, 720)


class TestRepairFact:
    def test_absent_history_is_silent(self):
        assert repair_fact(None) == ""
        assert repair_fact([]) == ""

    def test_history_carries_the_no_progress_rule(self):
        fact = repair_fact([
            {"round": 1, "failure": "fragment", "click": {"x": 0.41, "y": 0.62, "label": 1}}
        ])
        assert "round 1: fragment at (0.410, 0.620, label=1)" in fact
        assert "no-progress decision, not permission to accept a bad mask" in fact


class TestVerifyPrompt:
    def test_the_void_blob_defense_is_present(self):
        """The clause whose absence let masks be accepted on empty water."""
        prompt = build_verify_prompt("a coral", mask(), 1280, 720)
        assert "NOT evidence that an organism exists" in prompt
        assert "convincing blobs from water, shadow, haze, or substrate" in prompt
        assert "The untouched FIRST image must show biological texture" in prompt

    def test_three_images_are_described_in_order(self):
        prompt = build_verify_prompt("x", mask(), 1280, 720)
        assert "FIRST image is the raw full frame" in prompt
        assert "SECOND image is the same full frame with the candidate in cyan" in prompt
        assert "THIRD review sheet" in prompt

    def test_identity_gating_is_explicit(self):
        prompt = build_verify_prompt("x", mask(), 1280, 720)
        assert "Set keep=true only when BOTH identity fields are true" in prompt

    def test_frame_clip_contradiction_rejects(self):
        prompt = build_verify_prompt("x", mask(), 1280, 720)
        assert "frame-clipped but touches frame edge(s)=none, reject" in prompt

    def test_all_life_widens_the_subject(self):
        wide = build_verify_prompt("x", mask(), 1280, 720, allow_all_life=True)
        narrow = build_verify_prompt("x", mask(), 1280, 720, allow_all_life=False)
        assert "sponge, coral, macroalga" in wide
        assert "a real, distinct living animal" in narrow

    def test_repair_history_reaches_the_prompt(self):
        prompt = build_verify_prompt(
            "x", mask(), 1280, 720,
            repair_history=[{"round": 2, "failure": "merge",
                             "click": {"x": 0.1, "y": 0.2, "label": 0}}],
        )
        assert "REPAIR HISTORY FOR THIS SAME PROPOSAL" in prompt

    def test_ends_with_the_contract(self):
        assert build_verify_prompt("x", mask(), 1280, 720).endswith(MASK_VERIFY_FMT)

    def test_an_empty_description_gets_the_placeholder(self):
        assert "'the masked region'" in build_verify_prompt("", mask(), 1280, 720)
