"""Post-mask verdict interpretation."""
from __future__ import annotations

import pytest

from marine_autolabel.clickengine.verify import accept_mask_verdict, mask_quality_repair_click


class TestAcceptVerdict:
    def test_explicit_keep_is_accepted(self):
        assert accept_mask_verdict({"keep": True})

    def test_explicit_reject_is_rejected(self):
        assert not accept_mask_verdict({"keep": False})

    @pytest.mark.parametrize("answer", [{}, {"keep": None}, {"keep": "yes"}, {"other": 1}])
    def test_anything_but_an_explicit_false_is_accepted(self, answer):
        """Ground truth is incomplete; an unjudged mask is more likely real."""
        assert accept_mask_verdict(answer)

    def test_strict_identity_demands_all_three_confirmations(self):
        full = {"keep": True, "complete_identity": True, "single_identity": True}
        assert accept_mask_verdict(full, strict_identity=True)
        for missing in ("keep", "complete_identity", "single_identity"):
            partial = dict(full)
            partial[missing] = False
            assert not accept_mask_verdict(partial, strict_identity=True)

    def test_strict_identity_rejects_silence(self):
        assert accept_mask_verdict({"keep": True})
        assert not accept_mask_verdict({"keep": True}, strict_identity=True)


class TestRepairClick:
    def test_fragment_wants_a_positive_click(self):
        answer = {"failure": "fragment", "repair_click": {"x": 0.4, "y": 0.6, "label": 1}}
        assert mask_quality_repair_click(answer) == {"x": 0.4, "y": 0.6, "label": 1}

    @pytest.mark.parametrize("failure", ["merge", "background"])
    def test_merge_and_background_want_a_negative_click(self, failure):
        answer = {"failure": failure, "repair_click": {"x": 0.4, "y": 0.6, "label": 0}}
        assert mask_quality_repair_click(answer)["label"] == 0

    def test_a_label_contradicting_its_failure_mode_is_discarded(self):
        """Acting on a self-contradictory verdict would make the mask worse."""
        answer = {"failure": "fragment", "repair_click": {"x": 0.4, "y": 0.6, "label": 0}}
        assert mask_quality_repair_click(answer) is None

    def test_wrong_target_is_not_repairable(self):
        answer = {"failure": "wrong", "repair_click": {"x": 0.4, "y": 0.6, "label": 1}}
        assert mask_quality_repair_click(answer) is None

    def test_failure_matching_is_case_and_space_insensitive(self):
        answer = {"failure": "  FRAGMENT ", "repair_click": {"x": 0.1, "y": 0.1, "label": 1}}
        assert mask_quality_repair_click(answer) is not None

    @pytest.mark.parametrize("click", [
        {"x": 1.4, "y": 0.6, "label": 1},
        {"x": -0.1, "y": 0.6, "label": 1},
        {"x": "a", "y": 0.6, "label": 1},
        {"y": 0.6, "label": 1},
        {},
    ])
    def test_malformed_clicks_are_discarded(self, click):
        assert mask_quality_repair_click({"failure": "fragment", "repair_click": click}) is None

    def test_absent_repair_click_is_none(self):
        assert mask_quality_repair_click({"failure": "fragment"}) is None
        assert mask_quality_repair_click({}) is None
