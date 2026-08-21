"""Batch verification accounting, without an API key."""
from __future__ import annotations

import numpy as np
import pytest

from marine_autolabel.clickengine.verify_batch import (
    coerce_confidence,
    filter_by_confidence,
    verify_masks,
)


def result(area=100, gid=1):
    mask = np.zeros((20, 20), dtype=bool)
    if area:
        mask[: max(1, area // 20), :] = True
    return {"creature_id": gid, "mask": mask, "clicks_used": []}


def judge_returning(answer):
    return lambda r: answer


class TestConfidenceCoercion:
    @pytest.mark.parametrize("value,expected", [(0.5, 0.5), (1.5, 1.0), (-2, 0.0), ("0.3", 0.3)])
    def test_clamped_into_range(self, value, expected):
        assert coerce_confidence(value) == expected

    @pytest.mark.parametrize("value", [None, "high", {}, []])
    def test_non_numeric_is_none(self, value):
        assert coerce_confidence(value) is None


class TestVerifyMasks:
    def test_a_kept_mask_lands_in_kept_with_its_confidence(self):
        kept, dropped = verify_masks(
            [result()], judge=judge_returning({"keep": True, "confidence": 0.9})
        )
        assert len(kept) == 1 and dropped == []
        assert kept[0]["creature_confidence"] == 0.9

    def test_an_explicit_reject_is_dropped(self):
        kept, dropped = verify_masks(
            [result()], judge=judge_returning({"keep": False, "confidence": 0.2})
        )
        assert kept == [] and len(dropped) == 1

    def test_a_missing_confidence_falls_back_by_verdict(self):
        kept, _ = verify_masks([result()], judge=judge_returning({"keep": True}))
        assert kept[0]["creature_confidence"] == 0.75
        _, dropped = verify_masks([result()], judge=judge_returning({"keep": False}))
        assert dropped[0]["creature_confidence"] == 0.1

    def test_an_empty_mask_never_reaches_the_judge(self):
        """Nothing to look at, and asking costs one call per empty mask."""
        calls = []
        kept, dropped = verify_masks(
            [result(area=0)], judge=lambda r: calls.append(r) or {"keep": False}
        )
        assert calls == []
        assert len(kept) == 1 and kept[0]["creature_confidence"] == 0.0

    def test_identity_fields_are_recorded(self):
        kept, _ = verify_masks(
            [result()],
            judge=judge_returning(
                {"keep": True, "complete_identity": True, "single_identity": False}
            ),
        )
        assert kept[0]["mask_complete_identity"] is True
        assert kept[0]["mask_single_identity"] is False

    def test_strict_identity_demands_both_fields(self):
        answer = {"keep": True, "complete_identity": True, "single_identity": False}
        kept, dropped = verify_masks(
            [result()], judge=judge_returning(answer), strict_identity=True
        )
        assert kept == [] and len(dropped) == 1

    def test_a_repair_click_is_attached_for_the_repair_loop(self):
        answer = {
            "keep": False, "failure": "fragment",
            "repair_click": {"x": 0.4, "y": 0.4, "label": 1},
        }
        _, dropped = verify_masks([result()], judge=judge_returning(answer))
        assert dropped[0]["mask_quality_repair_click"] == {"x": 0.4, "y": 0.4, "label": 1}
        assert dropped[0]["mask_quality_failure"] == "fragment"

    def test_an_unknown_failure_label_is_discarded(self):
        _, dropped = verify_masks(
            [result()], judge=judge_returning({"keep": False, "failure": "vibes"})
        )
        assert dropped[0]["mask_quality_failure"] is None

    def test_a_wrong_target_gets_no_repair_click(self):
        answer = {
            "keep": False, "failure": "wrong",
            "repair_click": {"x": 0.4, "y": 0.4, "label": 1},
        }
        _, dropped = verify_masks([result()], judge=judge_returning(answer))
        assert dropped[0]["mask_quality_failure"] == "wrong"
        assert dropped[0]["mask_quality_repair_click"] is None

    def test_a_silent_judge_keeps_the_mask(self):
        """Ground truth is incomplete; an unjudged mask is more likely real."""
        kept, _ = verify_masks([result()], judge=lambda r: None)
        assert len(kept) == 1

    def test_every_result_is_accounted_for(self):
        results = [result(gid=i) for i in range(5)]
        kept, dropped = verify_masks(
            results, judge=lambda r: {"keep": r["creature_id"] % 2 == 0}
        )
        assert len(kept) + len(dropped) == 5

    def test_each_result_is_judged_once(self):
        calls = []
        verify_masks(
            [result(gid=1), result(gid=2)],
            judge=lambda r: calls.append(r["creature_id"]) or {"keep": True},
        )
        assert calls == [1, 2]


class TestSecondOpinion:
    BACKGROUND_REJECT = {"keep": False, "failure": "background", "confidence": 0.05}

    def test_a_background_reject_is_reprieved_by_a_keeping_second_opinion(self):
        kept, dropped = verify_masks(
            [result()],
            judge=judge_returning(self.BACKGROUND_REJECT),
            second_opinion=judge_returning({"keep": True, "confidence": 0.78}),
        )
        assert dropped == [] and len(kept) == 1
        assert kept[0]["creature_confidence"] == 0.78
        assert kept[0]["mask_quality_failure"] is None
        assert kept[0]["mask_second_opinion"] == "kept"

    def test_a_confirmed_background_reject_stays_dropped(self):
        kept, dropped = verify_masks(
            [result()],
            judge=judge_returning(self.BACKGROUND_REJECT),
            second_opinion=judge_returning({"keep": False, "confidence": 0.1}),
        )
        assert kept == [] and len(dropped) == 1
        assert dropped[0]["mask_second_opinion"] == "confirmed_reject"
        assert dropped[0]["mask_quality_failure"] == "background"

    def test_without_the_callable_behaviour_is_unchanged(self):
        kept, dropped = verify_masks(
            [result()], judge=judge_returning(self.BACKGROUND_REJECT)
        )
        assert kept == [] and len(dropped) == 1
        assert "mask_second_opinion" not in dropped[0]

    def test_a_repairable_background_reject_skips_the_second_call(self):
        """A verifier proposing a fix already treated the organism as real."""
        calls = []
        answer = {
            "keep": False, "failure": "background",
            "repair_click": {"x": 0.4, "y": 0.4, "label": 0},
        }
        _, dropped = verify_masks(
            [result()],
            judge=judge_returning(answer),
            second_opinion=lambda r: calls.append(r) or {"keep": True},
        )
        assert calls == []
        assert len(dropped) == 1

    @pytest.mark.parametrize("failure", ["fragment", "merge", "wrong"])
    def test_other_failures_never_consult_the_second_opinion(self, failure):
        calls = []
        _, dropped = verify_masks(
            [result()],
            judge=judge_returning({"keep": False, "failure": failure}),
            second_opinion=lambda r: calls.append(r) or {"keep": True},
        )
        assert calls == []
        assert len(dropped) == 1

    def test_a_kept_mask_never_consults_the_second_opinion(self):
        calls = []
        kept, _ = verify_masks(
            [result()],
            judge=judge_returning({"keep": True, "confidence": 0.9}),
            second_opinion=lambda r: calls.append(r) or {"keep": False},
        )
        assert calls == []
        assert kept[0]["creature_confidence"] == 0.9

    def test_a_reprieve_without_confidence_uses_the_kept_fallback(self):
        kept, _ = verify_masks(
            [result()],
            judge=judge_returning(self.BACKGROUND_REJECT),
            second_opinion=judge_returning({"keep": True}),
        )
        assert kept[0]["creature_confidence"] == 0.75

    def test_a_confirming_reject_with_a_repair_click_feeds_the_repair_loop(self):
        """A closer look may downgrade "no organism" to a fixable defect."""
        second = {
            "keep": False, "failure": "fragment",
            "repair_click": {"x": 0.6, "y": 0.6, "label": 1},
        }
        _, dropped = verify_masks(
            [result()],
            judge=judge_returning(self.BACKGROUND_REJECT),
            second_opinion=judge_returning(second),
        )
        assert dropped[0]["mask_second_opinion"] == "confirmed_reject"
        assert dropped[0]["mask_quality_failure"] == "fragment"
        assert dropped[0]["mask_quality_repair_click"] == {
            "x": 0.6, "y": 0.6, "label": 1}

    def test_a_double_background_reject_keeps_no_repair_click(self):
        second = {
            "keep": False, "failure": "background",
            "repair_click": {"x": 0.6, "y": 0.6, "label": 1},
        }
        _, dropped = verify_masks(
            [result()],
            judge=judge_returning(self.BACKGROUND_REJECT),
            second_opinion=judge_returning(second),
        )
        assert dropped[0]["mask_quality_failure"] == "background"
        assert dropped[0]["mask_quality_repair_click"] is None

    def test_strictness_applies_to_the_second_opinion_too(self):
        answer = {"keep": True, "complete_identity": True, "single_identity": False}
        kept, dropped = verify_masks(
            [result()],
            judge=judge_returning(self.BACKGROUND_REJECT),
            second_opinion=judge_returning(answer),
            strict_identity=True,
        )
        assert kept == [] and len(dropped) == 1
        assert dropped[0]["mask_second_opinion"] == "confirmed_reject"


class TestConfidenceFilter:
    def test_splits_on_the_threshold(self):
        results = [
            {"creature_confidence": 0.9}, {"creature_confidence": 0.5},
            {"creature_confidence": 0.1},
        ]
        above, below = filter_by_confidence(results, 0.5)
        assert len(above) == 2 and len(below) == 1

    def test_the_threshold_is_inclusive(self):
        above, _ = filter_by_confidence([{"creature_confidence": 0.5}], 0.5)
        assert len(above) == 1

    def test_a_missing_confidence_falls_below(self):
        _, below = filter_by_confidence([{}], 0.01)
        assert len(below) == 1

    def test_a_zero_threshold_keeps_everything(self):
        above, below = filter_by_confidence([{}, {"creature_confidence": 0.0}], 0.0)
        assert len(above) == 2 and below == []
