"""The post-verification repair loop, and its bound."""
from __future__ import annotations

import numpy as np
import pytest

from marine_autolabel.clickengine.recovery import (
    is_actionable_repair_click,
    run_repair_rounds,
)


def rejected(mask_area=100, repair=(0.5, 0.5, 1), clicks_used=()):
    mask = np.zeros((20, 20), dtype=bool)
    if mask_area:
        mask[: max(1, mask_area // 20), :] = True
    item = {"mask": mask, "clicks_used": list(clicks_used)}
    if repair is not None:
        item["mask_quality_repair_click"] = {"x": repair[0], "y": repair[1], "label": repair[2]}
    return item


def good_mask():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    return mask


class TestActionableRepairClick:
    def test_a_fresh_click_is_actionable(self):
        assert is_actionable_repair_click({"x": 0.5, "y": 0.5, "label": 1}, [])

    def test_a_click_on_top_of_an_earlier_same_label_one_is_not(self):
        prior = [{"x": 0.50, "y": 0.50, "label": 1}]
        assert not is_actionable_repair_click({"x": 0.505, "y": 0.50, "label": 1}, prior)

    def test_the_same_point_with_a_different_label_is_actionable(self):
        prior = [{"x": 0.50, "y": 0.50, "label": 1}]
        assert is_actionable_repair_click({"x": 0.50, "y": 0.50, "label": 0}, prior)

    def test_a_click_far_from_the_prior_one_is_actionable(self):
        prior = [{"x": 0.50, "y": 0.50, "label": 1}]
        assert is_actionable_repair_click({"x": 0.80, "y": 0.50, "label": 1}, prior)

    @pytest.mark.parametrize(
        "bad",
        [None, "x", {}, {"x": 0.5, "y": 0.5}, {"x": 0.5, "y": 0.5, "label": 2}],
    )
    def test_malformed_clicks_are_not_actionable(self, bad):
        assert not is_actionable_repair_click(bad, [])

    def test_boolean_coordinates_are_refused(self):
        assert not is_actionable_repair_click({"x": True, "y": 0.5, "label": 1}, [])


class TestRepairRounds:
    def test_nothing_rejected_means_no_work(self):
        out = run_repair_rounds([], regenerate=lambda *_: None, verify=lambda *_: ([], []))
        assert out["rounds_run"] == 0
        assert out["recovered"] == []

    def test_a_repair_that_verifies_is_recovered(self):
        def regenerate(item, click):
            return {"mask": good_mask(), "clicks_used": []}

        out = run_repair_rounds(
            [rejected()], regenerate=regenerate, verify=lambda results, r: (results, [])
        )
        assert out["repaired"] == 1
        assert out["rounds_run"] == 1
        assert out["attempts"] == 1
        assert not out["hit_round_cap"]

    def test_an_item_without_an_actionable_click_is_dropped_immediately(self):
        calls = []

        def regenerate(item, click):
            calls.append(click)
            return {"mask": good_mask(), "clicks_used": []}

        out = run_repair_rounds(
            [rejected(repair=None)], regenerate=regenerate, verify=lambda r, n: (r, [])
        )
        assert calls == [], "no repair click means no generation call"
        assert len(out["terminal_dropped"]) == 1

    def test_a_repair_producing_an_empty_mask_is_dropped(self):
        def regenerate(item, click):
            return {"mask": np.zeros((20, 20), dtype=bool), "clicks_used": []}

        out = run_repair_rounds(
            [rejected()], regenerate=regenerate, verify=lambda r, n: (r, [])
        )
        assert out["repaired"] == 0
        assert len(out["terminal_dropped"]) == 1

    def test_repair_history_accumulates_across_rounds(self):
        def regenerate(item, click):
            return {
                "mask": good_mask(),
                "clicks_used": [],
                "mask_quality_repair_click": {"x": 0.1 * len(
                    item.get("postverify_repair_history") or []) + 0.1, "y": 0.9, "label": 1},
                "postverify_repair_history": item.get("postverify_repair_history"),
            }

        rounds = {"n": 0}

        def verify(results, round_number):
            rounds["n"] += 1
            return ([], results) if rounds["n"] < 3 else (results, [])

        out = run_repair_rounds([rejected()], regenerate=regenerate, verify=verify)
        assert out["rounds_run"] == 3
        assert len(out["recovered"][0]["postverify_repair_history"]) == 3

    def test_the_round_cap_bounds_a_non_converging_verifier(self):
        """The original had no bound; a 9-round case was observed on dense coral."""
        def regenerate(item, click):
            # Always moves far enough to look like progress.
            n = len(item.get("postverify_repair_history") or [])
            return {
                "mask": good_mask(),
                "clicks_used": [],
                "mask_quality_repair_click": {"x": 0.05 * (n + 1), "y": 0.9, "label": 1},
                "postverify_repair_history": item.get("postverify_repair_history"),
            }

        out = run_repair_rounds(
            [rejected()],
            regenerate=regenerate,
            verify=lambda results, n: ([], results),  # never satisfied
            max_repair_rounds=4,
        )
        assert out["rounds_run"] == 4
        assert out["hit_round_cap"] is True
        assert out["attempts"] == 4
        assert len(out["terminal_dropped"]) == 1

    def test_the_cap_is_configurable(self):
        def regenerate(item, click):
            n = len(item.get("postverify_repair_history") or [])
            return {
                "mask": good_mask(),
                "clicks_used": [],
                "mask_quality_repair_click": {"x": 0.05 * (n + 1), "y": 0.9, "label": 1},
                "postverify_repair_history": item.get("postverify_repair_history"),
            }

        out = run_repair_rounds(
            [rejected()], regenerate=regenerate,
            verify=lambda results, n: ([], results), max_repair_rounds=2,
        )
        assert out["rounds_run"] == 2 and out["hit_round_cap"]

    def test_zero_rounds_disables_repair_entirely(self):
        out = run_repair_rounds(
            [rejected()], regenerate=lambda *_: {"mask": good_mask()},
            verify=lambda r, n: (r, []), max_repair_rounds=0,
        )
        assert out["rounds_run"] == 0
        assert out["attempts"] == 0
        assert out["hit_round_cap"] is True
        assert len(out["terminal_dropped"]) == 1

    def test_the_round_number_reaches_the_verifier(self):
        seen = []

        def verify(results, round_number):
            seen.append(round_number)
            return (results, [])

        run_repair_rounds(
            [rejected()], regenerate=lambda *_: {"mask": good_mask(), "clicks_used": []},
            verify=verify,
        )
        assert seen == [1]
