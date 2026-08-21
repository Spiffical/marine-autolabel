"""The mask generator's control flow, without SAM3 or an API key."""
from __future__ import annotations

import numpy as np
import pytest

from marine_autolabel.clickengine.generate import refine_group

H = W = 100


def masks_and_scores(areas, scores):
    """Build candidate masks with the given pixel areas."""
    stack = np.zeros((len(areas), H, W), dtype=bool)
    for i, area in enumerate(areas):
        rows = max(1, area // W)
        stack[i, :rows, :] = True
    return stack, np.array(scores, dtype=np.float32)


def group(gid=1, clicks=None):
    return {
        "id": gid,
        "description": "a crab",
        "clicks": clicks or [{"x": 0.5, "y": 0.5, "label": 1}],
    }


def constant_predict(areas=(400, 900, 2000), scores=(0.5, 0.9, 0.7)):
    return lambda clicks: masks_and_scores(areas, scores)


def verdicts(*sequence):
    """A judge that returns each verdict in turn, repeating the last."""
    calls = {"n": 0}

    def judge(**kwargs):
        index = min(calls["n"], len(sequence) - 1)
        calls["n"] += 1
        return sequence[index]

    judge.calls = calls
    return judge


class TestPicking:
    def test_a_good_verdict_returns_the_named_candidate(self):
        result, trace = refine_group(
            group(), predict=constant_predict(), judge=verdicts({"verdict": "good", "index": 0}),
            width=W, height=H,
        )
        assert result["select_reason"] == "mllm_pick"
        assert result["status"] == "verified"
        assert result["area_px"] == 400
        assert trace[-1]["verdict"] == "good#0"

    def test_an_out_of_range_pick_snaps_to_the_smallest_valid_candidate(self):
        """Original semantics, restored. The first port snapped to the highest
        score, but the highest-scoring candidate is often exactly the
        over-merged blob -- smallest-valid fights the blob problem."""
        result, _ = refine_group(
            group(), predict=constant_predict(), judge=verdicts({"verdict": "good", "index": 9}),
            width=W, height=H,
        )
        assert result["area_px"] == 400, "smallest plausible candidate wins"

    def test_the_original_choice_key_is_accepted(self):
        result, _ = refine_group(
            group(), predict=constant_predict(), judge=verdicts({"verdict": "good", "choice": 2}),
            width=W, height=H,
        )
        assert result["area_px"] == 2000, "choice=2 names the third candidate"

    def test_the_result_carries_the_group_identity(self):
        result, _ = refine_group(
            group(gid=7), predict=constant_predict(),
            judge=verdicts({"verdict": "good", "index": 0}), width=W, height=H,
        )
        assert result["creature_id"] == 7
        assert result["description"] == "a crab"


class TestAbandonment:
    def test_an_abandon_verdict_returns_an_empty_mask(self):
        result, _ = refine_group(
            group(), predict=constant_predict(), judge=verdicts({"verdict": "abandon"}),
            width=W, height=H,
        )
        assert result["status"] == "abandoned"
        assert result["area_px"] == 0

    def test_every_candidate_spanning_the_frame_abandons(self):
        """A click on busy substrate makes SAM3 return frame-spanning regions."""
        result, trace = refine_group(
            group(), predict=constant_predict(areas=(9000, 9500, 9900), scores=(0.9, 0.9, 0.9)),
            judge=verdicts({"verdict": "good", "index": 0}), width=W, height=H,
        )
        assert result["status"] == "abandoned"
        assert trace[0]["verdict"] == "no_valid_band"

    def test_a_creature_filling_half_the_frame_is_still_allowed(self):
        """The cap is loose on purpose; area cannot separate close creatures
        from substrate grabs."""
        result, _ = refine_group(
            group(), predict=constant_predict(areas=(5000,), scores=(0.9,)),
            judge=verdicts({"verdict": "good", "index": 0}), width=W, height=H,
        )
        assert result["status"] == "verified"


class TestAddingClicks:
    def test_an_add_verdict_extends_the_click_list(self):
        seen = []

        def predict(clicks):
            seen.append(len(clicks))
            return masks_and_scores((400,), (0.8,))

        refine_group(
            group(), predict=predict,
            judge=verdicts(
                {"verdict": "add", "click": {"x": 0.6, "y": 0.6, "label": 0}},
                {"verdict": "good", "index": 0},
            ),
            width=W, height=H,
        )
        assert seen == [1, 2]

    def test_a_duplicate_click_is_retried_then_gives_up(self):
        """Repeating the same click makes no progress."""
        duplicate = {"verdict": "add", "click": {"x": 0.5, "y": 0.5, "label": 1}}
        result, trace = refine_group(
            group(), predict=constant_predict(), judge=verdicts(duplicate),
            width=W, height=H, max_attempts=1, strict_quality=True,
        )
        reasons = [t.get("verdict") for t in trace]
        assert "duplicate_click_retry" in reasons
        assert "duplicate_click_no_progress" in reasons
        assert result["status"] == "abandoned"

    def test_the_click_budget_stops_further_additions(self):
        add = {"verdict": "add", "click": {"x": 0.1, "y": 0.1, "label": 1}}
        result, _ = refine_group(
            group(), predict=constant_predict(), judge=verdicts(add),
            width=W, height=H, max_clicks=2, max_attempts=1,
        )
        assert result["select_reason"] == "parse_fail_band", "budget exhausted, band kept"


class TestStrictQuality:
    def test_strict_refuses_an_unverified_fallback(self):
        result, trace = refine_group(
            group(), predict=constant_predict(), judge=verdicts({"verdict": "nonsense"}),
            width=W, height=H, max_attempts=1, strict_quality=True,
        )
        assert result["status"] == "abandoned"
        assert trace[-1]["verdict"] == "exhausted_no_verified_mask"

    def test_lenient_keeps_the_band_candidate(self):
        result, _ = refine_group(
            group(), predict=constant_predict(), judge=verdicts({"verdict": "nonsense"}),
            width=W, height=H, max_attempts=1, strict_quality=False,
        )
        assert result["select_reason"] == "parse_fail_band"
        assert result["status"] == "unverified"


class TestAttemptsAndReseed:
    def test_a_reject_restarts_with_a_fresh_seed(self):
        seeds = []

        def predict(clicks):
            seeds.append(round(clicks[0]["x"], 2))
            return masks_and_scores((400,), (0.8,))

        refine_group(
            group(), predict=predict,
            judge=verdicts({"verdict": "reject"}, {"verdict": "reject"},
                           {"verdict": "good", "index": 0}),
            reseed=lambda attempt: {"x": 0.1 * attempt, "y": 0.2, "label": 1},
            width=W, height=H, max_attempts=3,
        )
        assert seeds[0] == 0.5
        assert seeds[1] == 0.1, "second attempt uses the reseeded click"

    def test_a_reseed_that_abandons_ends_it(self):
        result, trace = refine_group(
            group(), predict=constant_predict(), judge=verdicts({"verdict": "reject"}),
            reseed=lambda attempt: {"verdict": "abandon"},
            width=W, height=H, max_attempts=3,
        )
        assert result["status"] == "abandoned"
        assert any(t.get("verdict") == "reseed_abandon" for t in trace)

    def test_without_a_reseed_the_attempt_simply_repeats(self):
        result, _ = refine_group(
            group(), predict=constant_predict(), judge=verdicts({"verdict": "reject"}),
            width=W, height=H, max_attempts=2, strict_quality=True,
        )
        assert result["status"] == "abandoned"

    def test_the_best_candidate_across_attempts_is_kept(self):
        scores = iter([0.4, 0.95, 0.5])

        def predict(clicks):
            return masks_and_scores((400,), (next(scores, 0.1),))

        result, trace = refine_group(
            group(), predict=predict, judge=verdicts({"verdict": "reject"}),
            width=W, height=H, max_attempts=3, strict_quality=False,
        )
        assert trace[-1]["verdict"] == "exhausted_keep_best"
        assert result["score"] == pytest.approx(0.95)

    def test_zero_attempts_is_refused_rather_than_crashing(self):
        """The original unpacked `best` unconditionally and raised TypeError."""
        with pytest.raises(ValueError, match="max_attempts must be >= 1"):
            refine_group(
                group(), predict=constant_predict(), judge=verdicts({"verdict": "good"}),
                width=W, height=H, max_attempts=0,
            )
