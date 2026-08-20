"""Click scoring against (incomplete) ground truth."""
from __future__ import annotations

import numpy as np
import pytest

from marine_autolabel.eval.score import mean_or_zero, recall, score_clicks

H = W = 100


def gt_mask(gid, y0, y1, x0, x1):
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return {"id": gid, "mask": m}


def group(gid, x, y, label=1, description=""):
    return {"id": gid, "description": description, "clicks": [{"x": x, "y": y, "label": label}]}


class TestRecall:
    def test_normal_case(self):
        assert recall(3, 4) == 0.75

    def test_empty_ground_truth_is_zero_not_a_crash(self):
        """The original raised ZeroDivisionError. empty has
        an empty first pass, and the first pass IS the ground truth."""
        assert recall(0, 0) == 0.0
        assert recall(5, 0) == 0.0

    def test_negative_is_treated_as_empty(self):
        assert recall(1, -1) == 0.0


class TestMeanOrZero:
    def test_empty_gives_zero_not_nan(self):
        value = mean_or_zero([])
        assert value == 0.0 and not np.isnan(value)

    def test_normal_mean(self):
        assert mean_or_zero([1.0, 2.0, 3.0]) == 2.0


class TestScoreClicks:
    def test_a_click_inside_a_mask_is_a_hit(self):
        gt = [gt_mask(1, 40, 60, 40, 60)]
        score = score_clicks([group(1, 0.5, 0.5)], gt, W, H)
        assert score["n_hit"] == 1
        assert score["coverage"] == 1.0
        assert score["hits"][1] == 0

    def test_a_click_far_away_is_a_stray(self):
        gt = [gt_mask(1, 40, 60, 40, 60)]
        score = score_clicks([group(1, 0.9, 0.9)], gt, W, H)
        assert score["n_hit"] == 0
        assert score["n_stray"] == 1

    def test_tolerance_rescues_a_near_miss(self):
        gt = [gt_mask(1, 40, 60, 40, 60)]  # x 40..59 -> normalised ~0.404..0.596
        just_outside = group(1, 0.62, 0.5)  # ~61 px, two outside the body
        assert score_clicks([just_outside], gt, W, H, tol_px=0)["n_hit"] == 0
        assert score_clicks([just_outside], gt, W, H, tol_px=8)["n_hit"] == 1

    def test_negative_clicks_are_not_counted(self):
        gt = [gt_mask(1, 40, 60, 40, 60)]
        score = score_clicks([group(1, 0.5, 0.5, label=0)], gt, W, H)
        assert score["n_fg_clicks"] == 0
        assert score["n_hit"] == 0

    def test_one_click_satisfies_every_gt_mask_it_lands_in(self):
        """Hits are computed per ground-truth mask, not per click.

        Overlapping GT entries -- the same animal segmented twice by the first
        pass, say -- are therefore both marked hit by a single click. That
        inflates coverage slightly where GT contains duplicates. `used_clicks`
        exists only to decide which clicks count as strays.
        """
        gt = [gt_mask(1, 40, 60, 40, 60), gt_mask(2, 40, 60, 40, 60)]
        score = score_clicks([group(1, 0.5, 0.5)], gt, W, H)
        assert score["n_hit"] == 2
        assert score["n_stray"] == 0, "the click was used, so it is not a stray"

    def test_nearest_distance_separates_precision_from_detection_misses(self):
        gt = [gt_mask(1, 40, 60, 40, 60), gt_mask(2, 0, 10, 0, 10)]
        score = score_clicks([group(1, 0.62, 0.5)], gt, W, H, tol_px=0)
        assert score["nearest_px"][1] < 10, "clicked near creature 1"
        assert score["nearest_px"][2] > 40, "never looked at creature 2"

    def test_coverage_tolerance_ladder(self):
        gt = [gt_mask(1, 40, 60, 40, 60)]
        score = score_clicks([group(1, 0.70, 0.5)], gt, W, H, tol_px=0)
        assert score["cov_at"][0] == 0
        assert score["cov_at"][40] == 1

    def test_empty_ground_truth_scores_without_crashing(self):
        score = score_clicks([group(1, 0.5, 0.5)], [], W, H)
        assert score["n_gt"] == 0
        assert score["coverage"] == 0.0
        assert score["n_stray"] == 1, "every click is a stray when GT is empty"

    def test_no_clicks_at_all(self):
        score = score_clicks([], [gt_mask(1, 40, 60, 40, 60)], W, H)
        assert score["n_hit"] == 0
        assert score["coverage"] == 0.0
        assert score["nearest_px"][1] is None

    def test_clicks_outside_the_frame_do_not_index_out_of_bounds(self):
        gt = [gt_mask(1, 40, 60, 40, 60)]
        score = score_clicks([group(1, 1.0, 1.0), group(2, 0.0, 0.0)], gt, W, H)
        assert score["n_hit"] == 0

    @pytest.mark.parametrize("n_gt,n_hit,expected", [(4, 2, 0.5), (1, 1, 1.0)])
    def test_coverage_matches_hits_over_gt(self, n_gt, n_hit, expected):
        gt = [gt_mask(i, 10 * i, 10 * i + 8, 10, 18) for i in range(1, n_gt + 1)]
        groups = [group(i, 0.14, (10 * i + 4) / H) for i in range(1, n_hit + 1)]
        score = score_clicks(groups, gt, W, H, tol_px=0)
        assert score["coverage"] == pytest.approx(expected)
