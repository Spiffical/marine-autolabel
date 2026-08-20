"""The algorithmic core of the click engine, exercised without a GPU."""
from __future__ import annotations

import numpy as np
import pytest

from marine_autolabel.geometry import (
    clean_candidate_components,
    duplicate_click,
    geom_dedup,
    iou,
    keep_positive_seed_components,
    mask_level_nms,
    merge_corrected_positive_click,
    norm_to_pixel,
    overlap_coefficient,
    select_in_band,
    smallest_valid,
)


def box(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


class TestNormToPixel:
    def test_maps_corners_onto_the_last_index(self):
        assert norm_to_pixel(0.0, 0.0, 100, 50) == (0, 0)
        assert norm_to_pixel(1.0, 1.0, 100, 50) == (99, 49)

    @pytest.mark.parametrize("x,y", [(-5.0, -5.0), (5.0, 5.0), (0.5, -1.0)])
    def test_out_of_range_is_clamped_not_wrapped(self, x, y):
        px, py = norm_to_pixel(x, y, 10, 10)
        assert 0 <= px <= 9 and 0 <= py <= 9


class TestIou:
    def test_identical_masks(self):
        m = box(10, 10, 2, 6, 2, 6)
        assert iou(m, m) == 1.0

    def test_disjoint_masks(self):
        assert iou(box(10, 10, 0, 3, 0, 3), box(10, 10, 6, 9, 6, 9)) == 0.0

    def test_two_empty_masks_are_zero_not_nan(self):
        empty = np.zeros((4, 4), dtype=bool)
        assert iou(empty, empty) == 0.0

    def test_half_overlap(self):
        a, b = box(10, 10, 0, 4, 0, 10), box(10, 10, 2, 6, 0, 10)
        assert iou(a, b) == pytest.approx(20 / 60)


class TestOverlapCoefficient:
    def test_contained_mask_scores_one_where_iou_would_be_low(self):
        big, small = box(20, 20, 0, 20, 0, 20), box(20, 20, 8, 10, 8, 10)
        assert overlap_coefficient(big, small) == 1.0
        assert iou(big, small) < 0.02

    def test_empty_input_is_zero(self):
        assert overlap_coefficient(np.zeros((4, 4), bool), box(4, 4, 0, 2, 0, 2)) == 0.0


class TestSeedComponents:
    @staticmethod
    def _two_blobs():
        m = np.zeros((20, 20), dtype=bool)
        m[2:6, 2:6] = True     # blob A, near (0.2, 0.2)
        m[14:18, 14:18] = True  # blob B, near (0.8, 0.8)
        return m

    def test_keeps_only_the_seeded_blob(self):
        clicks = [{"x": 0.21, "y": 0.21, "label": 1}]
        kept = keep_positive_seed_components(self._two_blobs(), clicks)
        assert kept[2:6, 2:6].all()
        assert not kept[14:18, 14:18].any()

    def test_two_seeds_keep_both_blobs(self):
        clicks = [{"x": 0.21, "y": 0.21, "label": 1}, {"x": 0.82, "y": 0.82, "label": 1}]
        kept = keep_positive_seed_components(self._two_blobs(), clicks)
        assert kept[2:6, 2:6].all() and kept[14:18, 14:18].all()

    def test_negative_clicks_do_not_seed(self):
        """With no positive click the mask is returned untouched."""
        m = self._two_blobs()
        kept = keep_positive_seed_components(m, [{"x": 0.21, "y": 0.21, "label": 0}])
        assert np.array_equal(kept, m)

    def test_seed_landing_outside_falls_back_to_the_largest_component(self):
        m = np.zeros((20, 20), dtype=bool)
        m[2:4, 2:4] = True      # small
        m[10:18, 10:18] = True  # large
        kept = keep_positive_seed_components(m, [{"x": 0.5, "y": 0.02, "label": 1}])
        assert kept[10:18, 10:18].all()
        assert not kept[2:4, 2:4].any()

    def test_empty_mask_stays_empty(self):
        empty = np.zeros((8, 8), dtype=bool)
        assert not keep_positive_seed_components(empty, [{"x": 0.5, "y": 0.5, "label": 1}]).any()

    def test_single_component_is_returned_untouched(self):
        m = box(10, 10, 1, 9, 1, 9)
        assert np.array_equal(keep_positive_seed_components(m, [{"x": .5, "y": .5, "label": 1}]), m)

    def test_clean_candidates_preserves_stack_shape(self):
        masks = np.stack([self._two_blobs(), self._two_blobs()])
        out = clean_candidate_components(masks, [{"x": 0.21, "y": 0.21, "label": 1}])
        assert out.shape == masks.shape


class TestSelectInBand:
    def test_prefers_the_smallest_plausible_mask(self):
        """SAM3 returns part / whole / scene; the individual is the smallest in band."""
        h = w = 100
        masks = np.stack([
            box(h, w, 0, 90, 0, 90),   # merged scene, over the 50% cap
            box(h, w, 10, 30, 10, 30),  # the individual, 400 px
            box(h, w, 10, 40, 10, 40),  # merged with a neighbour, 900 px
        ])
        index, reason = select_in_band(masks, np.array([0.9, 0.5, 0.7]))
        assert index == 1
        assert reason == "smallest_in_band"

    def test_all_below_minimum_falls_back_to_the_best_score(self):
        masks = np.stack([box(100, 100, 0, 2, 0, 2), box(100, 100, 0, 3, 0, 3)])
        index, reason = select_in_band(masks, np.array([0.2, 0.95]))
        assert reason == "fallback_highest_score"
        assert index == 1

    def test_all_above_the_band_takes_the_smallest_above_min(self):
        h = w = 100
        masks = np.stack([box(h, w, 0, 100, 0, 100), box(h, w, 0, 80, 0, 100)])
        index, reason = select_in_band(masks, np.array([0.5, 0.5]))
        assert reason == "smallest_above_min"
        assert index == 1

    def test_smallest_valid_ignores_rejected_candidates(self):
        masks = np.stack([box(50, 50, 0, 5, 0, 5), box(50, 50, 0, 20, 0, 20)])
        assert smallest_valid(masks, [False, True]) == 1


class TestMaskLevelNms:
    @staticmethod
    def _r(mask, seed=None):
        out = {"mask": mask}
        if seed is not None:
            out["seed_click"] = {"x": seed[0], "y": seed[1]}
        return out

    def test_near_duplicates_are_collapsed(self):
        a, b = box(20, 20, 2, 10, 2, 10), box(20, 20, 2, 10, 3, 11)
        kept, removed = mask_level_nms([self._r(a), self._r(b)])
        assert len(kept) == 1 and removed == 1

    def test_distinct_masks_all_survive(self):
        kept, removed = mask_level_nms(
            [self._r(box(20, 20, 0, 5, 0, 5)), self._r(box(20, 20, 12, 18, 12, 18))]
        )
        assert len(kept) == 2 and removed == 0

    def test_broad_container_is_dropped_when_its_seed_is_in_the_tight_mask(self):
        tight = box(40, 40, 10, 20, 10, 20)
        broad = box(40, 40, 5, 35, 5, 35)
        kept, removed = mask_level_nms(
            [self._r(broad, seed=(0.37, 0.37)), self._r(tight, seed=(0.37, 0.37))]
        )
        assert removed == 1
        assert int(kept[0]["mask"].sum()) == int(tight.sum()), "the tighter mask must win"

    def test_interlaced_colonies_with_distinct_seeds_are_kept(self):
        tight = box(40, 40, 10, 20, 10, 20)
        broad = box(40, 40, 5, 35, 5, 35)
        kept, _ = mask_level_nms(
            [self._r(broad, seed=(0.80, 0.80)), self._r(tight, seed=(0.37, 0.37))]
        )
        assert len(kept) == 2

    def test_original_order_is_restored(self):
        big, small = box(30, 30, 0, 25, 0, 25), box(30, 30, 26, 29, 26, 29)
        kept, _ = mask_level_nms([self._r(big), self._r(small)])
        assert [int(r["mask"].sum()) for r in kept] == [int(big.sum()), int(small.sum())]


class TestClickDedup:
    @staticmethod
    def _g(x, y, gid=0):
        return {"id": gid, "clicks": [{"x": x, "y": y, "label": 1}]}

    def test_nearby_clicks_collapse_and_ids_renumber(self):
        groups = [self._g(0.50, 0.50), self._g(0.51, 0.50), self._g(0.90, 0.90)]
        kept, removed = geom_dedup(groups, 1280, 720, px=40)
        assert removed == 1
        assert [g["id"] for g in kept] == [1, 2]

    def test_separation_beyond_the_threshold_is_kept(self):
        kept, removed = geom_dedup([self._g(0.10, 0.10), self._g(0.30, 0.10)], 1280, 720, px=40)
        assert removed == 0 and len(kept) == 2

    def test_threshold_is_in_pixels_so_frame_size_matters(self):
        groups = lambda: [self._g(0.50, 0.50), self._g(0.52, 0.50)]  # noqa: E731
        assert geom_dedup(groups(), 1280, 720, px=40)[1] == 1   # 25 px apart
        assert geom_dedup(groups(), 320, 180, px=40)[1] == 1    # 6 px apart
        assert geom_dedup(groups(), 8000, 4000, px=40)[1] == 0  # 160 px apart

    def test_duplicate_click_needs_a_matching_label(self):
        existing = [{"x": 0.5, "y": 0.5, "label": 1}]
        assert duplicate_click(existing, {"x": 0.5, "y": 0.5, "label": 1})
        assert not duplicate_click(existing, {"x": 0.5, "y": 0.5, "label": 0})

    def test_duplicate_click_tolerance(self):
        existing = [{"x": 0.5, "y": 0.5, "label": 1}]
        assert not duplicate_click(existing, {"x": 0.52, "y": 0.5, "label": 1})


class TestMergeCorrected:
    def test_replaces_the_nearest_positive_and_keeps_negatives(self):
        clicks = [
            {"x": 0.10, "y": 0.10, "label": 1},
            {"x": 0.80, "y": 0.80, "label": 1},
            {"x": 0.50, "y": 0.50, "label": 0},
        ]
        merged = merge_corrected_positive_click(clicks, {"x": 0.82, "y": 0.82, "label": 1})
        assert merged[0] == clicks[0]
        assert merged[1] == {"x": 0.82, "y": 0.82, "label": 1}
        assert merged[2] == clicks[2], "negative clicks must survive"

    def test_with_no_positive_the_correction_is_prepended(self):
        clicks = [{"x": 0.5, "y": 0.5, "label": 0}]
        merged = merge_corrected_positive_click(clicks, {"x": 0.1, "y": 0.1, "label": 1})
        assert len(merged) == 2 and merged[0]["label"] == 1

    def test_does_not_mutate_the_caller_list(self):
        clicks = [{"x": 0.1, "y": 0.1, "label": 1}]
        merge_corrected_positive_click(clicks, {"x": 0.9, "y": 0.9, "label": 1})
        assert clicks == [{"x": 0.1, "y": 0.1, "label": 1}]
