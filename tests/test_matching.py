"""Matching click-engine masks to the first pass."""
from __future__ import annotations

import numpy as np

from marine_autolabel.postprocess.matching import match_firstpass

H = W = 100


def box(y0, y1, x0, x1):
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def click_result(mask, seed=None):
    out = {"mask": mask}
    if seed:
        out["seed_click"] = {"x": seed[0], "y": seed[1]}
    return out


class TestIouMatching:
    def test_a_clean_refind_is_matched(self):
        mask = box(10, 30, 10, 30)
        matches = match_firstpass([click_result(mask)], [{"mask": mask}])
        assert matches[0]["method"] == "iou"
        assert matches[0]["iou"] == 1.0

    def test_a_disjoint_mask_is_a_stray(self):
        matches = match_firstpass(
            [click_result(box(0, 10, 0, 10))], [{"mask": box(50, 60, 50, 60)}]
        )
        assert matches == {}

    def test_empty_masks_never_match(self):
        matches = match_firstpass(
            [click_result(np.zeros((H, W), bool))], [{"mask": box(10, 30, 10, 30)}]
        )
        assert matches == {}

    def test_each_firstpass_object_is_claimed_once(self):
        """Two similar clicks on one animal: the better one wins the match."""
        known = box(10, 30, 10, 30)
        results = [click_result(box(10, 30, 10, 28)), click_result(known)]
        matches = match_firstpass(results, [{"mask": known}])
        assert matches[1]["method"] == "iou"
        assert matches[1]["iou"] == 1.0
        assert matches.get(0, {}).get("method") != "iou"

    def test_highest_iou_pairs_are_taken_first(self):
        a, b = box(0, 20, 0, 20), box(60, 80, 60, 80)
        matches = match_firstpass(
            [click_result(b), click_result(a)], [{"mask": a}, {"mask": b}]
        )
        assert matches[0]["firstpass_index"] == 1
        assert matches[1]["firstpass_index"] == 0


class TestContainmentRescue:
    def test_a_part_of_a_known_animal_is_rescued_by_its_seed(self):
        """A fish's head has low union IoU with the whole fish but is inside it."""
        whole = box(10, 60, 10, 90)
        part = box(12, 20, 12, 22)
        matches = match_firstpass([click_result(part, seed=(0.16, 0.16))], [{"mask": whole}])
        assert matches[0]["method"] == "seed_inside_firstpass"
        assert matches[0]["iou"] < 0.5

    def test_high_overlap_rescues_even_without_a_seed(self):
        whole = box(10, 60, 10, 90)
        part = box(12, 20, 12, 22)
        matches = match_firstpass([click_result(part)], [{"mask": whole}])
        assert matches[0]["method"] == "mask_containment"
        assert matches[0]["overlap_coefficient"] >= 0.8

    def test_partial_overlap_below_the_threshold_stays_a_stray(self):
        known = box(0, 20, 0, 20)
        half_outside = box(10, 30, 0, 20)  # 50% overlap of the smaller
        matches = match_firstpass([click_result(half_outside)], [{"mask": known}])
        assert matches == {}

    def test_seed_outside_every_known_mask_stays_a_stray(self):
        matches = match_firstpass(
            [click_result(box(70, 90, 70, 90), seed=(0.8, 0.8))], [{"mask": box(0, 20, 0, 20)}]
        )
        assert matches == {}

    def test_rescue_reports_the_iou_it_actually_had(self):
        whole = box(10, 60, 10, 90)
        part = box(12, 20, 12, 22)
        matches = match_firstpass([click_result(part)], [{"mask": whole}])
        assert 0.0 < matches[0]["iou"] < 0.5


class TestThreshold:
    def test_threshold_selects_which_pass_makes_the_match(self):
        known = box(0, 40, 0, 100)
        contained = box(0, 30, 0, 100)  # IoU 0.75, fully inside `known`

        loose = match_firstpass([click_result(contained)], [{"mask": known}], 0.5)
        assert loose[0]["method"] == "iou"

        # Raising the threshold past the IoU does NOT turn this into a stray:
        # the containment rescue still claims it, because the smaller mask lies
        # wholly inside the larger one. Tightening the IoU threshold therefore
        # changes the match *method*, not whether a match happens.
        strict = match_firstpass([click_result(contained)], [{"mask": known}], 0.9)
        assert strict[0]["method"] == "mask_containment"
        assert strict[0]["overlap_coefficient"] == 1.0

    def test_threshold_does_turn_partial_overlaps_into_strays(self):
        """Where containment cannot rescue, the threshold is decisive."""
        known = box(0, 40, 0, 100)
        straddling = box(20, 60, 0, 100)  # overlap of the smaller = 0.5
        assert match_firstpass([click_result(straddling)], [{"mask": known}], 0.3)[0]
        assert match_firstpass([click_result(straddling)], [{"mask": known}], 0.9) == {}
