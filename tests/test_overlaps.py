"""Selecting mask groups for identity review."""
from __future__ import annotations

import numpy as np
import pytest

from marine_autolabel.postprocess.overlaps import (
    ambiguous_overlap_components,
    duplicates_known_mask,
)

H = W = 200


def box(y0, y1, x0, x1):
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


class TestComponents:
    def test_distant_masks_are_not_flagged(self):
        components, pairs = ambiguous_overlap_components(
            [box(0, 20, 0, 20), box(150, 180, 150, 180)]
        )
        assert components == []
        assert pairs == []

    def test_a_contained_mask_is_flagged_as_overlap(self):
        big, small = box(20, 120, 20, 120), box(40, 60, 40, 60)
        components, pairs = ambiguous_overlap_components([big, small])
        assert components == [[0, 1]]
        assert pairs[0]["trigger"] == "overlap"
        assert pairs[0]["smaller_mask_overlap"] == 1.0

    def test_touching_masks_are_flagged_as_near_contact(self):
        left, right = box(20, 60, 20, 60), box(20, 60, 60, 100)
        components, pairs = ambiguous_overlap_components([left, right])
        assert components == [[0, 1]]
        assert pairs[0]["trigger"] == "near_contact"
        assert pairs[0]["smaller_mask_overlap"] == 0.0

    def test_the_gap_threshold_scales_with_the_frame_diagonal(self):
        """0.006 of the diagonal keeps the rule resolution independent."""
        adjacent = [box(20, 60, 20, 60), box(20, 60, 60, 100)]
        assert ambiguous_overlap_components(adjacent, max_gap_fraction=0.006)[0] == [[0, 1]]
        assert ambiguous_overlap_components(adjacent, max_gap_fraction=0.0)[0] == []

        # A two-pixel gap already exceeds 0.006 of a 200x200 diagonal (1.70 px).
        two_px_apart = [box(20, 60, 20, 60), box(20, 60, 61, 100)]
        assert ambiguous_overlap_components(two_px_apart)[0] == []

    def test_transitive_relationships_form_one_component(self):
        a, b, c = box(20, 60, 20, 60), box(20, 60, 60, 100), box(20, 60, 100, 140)
        components, _ = ambiguous_overlap_components([a, b, c])
        assert components == [[0, 1, 2]], "a-b and b-c must connect a to c"

    def test_separate_clusters_stay_separate(self):
        masks = [
            box(10, 40, 10, 40), box(10, 40, 40, 70),          # cluster one
            box(150, 180, 150, 180), box(150, 180, 180, 199),  # cluster two
        ]
        components, _ = ambiguous_overlap_components(masks)
        assert components == [[0, 1], [2, 3]]

    def test_unrelated_masks_are_absent_from_components(self):
        masks = [box(10, 40, 10, 40), box(10, 40, 40, 70), box(150, 190, 150, 190)]
        components, _ = ambiguous_overlap_components(masks)
        assert components == [[0, 1]]

    def test_pairs_carry_the_evidence_for_review(self):
        _, pairs = ambiguous_overlap_components([box(20, 120, 20, 120), box(40, 60, 40, 60)])
        assert set(pairs[0]) == {
            "left_index", "right_index", "smaller_mask_overlap", "iou", "gap_px", "trigger"
        }

    def test_containment_threshold_labels_the_trigger_it_does_not_gate_flagging(self):
        """Any two overlapping masks are flagged, whatever the threshold.

        Overlapping masks have a gap of 0, and 0 <= max_gap_px holds for every
        non-negative max_gap_fraction, so the near_contact arm always catches
        them. containment_threshold therefore only decides whether the pair is
        reported as "overlap" or "near_contact" -- useful evidence for the
        review stage, but not a filter.
        """
        big = box(0, 100, 0, 100)
        partial = box(90, 110, 0, 100)  # 50% of the smaller lies inside

        loose = ambiguous_overlap_components([big, partial], containment_threshold=0.4)
        assert loose[0] == [[0, 1]]
        assert loose[1][0]["trigger"] == "overlap"

        strict = ambiguous_overlap_components([big, partial], containment_threshold=0.9)
        assert strict[0] == [[0, 1]], "still flagged, only the label changes"
        assert strict[1][0]["trigger"] == "near_contact"
        assert strict[1][0]["gap_px"] == 0.0

    def test_empty_input(self):
        assert ambiguous_overlap_components([]) == ([], [])

    def test_a_single_mask_has_no_relationships(self):
        assert ambiguous_overlap_components([box(0, 10, 0, 10)]) == ([], [])

    def test_an_empty_mask_is_skipped_rather_than_crashing(self):
        """The original raised ValueError here: an empty mask has no nearest
        pixel, so distance_map[empty].min() reduces a zero-size array.

        Not reachable from today's call site, which passes accepted masks, but
        that is an unenforced invariant rather than a guarantee.
        """
        masks = [box(20, 60, 20, 60), np.zeros((H, W), dtype=bool), box(20, 60, 60, 100)]
        components, pairs = ambiguous_overlap_components(masks)
        assert components == [[0, 2]]
        assert all(1 not in (p["left_index"], p["right_index"]) for p in pairs)


class TestDuplicatesKnownMask:
    def test_a_near_identical_mask_is_a_duplicate(self):
        known = [{"mask": box(20, 60, 20, 60)}]
        assert duplicates_known_mask(box(20, 60, 21, 61), known)

    def test_a_contained_mask_is_a_duplicate_even_at_low_iou(self):
        known = [{"mask": box(20, 120, 20, 120)}]
        assert duplicates_known_mask(box(40, 60, 40, 60), known)

    def test_a_nearby_distinct_organism_is_not_a_duplicate(self):
        known = [{"mask": box(20, 60, 20, 60)}]
        assert not duplicates_known_mask(box(20, 60, 62, 100), known)

    def test_nothing_known_means_nothing_duplicated(self):
        assert not duplicates_known_mask(box(20, 60, 20, 60), [])

    @pytest.mark.parametrize("iou_threshold,expected", [(0.5, True), (0.99, False)])
    def test_iou_threshold_is_configurable(self, iou_threshold, expected):
        known = [{"mask": box(0, 100, 0, 100)}]
        candidate = box(0, 80, 0, 100)  # IoU 0.8, containment 1.0
        assert duplicates_known_mask(
            candidate, known, iou_threshold=iou_threshold, containment_threshold=1.01
        ) is expected
