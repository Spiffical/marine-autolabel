"""Cheap geometric rejection before anything reaches the model."""
from __future__ import annotations

import numpy as np
import pytest

from marine_autolabel.clickengine.candidates import (
    count_significant_components,
    filter_candidates,
    filter_candidates_with_reasons,
    touches_edges,
)

H = W = 100
FILTER = {"iou_dedup": 0.5, "min_area_px": 100, "max_area_frac": 0.5, "edge_tol_px": 2}


def box(y0, y1, x0, x1):
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def candidate(mask):
    return {"mask": mask}


def reasons(cands, existing=(), **overrides):
    opts = {**FILTER, **overrides}
    return [r for _, r in filter_candidates_with_reasons(cands, list(existing), **opts)]


class TestTouchesEdges:
    def test_counts_each_edge_once(self):
        assert touches_edges(box(0, 5, 40, 60), 2) == 1        # top only
        assert touches_edges(box(0, H, 40, 60), 2) == 2        # top and bottom
        assert touches_edges(np.ones((H, W), bool), 2) == 4

    def test_interior_mask_touches_nothing(self):
        assert touches_edges(box(40, 60, 40, 60), 2) == 0

    def test_zero_tolerance_disables_the_test(self):
        assert touches_edges(np.ones((H, W), bool), 0) == 0


class TestComponents:
    def test_one_blob_is_one_component(self):
        assert count_significant_components(box(20, 40, 20, 40)) == 1

    def test_two_similar_blobs_count_separately(self):
        mask = box(10, 30, 10, 30) | box(60, 80, 60, 80)
        assert count_significant_components(mask) == 2

    def test_a_speck_beside_a_blob_does_not_count(self):
        mask = box(10, 50, 10, 50) | box(90, 92, 90, 92)
        assert count_significant_components(mask, min_component_frac=0.15) == 1

    def test_empty_mask_has_no_components(self):
        assert count_significant_components(np.zeros((H, W), bool)) == 0


class TestFilter:
    def test_a_clean_candidate_survives(self):
        assert reasons([candidate(box(40, 60, 40, 60))]) == [None]

    def test_too_small_is_rejected(self):
        assert reasons([candidate(box(40, 45, 40, 45))]) == ["too_small"]

    def test_too_large_is_rejected(self):
        assert reasons([candidate(box(0, 80, 0, 80))]) == ["too_large"]

    def test_a_mask_spanning_two_edges_is_rejected(self):
        assert reasons([candidate(box(0, H, 40, 60))]) == ["multi_edge_clipped"]

    def test_touching_one_edge_is_allowed(self):
        """Organisms legitimately enter frame from a single side."""
        assert reasons([candidate(box(0, 30, 40, 60))]) == [None]

    def test_a_duplicate_of_an_accepted_mask_is_rejected(self):
        known = box(40, 60, 40, 60)
        assert reasons([candidate(known)], existing=[{"mask": known}]) == [
            "duplicate_of_existing"
        ]

    def test_a_distinct_mask_beside_an_accepted_one_survives(self):
        assert reasons(
            [candidate(box(40, 60, 40, 60))], existing=[{"mask": box(10, 25, 10, 25)}]
        ) == [None]

    def test_a_multi_object_blob_is_rejected(self):
        blob = box(10, 30, 10, 30) | box(60, 80, 60, 80)
        assert reasons([candidate(blob)]) == ["multi_component_blob"]

    def test_rejection_order_size_precedes_everything(self):
        """A tiny mask spanning two edges reports size, not edges."""
        tiny_spanning = np.zeros((H, W), dtype=bool)
        tiny_spanning[0, 0] = True
        tiny_spanning[H - 1, 0] = True
        assert reasons([candidate(tiny_spanning)]) == ["too_small"]

    def test_every_candidate_gets_a_verdict(self):
        cands = [candidate(box(40, 60, 40, 60)), candidate(box(40, 45, 40, 45))]
        assert len(filter_candidates_with_reasons(cands, [], **FILTER)) == 2

    def test_filter_candidates_returns_only_survivors(self):
        cands = [candidate(box(40, 60, 40, 60)), candidate(box(40, 45, 40, 45))]
        assert len(filter_candidates(cands, [], **FILTER)) == 1

    @pytest.mark.parametrize("iou_dedup,expected", [(0.9, None), (0.1, "duplicate_of_existing")])
    def test_dedup_threshold_is_configurable(self, iou_dedup, expected):
        known = box(40, 60, 40, 60)
        overlapping = box(42, 62, 42, 62)
        assert reasons(
            [candidate(overlapping)], existing=[{"mask": known}], iou_dedup=iou_dedup
        ) == [expected]
