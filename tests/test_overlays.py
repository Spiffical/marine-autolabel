"""Overlays are the model's input, so what they make visible is behaviour."""
from __future__ import annotations

import numpy as np
import pytest

from marine_autolabel.viz.overlays import (
    render_existing_masks_overlay,
    render_grid_overlay,
    render_outline_overlay,
    render_proposed_click_groups_overlay,
)

H, W = 120, 160


@pytest.fixture
def frame():
    # Mid-grey so a green overlay is unambiguous in the green channel.
    return np.full((H, W, 3), 100, dtype=np.uint8)


def mask_entry(y0, y1, x0, x1):
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return {"mask": m}


class TestFilledOverlay:
    def test_masked_region_turns_green(self, frame):
        out = render_existing_masks_overlay(frame, [mask_entry(40, 60, 40, 60)])
        assert out[50, 50, 1] > frame[50, 50, 1], "green channel should rise"
        assert out[50, 50, 2] < frame[50, 50, 2], "red channel should fall"

    def test_unmasked_region_is_untouched(self, frame):
        out = render_existing_masks_overlay(frame, [mask_entry(40, 60, 40, 60)])
        assert np.array_equal(out[5, 5], frame[5, 5])

    def test_the_input_frame_is_not_mutated(self, frame):
        before = frame.copy()
        render_existing_masks_overlay(frame, [mask_entry(40, 60, 40, 60)])
        assert np.array_equal(frame, before)

    def test_higher_alpha_is_more_opaque(self, frame):
        faint = render_existing_masks_overlay(frame, [mask_entry(40, 60, 40, 60)], alpha=0.2)
        strong = render_existing_masks_overlay(frame, [mask_entry(40, 60, 40, 60)], alpha=0.8)
        assert strong[50, 50, 1] > faint[50, 50, 1]

    def test_empty_masks_are_skipped(self, frame):
        out = render_existing_masks_overlay(frame, [{"mask": np.zeros((H, W), bool)}])
        assert np.array_equal(out, frame)

    def test_no_masks_is_a_copy(self, frame):
        out = render_existing_masks_overlay(frame, [])
        assert np.array_equal(out, frame)
        assert out is not frame

    def test_shape_and_dtype_are_preserved(self, frame):
        out = render_existing_masks_overlay(frame, [mask_entry(40, 60, 40, 60)])
        assert out.shape == frame.shape and out.dtype == np.uint8


class TestOutlineOverlay:
    def test_the_interior_stays_raw(self, frame):
        """The whole point: a colony seen through an accepted silhouette
        must remain visible, so the fill must not cover it."""
        out = render_outline_overlay(frame, [mask_entry(30, 90, 30, 130)])
        assert np.array_equal(out[60, 80], frame[60, 80]), "centre must be untouched"

    def test_the_boundary_is_drawn(self, frame):
        out = render_outline_overlay(frame, [mask_entry(30, 90, 30, 130)])
        border = out[30, 30:130]
        assert (border[:, 1] > 200).any(), "top edge should carry green contour"

    def test_filled_and_outline_differ_exactly_at_the_interior(self, frame):
        masks = [mask_entry(30, 90, 30, 130)]
        filled = render_existing_masks_overlay(frame, masks)
        outlined = render_outline_overlay(frame, masks)
        assert filled[60, 80, 1] > frame[60, 80, 1]
        assert outlined[60, 80, 1] == frame[60, 80, 1]

    def test_empty_masks_are_skipped(self, frame):
        assert np.array_equal(render_outline_overlay(frame, [{"mask": np.zeros((H, W), bool)}]),
                              frame)


class TestGridOverlay:
    def test_shape_is_preserved(self, frame):
        assert render_grid_overlay(frame).shape == frame.shape

    def test_grid_lines_are_drawn(self, frame):
        out = render_grid_overlay(frame, grid_n=10)
        changed = (out != frame).any(axis=2)
        assert changed.any()

    def test_cell_count_changes_the_line_positions(self, frame):
        four = render_grid_overlay(frame, grid_n=4)
        ten = render_grid_overlay(frame, grid_n=10)
        assert not np.array_equal(four, ten)

    def test_lines_land_on_cell_boundaries(self, frame):
        out = render_grid_overlay(frame, grid_n=4)
        boundary = int(round(W / 4))
        column_changed = (out[:, boundary] != frame[:, boundary]).any()
        assert column_changed, "a vertical line belongs at width/4"

    def test_the_input_frame_is_not_mutated(self, frame):
        before = frame.copy()
        render_grid_overlay(frame)
        assert np.array_equal(frame, before)


class TestClickGroupsOverlay:
    def test_a_positive_click_is_drawn_near_its_location(self, frame):
        groups = [{"id": 1, "clicks": [{"x": 0.5, "y": 0.5, "label": 1}]}]
        out = render_proposed_click_groups_overlay(frame, groups)
        region = out[45:75, 60:100]
        assert (region != 100).any(), "a marker should appear around the click"

    def test_positive_and_negative_markers_differ(self, frame):
        pos = render_proposed_click_groups_overlay(
            frame, [{"id": 1, "clicks": [{"x": 0.5, "y": 0.5, "label": 1}]}]
        )
        neg = render_proposed_click_groups_overlay(
            frame, [{"id": 1, "clicks": [{"x": 0.5, "y": 0.5, "label": 0}]}]
        )
        assert not np.array_equal(pos, neg)

    def test_different_creature_ids_get_different_colours(self, frame):
        one = render_proposed_click_groups_overlay(
            frame, [{"id": 1, "clicks": [{"x": 0.5, "y": 0.5, "label": 1}]}]
        )
        two = render_proposed_click_groups_overlay(
            frame, [{"id": 2, "clicks": [{"x": 0.5, "y": 0.5, "label": 1}]}]
        )
        assert not np.array_equal(one, two)

    def test_the_palette_wraps_around(self, frame):
        first = render_proposed_click_groups_overlay(
            frame, [{"id": 1, "clicks": [{"x": 0.5, "y": 0.5, "label": 1}]}]
        )
        seventh = render_proposed_click_groups_overlay(
            frame, [{"id": 7, "clicks": [{"x": 0.5, "y": 0.5, "label": 1}]}]
        )
        # ids 1 and 7 share a colour, but their text labels differ.
        assert (first != seventh).any(), "labels '1.1' and '7.1' still differ"

    def test_existing_masks_are_shown_faintly_underneath(self, frame):
        groups = [{"id": 1, "clicks": [{"x": 0.9, "y": 0.9, "label": 1}]}]
        masks = [mask_entry(10, 30, 10, 30)]
        with_masks = render_proposed_click_groups_overlay(frame, groups, masks)
        assert with_masks[20, 20, 1] > frame[20, 20, 1]

    def test_no_groups_leaves_the_frame_alone(self, frame):
        assert np.array_equal(render_proposed_click_groups_overlay(frame, []), frame)

    def test_clicks_at_the_frame_edge_do_not_crash(self, frame):
        groups = [{"id": 1, "clicks": [
            {"x": 0.0, "y": 0.0, "label": 1}, {"x": 1.0, "y": 1.0, "label": 0}
        ]}]
        out = render_proposed_click_groups_overlay(frame, groups)
        assert out.shape == frame.shape
