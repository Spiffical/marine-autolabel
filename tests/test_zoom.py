"""Zoomed-crop segmentation and the never-abandon generator mode."""
from __future__ import annotations

import numpy as np
import pytest

from marine_autolabel.clickengine.generate import refine_group
from marine_autolabel.sam3svc.zoom import (
    clicks_to_crop_pixels,
    crop_and_upscale,
    masks_to_full_frame,
    predict_on_crop,
)

H, W = 200, 300
GEOM = (60, 40, 120, 80)  # left, top, crop_w, crop_h


def frame():
    return np.full((H, W, 3), 80, dtype=np.uint8)


class TestCropMechanics:
    def test_crop_is_upscaled(self):
        zoomed = crop_and_upscale(frame(), GEOM, 4)
        assert zoomed.shape == (80 * 4, 120 * 4, 3)

    def test_clicks_map_into_crop_pixels(self):
        clicks = [{"x": 60 / W, "y": 40 / H, "label": 1}]  # the crop's top-left
        coords, labels = clicks_to_crop_pixels(clicks, GEOM, 4, (H, W))
        assert coords.tolist() == [[0.0, 0.0]]
        assert labels.tolist() == [1]

    def test_a_centre_click_maps_to_the_crop_centre(self):
        clicks = [{"x": (60 + 60) / W, "y": (40 + 40) / H, "label": 1}]
        coords, _ = clicks_to_crop_pixels(clicks, GEOM, 2, (H, W))
        assert coords.tolist() == [[120.0, 80.0]]

    def test_masks_return_to_full_frame_position(self):
        crop_masks = np.zeros((1, 80 * 3, 120 * 3), dtype=bool)
        crop_masks[0, :, :] = True
        full = masks_to_full_frame(crop_masks, GEOM, (H, W))
        assert full.shape == (1, H, W)
        assert full[0, 40:120, 60:180].all()
        assert not full[0, 0:39, :].any(), "nothing outside the crop window"

    def test_fine_structure_survives_the_downscale(self):
        """INTER_AREA with a 0.5 threshold keeps thin branches that
        INTER_NEAREST erodes -- and thin branches are why zoom exists."""
        crop_masks = np.zeros((1, 80 * 4, 120 * 4), dtype=bool)
        crop_masks[0, :, 100:140] = True  # a 40px-wide stripe in crop space
        full = masks_to_full_frame(crop_masks, GEOM, (H, W))
        assert full[0].any()

    def test_an_empty_crop_mask_stays_empty(self):
        crop_masks = np.zeros((1, 80 * 2, 120 * 2), dtype=bool)
        assert not masks_to_full_frame(crop_masks, GEOM, (H, W)).any()

    def test_predict_on_crop_round_trips(self):
        seen = {}

        def predict(image, coords, labels):
            seen["shape"] = image.shape
            seen["coords"] = coords.tolist()
            masks = np.zeros((2, image.shape[0], image.shape[1]), dtype=bool)
            masks[0, 10:50, 10:50] = True
            masks[1] = True
            return masks, np.array([0.8, 0.3])

        clicks = [{"x": 0.4, "y": 0.4, "label": 1}]
        masks, scores = predict_on_crop(frame(), clicks, GEOM, 3, predict)
        assert seen["shape"] == (80 * 3, 120 * 3, 3)
        assert masks.shape == (2, H, W)
        assert scores.tolist() == [0.8, 0.3]
        assert masks[1, 40:120, 60:180].all(), "a full-crop mask fills the window"


def group():
    return {"id": 1, "description": "thin sea whip", "clicks": [{"x": 0.5, "y": 0.5, "label": 1}]}


def candidates(areas, scores):
    stack = np.zeros((len(areas), 100, 100), dtype=bool)
    for i, area in enumerate(areas):
        stack[i, : max(1, area // 100), :] = True
    return stack, np.array(scores, dtype=np.float32)


class TestNeverAbandon:
    """The zoom generator runs on a seed already verified to be a creature,
    so "there is nothing here" is not an answer it may give."""

    def test_an_abandon_verdict_still_returns_the_best_mask(self):
        result, _ = refine_group(
            group(),
            predict=lambda clicks: candidates((400,), (0.7,)),
            judge=lambda **kw: {"verdict": "abandon"},
            width=100, height=100, never_abandon=True,
        )
        assert result["status"] != "abandoned"
        assert result["area_px"] > 0

    def test_abandon_still_abandons_in_normal_mode(self):
        result, _ = refine_group(
            group(),
            predict=lambda clicks: candidates((400,), (0.7,)),
            judge=lambda **kw: {"verdict": "abandon"},
            width=100, height=100,
        )
        assert result["status"] == "abandoned"

    def test_oversized_candidates_still_yield_a_mask(self):
        result, trace = refine_group(
            group(),
            predict=lambda clicks: candidates((9000, 9500), (0.6, 0.4)),
            judge=lambda **kw: {"verdict": "good", "index": 0},
            width=100, height=100, never_abandon=True,
        )
        assert result["area_px"] > 0
        assert trace[0]["verdict"] == "no_valid_band"

    def test_exhaustion_returns_the_best_seen(self):
        result, trace = refine_group(
            group(),
            predict=lambda clicks: candidates((400,), (0.55,)),
            judge=lambda **kw: {"verdict": "reject"},
            width=100, height=100, max_attempts=2,
            strict_quality=True, never_abandon=True,
        )
        assert result["area_px"] > 0
        assert trace[-1]["verdict"] == "exhausted_keep_best"

    @pytest.mark.parametrize("never", [True, False])
    def test_a_good_verdict_behaves_the_same_either_way(self, never):
        result, _ = refine_group(
            group(),
            predict=lambda clicks: candidates((400,), (0.9,)),
            judge=lambda **kw: {"verdict": "good", "index": 0},
            width=100, height=100, never_abandon=never,
        )
        assert result["select_reason"] == "mllm_pick"
