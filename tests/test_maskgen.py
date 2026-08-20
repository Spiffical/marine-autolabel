"""Crop geometry and the hybrid mask-generation policy."""
from __future__ import annotations

import numpy as np
import pytest

from marine_autolabel.clickengine.crop import mask_crop_geom
from marine_autolabel.clickengine.maskgen import hybrid_policy, is_degenerate, mask_area

W, H = 1280, 720


def result(area_px, status="ok", reason="smallest_in_band"):
    mask = np.zeros((H, W), dtype=bool)
    if area_px:
        side = int(area_px**0.5)
        mask[:side, : max(1, area_px // max(1, side))] = True
    return {"mask": mask, "status": status, "select_reason": reason}


class TestCropGeom:
    def test_window_is_centred_on_the_mask(self):
        mask = np.zeros((H, W), dtype=bool)
        mask[300:340, 600:660] = True
        left, top, cw, ch = mask_crop_geom(mask, [], W, H, 0.10)
        assert left <= 630 <= left + cw
        assert top <= 320 <= top + ch

    def test_window_never_falls_below_region_frac(self):
        mask = np.zeros((H, W), dtype=bool)
        mask[100:102, 100:102] = True  # a 2x2 speck
        _, _, cw, ch = mask_crop_geom(mask, [], W, H, 0.25)
        assert cw == int(W * 0.25)
        assert ch == int(H * 0.25)

    def test_large_mask_gets_padding_beyond_its_bounding_box(self):
        mask = np.zeros((H, W), dtype=bool)
        mask[200:400, 400:800] = True
        _, _, cw, _ = mask_crop_geom(mask, [], W, H, 0.10)
        # bbox width is xs.max() - xs.min() = 799 - 400 = 399, padded 1.8x
        assert cw == int(399 * 1.8)
        assert cw > int(W * 0.10), "padding must beat the region_frac floor here"

    def test_window_is_clamped_inside_the_frame(self):
        mask = np.zeros((H, W), dtype=bool)
        mask[0:5, 0:5] = True  # hard against the corner
        left, top, cw, ch = mask_crop_geom(mask, [], W, H, 0.30)
        assert left >= 0 and top >= 0
        assert left + cw <= W and top + ch <= H

    def test_window_never_exceeds_the_frame(self):
        mask = np.ones((H, W), dtype=bool)
        left, top, cw, ch = mask_crop_geom(mask, [], W, H, 1.0)
        assert (left, top, cw, ch) == (0, 0, W, H)

    def test_empty_mask_centres_on_the_mean_click(self):
        clicks = [{"x": 0.2, "y": 0.4}, {"x": 0.4, "y": 0.6}]
        left, top, cw, ch = mask_crop_geom(np.zeros((H, W), bool), clicks, W, H, 0.20)
        assert left <= 0.3 * W <= left + cw
        assert top <= 0.5 * H <= top + ch

    def test_empty_mask_and_no_clicks_centres_on_the_frame(self):
        left, top, cw, ch = mask_crop_geom(np.zeros((H, W), bool), [], W, H, 0.20)
        assert left + cw // 2 == pytest.approx(W // 2, abs=2)
        assert top + ch // 2 == pytest.approx(H // 2, abs=2)


class TestHybridPolicy:
    def test_a_good_full_frame_result_skips_the_zoom_entirely(self):
        zoom_called = []

        def zoom():
            zoom_called.append(True)
            return result(5000), []

        out, trace = hybrid_policy(lambda: (result(5000), [{"step": "ff"}]), zoom)
        assert not zoom_called, "zoom must not run when full-frame succeeded"
        assert trace == [{"step": "ff"}]
        assert not out["select_reason"].startswith("hybrid_zoom/")

    def test_abandoned_full_frame_triggers_the_zoom(self):
        out, trace = hybrid_policy(
            lambda: (result(0, status="abandoned"), []),
            lambda: (result(5000), [{"step": "zoom"}]),
        )
        assert out["select_reason"] == "hybrid_zoom/smallest_in_band"
        assert {"hybrid": "zoom_recover", "fullframe_area": 0} in trace
        assert {"step": "zoom"} in trace

    def test_degenerate_area_triggers_the_zoom(self):
        out, _ = hybrid_policy(
            lambda: (result(50), []), lambda: (result(5000), []), min_area_px=200
        )
        assert out["select_reason"].startswith("hybrid_zoom/")

    def test_a_degenerate_zoom_result_is_rejected_and_full_frame_is_kept(self):
        """A bad zoom is no better; keep the more informative full-frame trace."""
        full = result(10, status="abandoned")
        out, trace = hybrid_policy(
            lambda: (full, [{"step": "ff"}]), lambda: (result(20), [{"step": "zoom"}])
        )
        assert out is full
        assert trace == [{"step": "ff"}]

    def test_trace_order_is_fullframe_then_marker_then_zoom(self):
        _, trace = hybrid_policy(
            lambda: (result(0, status="abandoned"), [{"n": 1}]),
            lambda: (result(9000), [{"n": 3}]),
        )
        assert [t.get("n", "marker") for t in trace] == [1, "marker", 3]

    def test_min_area_threshold_is_respected(self):
        out, _ = hybrid_policy(
            lambda: (result(300), []), lambda: (result(9000), []), min_area_px=200
        )
        assert not out["select_reason"].startswith("hybrid_zoom/"), "300 >= 200 is not degenerate"


class TestDegeneracy:
    def test_abandoned_status_is_degenerate_whatever_the_area(self):
        assert is_degenerate(result(9000, status="abandoned"))

    def test_small_area_is_degenerate(self):
        assert is_degenerate(result(10))

    def test_a_healthy_mask_is_not(self):
        assert not is_degenerate(result(5000))

    def test_mask_area_counts_true_pixels(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[0:2, 0:3] = True
        assert mask_area({"mask": mask}) == 6
