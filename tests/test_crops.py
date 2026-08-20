"""Zoomed crop rendering."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from marine_autolabel.viz.crops import (
    default_upscale,
    render_binary_mask_crop,
    render_candidate_sheet,
    render_mask_crop,
)

H, W = 200, 300
GEOM = (50, 40, 100, 80)  # left, top, crop_w, crop_h


@pytest.fixture
def frame():
    return np.full((H, W, 3), 90, dtype=np.uint8)


def mask(y0=50, y1=90, x0=60, x1=120):
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


CLICKS = [{"x": 0.3, "y": 0.35, "label": 1}, {"x": 0.35, "y": 0.4, "label": 0}]


class TestUpscale:
    def test_small_crops_get_more_magnification(self):
        assert default_upscale(20, 20) > default_upscale(200, 200)

    def test_never_below_two(self):
        assert default_upscale(2000, 2000) == 2


class TestMaskCrop:
    def test_writes_a_zoomed_panel(self, frame, tmp_path):
        path = render_mask_crop(frame, mask(), CLICKS, GEOM, tmp_path / "c.png", 3)
        image = cv2.imread(path)
        assert image.shape == (80 * 3, 100 * 3, 3)

    def test_the_mask_region_is_tinted(self, frame, tmp_path):
        path = render_mask_crop(frame, mask(), CLICKS, GEOM, tmp_path / "c.png", 3)
        image = cv2.imread(path)
        assert image[..., 0].max() > 150, "cyan raises the blue channel"

    def test_the_input_frame_is_untouched(self, frame, tmp_path):
        before = frame.copy()
        render_mask_crop(frame, mask(), CLICKS, GEOM, tmp_path / "c.png", 3)
        assert np.array_equal(frame, before)


class TestBinaryCrop:
    def test_is_strictly_black_and_white(self, tmp_path):
        """Its whole purpose: an unambiguous statement of mask membership."""
        path = render_binary_mask_crop(mask(), GEOM, tmp_path / "b.png", 3)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        assert set(np.unique(image)).issubset({0, 255})

    def test_contains_both_inside_and_outside(self, tmp_path):
        path = render_binary_mask_crop(mask(), GEOM, tmp_path / "b.png", 3)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        assert 0 in np.unique(image) and 255 in np.unique(image)

    def test_an_empty_mask_is_all_black(self, tmp_path):
        path = render_binary_mask_crop(np.zeros((H, W), bool), GEOM, tmp_path / "b.png", 3)
        assert cv2.imread(path, cv2.IMREAD_GRAYSCALE).max() == 0


class TestCandidateSheet:
    def test_panels_are_overlay_above_binary(self, frame, tmp_path):
        masks = np.stack([mask(), mask(60, 100, 70, 130), mask(40, 120, 55, 140)])
        path = render_candidate_sheet(frame, masks, CLICKS, GEOM, tmp_path / "s.png", 3)
        image = cv2.imread(path)
        assert image.shape == (80 * 3 * 2, 100 * 3 * 3, 3)

    def test_one_candidate_still_renders(self, frame, tmp_path):
        masks = np.stack([mask()])
        path = render_candidate_sheet(frame, masks, CLICKS, GEOM, tmp_path / "s.png", 3)
        assert cv2.imread(path) is not None

    def test_the_binary_row_is_black_and_white(self, frame, tmp_path):
        masks = np.stack([mask(), mask(60, 100, 70, 130)])
        path = render_candidate_sheet(frame, masks, CLICKS, GEOM, tmp_path / "s.png", 3)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        bottom = image[image.shape[0] // 2 :]
        assert set(np.unique(bottom)).issubset({0, 255})

    def test_upscale_defaults_when_omitted(self, frame, tmp_path):
        masks = np.stack([mask()])
        path = render_candidate_sheet(frame, masks, CLICKS, GEOM, tmp_path / "s.png")
        assert cv2.imread(path) is not None
