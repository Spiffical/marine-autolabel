"""SAM3 click-mode service, driven by a fake backend (no CUDA, no weights).

Putting a protocol at this boundary is what makes the click engine testable at
all -- previously nothing below the model call could be exercised off-GPU.
"""
from __future__ import annotations

import numpy as np
import pytest

from marine_autolabel.sam3svc.service import Sam3PointService, valid_clicks

H = W = 100


def box(y0, y1, x0, x1):
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


class FakeSam3:
    """Returns a scripted (masks, scores) triple and records its prompts."""

    def __init__(self, masks=None, scores=None, raises=None):
        self.masks = masks
        self.scores = scores
        self.raises = raises
        self.calls: list[dict] = []
        self.set_image_calls = 0
        self.received_image = None

    def set_image(self, image):
        self.set_image_calls += 1
        self.received_image = image
        return {"state": self.set_image_calls}

    def predict_inst(self, state, **kwargs):
        self.calls.append(kwargs)
        if self.raises:
            raise self.raises
        return self.masks, self.scores, None


def service(**kw):
    return Sam3PointService(FakeSam3(**kw))


def group(gid=1, clicks=None, description="a fish"):
    return {
        "id": gid,
        "description": description,
        "clicks": clicks if clicks is not None else [{"x": 0.5, "y": 0.5, "label": 1}],
    }


IMAGE = np.zeros((H, W, 3), dtype=np.uint8)


class TestConstruction:
    def test_rejects_a_model_without_predict_inst(self):
        class NoInteractivity:
            def set_image(self, image):
                return None

        with pytest.raises(RuntimeError, match="enable_inst_interactivity"):
            Sam3PointService(NoInteractivity())


class TestValidClicks:
    def test_drops_malformed_entries(self):
        clicks = [
            {"x": 0.5, "y": 0.5, "label": 1},
            {"x": "nope", "y": 0.5, "label": 1},
            {"x": 0.5, "y": 0.5, "label": 7},
            {"x": 0.5, "label": 0},
        ]
        assert valid_clicks(clicks) == [{"x": 0.5, "y": 0.5, "label": 1}]

    def test_label_zero_is_valid(self):
        assert valid_clicks([{"x": 0.1, "y": 0.1, "label": 0}])


class TestGroupSegment:
    def test_clicks_are_scaled_from_normalised_to_pixels(self):
        svc = service(masks=np.stack([box(10, 30, 10, 30)]), scores=np.array([0.9]))
        svc.group_segment(IMAGE, [group(clicks=[{"x": 0.25, "y": 0.75, "label": 1}])])
        coords = svc.model.calls[0]["point_coords"]
        assert coords.tolist() == [[25.0, 75.0]]

    def test_labels_are_passed_through(self):
        svc = service(masks=np.stack([box(10, 30, 10, 30)]), scores=np.array([0.9]))
        clicks = [{"x": 0.2, "y": 0.2, "label": 1}, {"x": 0.8, "y": 0.8, "label": 0}]
        svc.group_segment(IMAGE, [group(clicks=clicks)])
        assert svc.model.calls[0]["point_labels"].tolist() == [1, 0]

    def test_the_image_is_embedded_once_for_many_groups(self):
        svc = service(masks=np.stack([box(10, 30, 10, 30)]), scores=np.array([0.9]))
        svc.group_segment(IMAGE, [group(1), group(2), group(3)])
        assert svc.model.set_image_calls == 1
        assert len(svc.model.calls) == 3

    def test_picks_the_smallest_in_band_candidate(self):
        masks = np.stack([box(0, 90, 0, 90), box(10, 30, 10, 30), box(10, 40, 10, 40)])
        svc = service(masks=masks, scores=np.array([0.99, 0.4, 0.7]))
        result = svc.group_segment(IMAGE, [group()])[0]
        assert result["select_reason"] == "smallest_in_band"
        assert result["area_px"] == 400
        assert result["score"] == pytest.approx(0.4)

    def test_batched_output_is_squeezed(self):
        masks = np.stack([box(10, 30, 10, 30)])[None, ...]  # (1, N, H, W)
        svc = service(masks=masks, scores=np.array([[0.8]]))
        result = svc.group_segment(IMAGE, [group()])[0]
        assert result["mask"].shape == (H, W)
        assert result["area_px"] == 400

    def test_result_carries_the_group_identity(self):
        svc = service(masks=np.stack([box(10, 30, 10, 30)]), scores=np.array([0.9]))
        result = svc.group_segment(IMAGE, [group(gid=7, description="a crab")])[0]
        assert result["creature_id"] == 7
        assert result["description"] == "a crab"
        assert result["spatial_match"] == "click_mode"


class TestGroupSegmentFailureModes:
    def test_group_with_no_valid_clicks_yields_an_empty_mask(self):
        svc = service(masks=np.stack([box(10, 30, 10, 30)]), scores=np.array([0.9]))
        result = svc.group_segment(IMAGE, [group(clicks=[{"x": None, "y": 1, "label": 1}])])[0]
        assert result["select_reason"] == "no_clicks"
        assert result["spatial_match"] == "click_mode_empty"
        assert result["area_px"] == 0
        assert not svc.model.calls, "a group with no clicks must not reach the model"

    def test_a_backend_exception_is_contained_to_its_group(self):
        svc = service(raises=RuntimeError("CUDA oom"))
        results = svc.group_segment(IMAGE, [group(1), group(2)])
        assert len(results) == 2
        assert all(r["spatial_match"] == "click_mode_error" for r in results)
        assert all(r["mask"].shape == (H, W) for r in results)

    def test_empty_prediction_is_reported_not_crashed(self):
        svc = service(masks=np.zeros((0, H, W), dtype=bool), scores=np.zeros(0))
        result = svc.group_segment(IMAGE, [group()])[0]
        assert result["spatial_match"] == "click_mode_empty"
        assert result["area_px"] == 0

    def test_one_failing_group_does_not_stop_the_others(self):
        """The first group raises; later groups still get processed."""
        class Flaky(FakeSam3):
            def predict_inst(self, state, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 1:
                    raise ValueError("transient")
                return np.stack([box(10, 30, 10, 30)]), np.array([0.9]), None

        svc = Sam3PointService(Flaky())
        results = svc.group_segment(IMAGE, [group(1), group(2)])
        assert results[0]["spatial_match"] == "click_mode_error"
        assert results[1]["spatial_match"] == "click_mode"


class TestRawPredict:
    def test_forwards_box_prompts(self):
        svc = service(masks=np.stack([box(5, 20, 5, 20)]), scores=np.array([0.7]))
        masks, scores = svc.raw_predict(IMAGE, box=np.array([1, 2, 3, 4]))
        assert masks.shape == (1, H, W)
        assert scores.shape == (1,)
        assert "box" in svc.model.calls[0]
        assert "point_coords" not in svc.model.calls[0]

    def test_scalar_score_is_promoted_to_an_array(self):
        svc = service(masks=np.stack([box(5, 20, 5, 20)]), scores=np.float32(0.5))
        _, scores = svc.raw_predict(IMAGE, point_coords=[[1, 1]], point_labels=[1])
        assert scores.shape == (1,)


class TestImageSize:
    """Production passes PIL images; tests pass arrays. Both must work.

    PIL's `.size` is (width, height); numpy's `.size` is an element count.
    """

    def test_pil_image_gives_the_right_orientation(self):
        from PIL import Image

        svc = service(masks=np.zeros((0, 40, 90), dtype=bool), scores=np.zeros(0))
        result = svc.group_segment(Image.new("RGB", (90, 40)), [group()])[0]
        assert result["mask"].shape == (40, 90), "PIL (w, h) must become (h, w)"

    def test_numpy_image_is_read_from_its_shape(self):
        svc = service(masks=np.zeros((0, 40, 90), dtype=bool), scores=np.zeros(0))
        result = svc.group_segment(np.zeros((40, 90, 3), np.uint8), [group()])[0]
        assert result["mask"].shape == (40, 90)

    def test_explicit_size_overrides_the_image(self):
        svc = service(masks=np.zeros((0, 5, 5), dtype=bool), scores=np.zeros(0))
        result = svc.group_segment(IMAGE, [group()], size_hw=(7, 11))[0]
        assert result["mask"].shape == (7, 11)

    def test_pixel_scaling_uses_the_real_frame_dimensions(self):
        from PIL import Image

        svc = service(masks=np.stack([np.ones((40, 90), dtype=bool)]), scores=np.array([0.5]))
        svc.group_segment(
            Image.new("RGB", (90, 40)), [group(clicks=[{"x": 0.5, "y": 0.5, "label": 1}])]
        )
        assert svc.model.calls[0]["point_coords"].tolist() == [[45.0, 20.0]]


class TestImageCoercion:
    """The SAM3 processor requires PIL RGB.

    Handing it an OpenCV BGR array produces masks of the wrong SHAPE as well as
    the wrong colour: a centre click on a 720x1280 frame returned (3, 1280, 3)
    instead of (3, 720, 1280), which then flows downstream as a plausible but
    meaningless mask. Found on the GPU box, not by any unit test.
    """

    def test_a_bgr_array_becomes_pil_rgb(self):
        from PIL import Image

        from marine_autolabel.sam3svc.service import as_pil_rgb

        bgr = np.zeros((4, 6, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255  # pure blue in BGR
        out = as_pil_rgb(bgr)
        assert isinstance(out, Image.Image)
        assert out.size == (6, 4), "PIL reports (width, height)"
        assert out.getpixel((0, 0)) == (0, 0, 255), "blue must land in the R,G,B blue slot"

    def test_a_pil_image_passes_through_as_rgb(self):
        from PIL import Image

        from marine_autolabel.sam3svc.service import as_pil_rgb

        out = as_pil_rgb(Image.new("L", (5, 3)))
        assert out.mode == "RGB" and out.size == (5, 3)

    def test_the_processor_receives_a_pil_image(self):
        from PIL import Image

        svc = service(masks=np.stack([box(10, 30, 10, 30)]), scores=np.array([0.9]))
        svc.group_segment(np.zeros((H, W, 3), np.uint8), [group()])
        assert isinstance(svc.model.received_image, Image.Image)
