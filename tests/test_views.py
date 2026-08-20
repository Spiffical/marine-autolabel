"""Rendering the discovery view set, and its contract with the prompt."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from marine_autolabel.clickengine.discovery import build_content, discovery_views
from marine_autolabel.viz.views import extract_reference_frames, render_discovery_views

# A realistic frame size. The 10x10 grid draws a "row,col" label per cell; at
# small resolutions those labels are wider than the cells and tile the image,
# which is a rendering artefact of the test size, not of the pipeline (real
# frames are 1280x720).
H, W = 360, 640


@pytest.fixture
def frame():
    return np.full((H, W, 3), 100, dtype=np.uint8)


def mask_entry(y0, y1, x0, x1):
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return {"mask": m}


class TestViewSet:
    def test_produces_exactly_the_base_views(self, frame, tmp_path):
        views = render_discovery_views(frame, [], tmp_path)
        assert set(views) == {v.key for v in discovery_views(has_focus_crops=False)}

    def test_focus_region_adds_the_crop_views(self, frame, tmp_path):
        views = render_discovery_views(frame, [], tmp_path, focus_region=(0.0, 0.5, 0.0, 0.5))
        assert set(views) == {v.key for v in discovery_views(has_focus_crops=True)}

    def test_the_rendered_set_satisfies_the_prompt_assembler(self, frame, tmp_path):
        """The two modules must agree on which views exist."""
        for region in (None, (0.5, 1.0, 0.0, 1.0)):
            views = render_discovery_views(frame, [], tmp_path, focus_region=region)
            content = build_content(
                view_paths=views, neighbour_paths=[], pass_instruction=""
            )
            assert sum(1 for b in content if b["type"] == "image") == len(views)

    def test_every_view_is_actually_written(self, frame, tmp_path):
        views = render_discovery_views(frame, [mask_entry(40, 160, 40, 160)], tmp_path)
        for path in views.values():
            image = cv2.imread(path)
            assert image is not None and image.shape == (H, W, 3)

    def test_the_strong_view_is_more_opaque_than_the_light_one(self, frame, tmp_path):
        entry = mask_entry(40, 160, 40, 160)
        views = render_discovery_views(frame, [entry], tmp_path)
        interior = entry["mask"]
        light = cv2.imread(views["grid"])[..., 1][interior].mean()
        strong = cv2.imread(views["strong"])[..., 1][interior].mean()
        assert strong > light

    def test_the_outline_view_leaves_the_interior_raw(self, frame, tmp_path):
        """The point of the view: a colony seen through an accepted silhouette
        must stay visible, so the fill must not cover it.

        Measured in aggregate because every view also carries the coordinate
        grid and its cell labels, which legitimately mark some interior pixels.
        """
        entry = mask_entry(40, 240, 40, 400)
        views = render_discovery_views(frame, [entry], tmp_path)
        interior = entry["mask"]

        raw = cv2.imread(views["raw"])[..., 1][interior]
        outlined = cv2.imread(views["outline"])[..., 1][interior]
        filled = cv2.imread(views["strong"])[..., 1][interior]

        unchanged_outline = float((outlined == raw).mean())
        unchanged_filled = float((filled == raw).mean())
        # The remainder of the outline view's interior is the coordinate grid
        # and its labels, which every view carries by design.
        assert unchanged_outline > 0.50, "most of the interior must survive untouched"
        assert unchanged_filled < 0.05, "the filled view covers the interior"
        assert unchanged_outline > 10 * unchanged_filled

    def test_focus_crops_are_enlarged_to_full_frame(self, frame, tmp_path):
        views = render_discovery_views(frame, [], tmp_path, focus_region=(0.0, 0.25, 0.0, 0.25))
        assert cv2.imread(views["focus_raw"]).shape == (H, W, 3)

    def test_a_degenerate_focus_region_still_renders(self, frame, tmp_path):
        views = render_discovery_views(frame, [], tmp_path, focus_region=(0.0, 0.001, 0.0, 0.001))
        assert cv2.imread(views["focus_raw"]) is not None


class TestReferenceFrames:
    @pytest.fixture
    def video(self, tmp_path):
        path = tmp_path / "clip.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H)
        )
        for i in range(60):
            writer.write(np.full((H, W, 3), i * 4 % 255, dtype=np.uint8))
        writer.release()
        return path

    def test_extracts_frames_on_both_sides(self, video, tmp_path):
        paths = extract_reference_frames(video, 30, [10], tmp_path / "refs")
        assert len(paths) == 2

    def test_offsets_beyond_the_clip_are_skipped_not_errors(self, video, tmp_path):
        """A frame near the start legitimately has fewer neighbours."""
        paths = extract_reference_frames(video, 2, [10, 20], tmp_path / "refs")
        assert len(paths) < 4
        assert paths, "the forward neighbours should still be found"

    def test_no_offsets_yields_nothing(self, video, tmp_path):
        assert extract_reference_frames(video, 30, [], tmp_path / "refs") == []

    def test_a_missing_video_is_a_clear_error(self, tmp_path):
        with pytest.raises(RuntimeError, match="could not open video"):
            extract_reference_frames(tmp_path / "nope.mp4", 0, [1], tmp_path)
