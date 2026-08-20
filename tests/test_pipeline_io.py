"""Frame input loading and resume decisions."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from marine_autolabel.pipeline.loading import (
    load_firstpass,
    load_initial_masks,
    read_json,
    write_json,
)
from marine_autolabel.pipeline.resume import api_failed, is_frame_reusable

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def firstpass_dir(tmp_path):
    """A first-pass frame directory built from real captured output."""
    def build(frame="populated", model="claude-sonnet-5", errors=0):
        d = tmp_path / frame
        d.mkdir(exist_ok=True)
        shutil.copy(FIXTURES / "firstpass" / f"{frame}_frame_outputs_rle.json",
                    d / "frame_outputs_rle.json")
        write_json(d / "summary.json", {"model": model, "error_count": errors})
        return d
    return build


class TestLoadFirstpass:
    def test_loads_real_masks(self, firstpass_dir):
        masks, summary = load_firstpass(firstpass_dir(), "claude-sonnet-5")
        assert len(masks) == 2
        assert summary["model"] == "claude-sonnet-5"
        for entry in masks:
            assert entry["mask"].dtype == np.bool_
            assert entry["mask"].shape == (720, 1280)
            assert entry["source"] == "firstpass"

    def test_an_empty_first_pass_is_a_valid_result(self):
        """empty found nothing; the click engine found 12."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "04"
            d.mkdir()
            shutil.copy(
                FIXTURES / "firstpass" / "empty_frame_outputs_rle.json",
                d / "frame_outputs_rle.json",
            )
            write_json(d / "summary.json", {"model": "claude-sonnet-5", "error_count": 0})
            masks, _ = load_firstpass(d, "claude-sonnet-5")
        assert masks == []

    def test_a_model_mismatch_refuses_to_load(self, firstpass_dir):
        """Wrong model silently poisons every downstream number."""
        with pytest.raises(RuntimeError, match="model mismatch"):
            load_firstpass(firstpass_dir(), "claude-opus-5")

    def test_a_run_with_recorded_errors_refuses_to_load(self, firstpass_dir):
        with pytest.raises(RuntimeError, match="has errors"):
            load_firstpass(firstpass_dir(errors=3), "claude-sonnet-5")

    def test_probabilities_and_boxes_are_attached(self, firstpass_dir):
        masks, _ = load_firstpass(firstpass_dir(), "claude-sonnet-5")
        assert all(0.0 <= m["prob"] <= 1.0 for m in masks)
        assert all(len(m["box_xywh"]) == 4 for m in masks)

    def test_a_multi_frame_document_is_refused(self, firstpass_dir, tmp_path):
        d = firstpass_dir()
        doc = read_json(d / "frame_outputs_rle.json")
        doc["frames"] = doc["frames"] * 2
        write_json(d / "frame_outputs_rle.json", doc)
        with pytest.raises(RuntimeError, match="exactly one"):
            load_firstpass(d, "claude-sonnet-5")


class TestLoadInitialMasks:
    def test_no_root_means_no_masks(self):
        assert load_initial_masks(None, "any", (720, 1280)) == []

    def test_loads_the_final_mask_set(self, tmp_path):
        d = tmp_path / "frame"
        d.mkdir()
        shutil.copy(
            FIXTURES / "run" / "final_masks_rle.json",
            d / "final_masks_rle.json",
        )
        masks = load_initial_masks(tmp_path, "frame", (720, 1280))
        assert len(masks) == 6
        assert all(m["source"] == "initial" for m in masks)

    def test_falls_back_to_a_checkpoint(self, tmp_path):
        """An interrupted run leaves a checkpoint but no final set."""
        d = tmp_path / "frame"
        d.mkdir()
        shutil.copy(
            FIXTURES / "run" / "final_masks_rle.json",
            d / "checkpoint_masks_rle.json",
        )
        assert len(load_initial_masks(tmp_path, "frame", (720, 1280))) == 6

    def test_a_size_mismatch_is_refused(self, tmp_path):
        d = tmp_path / "frame"
        d.mkdir()
        shutil.copy(
            FIXTURES / "run" / "final_masks_rle.json",
            d / "final_masks_rle.json",
        )
        with pytest.raises(RuntimeError, match="size mismatch"):
            load_initial_masks(tmp_path, "frame", (480, 640))

    def test_a_missing_mask_file_is_a_clear_error(self, tmp_path):
        (tmp_path / "frame").mkdir()
        with pytest.raises(FileNotFoundError, match="checkpoint_masks_rle"):
            load_initial_masks(tmp_path, "frame", (720, 1280))


class TestResume:
    def test_a_clean_frame_has_not_failed(self, tmp_path):
        (tmp_path / "response.txt").write_text("a normal model reply")
        assert not api_failed(tmp_path)

    def test_the_failure_sentinel_is_detected(self, tmp_path):
        (tmp_path / "response.txt").write_text("model said <none>")
        assert api_failed(tmp_path)

    def test_nested_artefacts_are_scanned(self, tmp_path):
        nested = tmp_path / "pass_1" / "repair_round_2"
        nested.mkdir(parents=True)
        (nested / "verify.txt").write_text("<none>")
        assert api_failed(tmp_path)

    def test_non_text_artefacts_are_ignored(self, tmp_path):
        (tmp_path / "mask.json").write_text("<none>")
        assert not api_failed(tmp_path)

    def test_an_empty_directory_has_not_failed(self, tmp_path):
        assert not api_failed(tmp_path)

    def test_a_finished_clean_frame_is_reusable(self, tmp_path):
        (tmp_path / "final_masks_rle.json").write_text("{}")
        (tmp_path / "reply.txt").write_text("fine")
        assert is_frame_reusable(tmp_path)

    def test_a_finished_frame_with_a_failed_call_is_retried(self, tmp_path):
        """Caching it would bake the failure in permanently."""
        (tmp_path / "final_masks_rle.json").write_text("{}")
        (tmp_path / "reply.txt").write_text("<none>")
        assert not is_frame_reusable(tmp_path)

    def test_an_unfinished_frame_is_not_reusable(self, tmp_path):
        (tmp_path / "reply.txt").write_text("fine")
        assert not is_frame_reusable(tmp_path)

    def test_json_round_trip(self, tmp_path):
        path = tmp_path / "x.json"
        write_json(path, {"a": [1, 2], "b": "c"})
        assert read_json(path) == {"a": [1, 2], "b": "c"}
        assert json.loads(path.read_text())["a"] == [1, 2]
