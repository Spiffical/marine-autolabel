"""Run-level orchestration: selection, resume, containment, reporting."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from marine_autolabel.config import RunConfig
from marine_autolabel.pipeline.frame import FrameOutcome
from marine_autolabel.pipeline.graph import RESULT_NAME, load_manifest, run_manifest


def config(tmp_path, **kwargs):
    return RunConfig(
        benchmark_id="test", output_dir=tmp_path, manifest=Path("m.json"), **kwargs
    )


FRAMES = [{"id": "a"}, {"id": "b"}, {"id": "c"}]


def ok(record, frame_dir):
    (frame_dir / RESULT_NAME).write_text("{}")
    return FrameOutcome(frame_id=str(record["id"]), accepted=[{"mask": None}])


class TestSelection:
    def test_all_frames_run_by_default(self, tmp_path):
        seen = []
        run_manifest(
            FRAMES, config(tmp_path),
            process=lambda r, d: seen.append(r["id"]) or ok(r, d),
        )
        assert seen == ["a", "b", "c"]

    def test_frame_ids_restrict_the_run(self, tmp_path):
        seen = []
        run_manifest(
            FRAMES, config(tmp_path, frame_ids=("b",)),
            process=lambda r, d: seen.append(r["id"]) or ok(r, d),
        )
        assert seen == ["b"]

    def test_an_unknown_frame_id_selects_nothing(self, tmp_path):
        summary = run_manifest(FRAMES, config(tmp_path, frame_ids=("zz",)), process=ok)
        assert summary["n_frames"] == 0


class TestResume:
    def test_a_completed_clean_frame_is_reused(self, tmp_path):
        first = run_manifest(FRAMES, config(tmp_path), process=ok)
        assert first["n_done"] == 3

        seen = []
        second = run_manifest(
            FRAMES, config(tmp_path), process=lambda r, d: seen.append(r["id"]) or ok(r, d)
        )
        assert seen == [], "nothing should be recomputed"
        assert second["n_reused"] == 3

    def test_a_frame_with_a_failed_call_is_retried(self, tmp_path):
        """Its results are wrong, not merely incomplete."""
        run_manifest(FRAMES, config(tmp_path), process=ok)
        (tmp_path / "b" / "response.txt").write_text("<none>")

        seen = []
        summary = run_manifest(
            FRAMES, config(tmp_path), process=lambda r, d: seen.append(r["id"]) or ok(r, d)
        )
        assert seen == ["b"]
        assert summary["n_reused"] == 2 and summary["n_done"] == 1

    def test_resume_can_be_disabled(self, tmp_path):
        run_manifest(FRAMES, config(tmp_path), process=ok)
        seen = []
        run_manifest(
            FRAMES, config(tmp_path), resume=False,
            process=lambda r, d: seen.append(r["id"]) or ok(r, d),
        )
        assert seen == ["a", "b", "c"]

    def test_an_unfinished_frame_is_not_reused(self, tmp_path):
        def half_done(record, frame_dir):
            return FrameOutcome(frame_id=str(record["id"]))  # writes no result

        run_manifest(FRAMES, config(tmp_path), process=half_done)
        seen = []
        run_manifest(
            FRAMES, config(tmp_path), process=lambda r, d: seen.append(r["id"]) or ok(r, d)
        )
        assert seen == ["a", "b", "c"]


class TestFailureContainment:
    def test_one_failing_frame_does_not_end_the_run(self, tmp_path):
        def flaky(record, frame_dir):
            if record["id"] == "b":
                raise RuntimeError("CUDA out of memory")
            return ok(record, frame_dir)

        summary = run_manifest(FRAMES, config(tmp_path), process=flaky)
        assert summary["n_done"] == 2
        assert summary["n_failed"] == 1
        failed = next(f for f in summary["frames"] if f["frame_id"] == "b")
        assert "CUDA out of memory" in failed["error"]

    def test_a_failed_frame_is_retried_on_the_next_run(self, tmp_path):
        calls = []

        def flaky(record, frame_dir):
            calls.append(record["id"])
            if record["id"] == "b" and calls.count("b") == 1:
                raise RuntimeError("transient")
            return ok(record, frame_dir)

        run_manifest(FRAMES, config(tmp_path), process=flaky)
        summary = run_manifest(FRAMES, config(tmp_path), process=flaky)
        assert summary["n_done"] == 1 and summary["n_reused"] == 2


class TestReporting:
    def test_the_summary_is_written_and_carries_the_config(self, tmp_path):
        run_manifest(FRAMES, config(tmp_path), process=ok)
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["config"]["benchmark_id"] == "test"
        assert summary["n_frames"] == 3

    def test_a_screened_out_frame_counts_as_skipped(self, tmp_path):
        def screened(record, frame_dir):
            return FrameOutcome(frame_id=str(record["id"]), skipped_reason="all_black")

        summary = run_manifest(FRAMES, config(tmp_path), process=screened)
        assert summary["n_skipped"] == 3 and summary["n_done"] == 0

    def test_events_are_emitted_in_order(self, tmp_path):
        events = []
        run_manifest(
            FRAMES[:1], config(tmp_path), process=ok,
            on_event=lambda name, payload: events.append(name),
        )
        assert events == ["start", "done"]

    def test_a_reuse_emits_its_own_event(self, tmp_path):
        run_manifest(FRAMES[:1], config(tmp_path), process=ok)
        events = []
        run_manifest(
            FRAMES[:1], config(tmp_path), process=ok,
            on_event=lambda name, payload: events.append(name),
        )
        assert events == ["reused"]


class TestManifest:
    def test_loads_the_real_benchmark_manifest_shape(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"frames": [{"id": "a", "frame_index": 1}]}))
        assert load_manifest(path) == [{"id": "a", "frame_index": 1}]

    def test_accepts_the_samples_key_used_by_seatube_manifests(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"samples": [{"id": "s1"}]}))
        assert load_manifest(path) == [{"id": "s1"}]

    def test_a_manifest_without_records_is_refused(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"other": []}))
        with pytest.raises(ValueError, match="no 'frames' or 'samples'"):
            load_manifest(path)

    def test_records_without_an_id_are_named(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"frames": [{"id": "a"}, {"frame_index": 2}]}))
        with pytest.raises(ValueError, match=r"records at \[1\]"):
            load_manifest(path)


class TestRealManifests:
    """The shipped manifests must actually load, not just synthetic ones."""

    CONFIGS = Path(__file__).parent.parent / "configs"

    @pytest.mark.parametrize(
        "path", sorted((Path(__file__).parent.parent / "configs").glob("*.json")),
        ids=lambda p: p.name,
    )
    def test_every_shipped_manifest_loads(self, path):
        frames = load_manifest(path)
        assert frames
        assert all("id" in record for record in frames)

    def test_the_example_manifest_uses_placeholders_not_real_paths(self):
        text = (self.CONFIGS / "example_frames.json").read_text()
        assert "${MAL_VIDEO_ROOT}" in text

    def test_no_manifest_carries_a_machine_path(self):
        for path in self.CONFIGS.glob("*.json"):
            assert "/home/" not in path.read_text(), f"{path.name} carries an absolute path"
