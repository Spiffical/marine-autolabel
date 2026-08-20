"""Agent tool-use telemetry."""
from __future__ import annotations

import json

import pytest

from marine_autolabel.eval.telemetry import (
    describe,
    iter_json_objects,
    parse_history_file,
    parse_tool_calls,
    percentile,
    summarize_run,
)


def assistant(*tool_calls):
    text = "".join(f"<tool>{json.dumps(c)}</tool>" for c in tool_calls)
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def history(*messages):
    """Pretty-printed and concatenated -- the format the agent actually writes."""
    return "\n".join(json.dumps(m, indent=4) for m in messages)


def call(name, **params):
    return {"name": name, "parameters": params}


class TestStreamParsing:
    def test_reads_concatenated_pretty_printed_objects(self):
        """The debug history is NOT valid JSONL: objects span many lines."""
        text = history({"a": 1}, {"b": 2})
        assert iter_json_objects(text) == [{"a": 1}, {"b": 2}]

    def test_tolerates_junk_between_objects(self):
        text = json.dumps({"a": 1}) + "\n--- truncated ---\n" + json.dumps({"b": 2})
        assert iter_json_objects(text) == [{"a": 1}, {"b": 2}]

    def test_empty_input(self):
        assert iter_json_objects("") == []

    def test_non_objects_are_skipped(self):
        assert iter_json_objects('[1,2]\n{"a":1}') == [{"a": 1}]


class TestToolParsing:
    def test_counts_each_tool(self):
        text = history(
            assistant(call("segment_phrase", text_prompt="coral")),
            assistant(call("examine_each_mask"), call("drop_masks")),
        )
        out = parse_tool_calls(text)
        assert out["n_segment_phrase"] == 1
        assert out["n_examine_each_mask"] == 1
        assert out["n_drop_masks"] == 1
        assert out["n_tool_calls"] == 3

    def test_order_is_preserved(self):
        """Five segments then a verify tells a different story from alternating."""
        text = history(assistant(
            call("segment_phrase", text_prompt="a"),
            call("segment_phrase", text_prompt="b"),
            call("examine_each_mask"),
        ))
        assert parse_tool_calls(text)["tools_in_order"] == [
            "segment_phrase", "segment_phrase", "examine_each_mask"
        ]

    def test_phrases_are_collected(self):
        text = history(assistant(
            call("segment_phrase", text_prompt="coral"),
            call("segment_phrase", text_prompt="sea fan"),
        ))
        assert parse_tool_calls(text)["segment_phrases"] == ["coral", "sea fan"]

    def test_user_messages_are_ignored(self):
        text = history({"role": "user", "content": [
            {"type": "text", "text": '<tool>{"name": "segment_phrase"}</tool>'}
        ]})
        assert parse_tool_calls(text)["n_tool_calls"] == 0

    def test_a_malformed_tool_payload_is_skipped(self):
        text = json.dumps({"role": "assistant", "content": [
            {"type": "text", "text": "<tool>{not json}</tool>"}
        ], }, indent=4)
        assert parse_tool_calls(text)["n_tool_calls"] == 0

    def test_an_unknown_tool_is_recorded_but_not_counted(self):
        text = history(assistant(call("some_new_tool")))
        out = parse_tool_calls(text)
        assert out["tools_in_order"] == ["some_new_tool"]
        assert out["n_segment_phrase"] == 0

    def test_a_missing_history_file_is_empty_not_an_error(self, tmp_path):
        assert parse_history_file(tmp_path / "absent.json")["n_tool_calls"] == 0

    def test_reads_a_real_file(self, tmp_path):
        path = tmp_path / "debug_history.json"
        path.write_text(history(assistant(call("segment_phrase", text_prompt="coral"))))
        assert parse_history_file(path)["segment_phrases"] == ["coral"]


class TestStats:
    @pytest.mark.parametrize(
        "values,q,expected",
        [([1, 2, 3, 4], 0.0, 1.0), ([1, 2, 3, 4], 1.0, 4.0), ([1, 2, 3, 4], 0.5, 2.5)],
    )
    def test_percentile(self, values, q, expected):
        assert percentile(values, q) == pytest.approx(expected)

    def test_percentile_of_nothing_is_zero(self):
        assert percentile([], 0.9) == 0.0

    def test_describe_of_nothing_is_zeros_not_nan(self):
        out = describe([])
        assert all(v == 0.0 for v in out.values())

    def test_describe_reports_the_tail(self):
        out = describe([1, 1, 1, 1, 10])
        assert out["median"] == 1.0
        assert out["max"] == 10.0
        assert out["p90"] > out["median"], "p90 must expose the tail the mean hides"


class TestRunSummary:
    def test_totals_across_frames(self):
        frames = [
            parse_tool_calls(history(assistant(call("segment_phrase", text_prompt="a"),
                                               call("examine_each_mask")))),
            parse_tool_calls(history(assistant(call("segment_phrase", text_prompt="b")))),
        ]
        out = summarize_run(frames)
        assert out["n_frames"] == 2
        assert out["n_segment_phrase"] == 2
        assert out["n_examine_each_mask"] == 1

    def test_frames_that_segmented_without_verifying_are_flagged(self):
        """Those masks are unexamined, and nothing downstream says so."""
        frames = [
            parse_tool_calls(history(assistant(call("segment_phrase", text_prompt="a")))),
            parse_tool_calls(history(assistant(call("segment_phrase", text_prompt="b"),
                                               call("examine_each_mask")))),
        ]
        assert summarize_run(frames)["never_verified_frames"] == 1

    def test_a_frame_that_did_nothing_is_not_flagged(self):
        assert summarize_run([parse_tool_calls("")])["never_verified_frames"] == 0

    def test_distinct_phrases_are_deduplicated_and_sorted(self):
        frames = [
            parse_tool_calls(history(assistant(call("segment_phrase", text_prompt="coral")))),
            parse_tool_calls(history(assistant(call("segment_phrase", text_prompt="coral"),
                                               call("segment_phrase", text_prompt="anemone")))),
        ]
        assert summarize_run(frames)["distinct_phrases"] == ["anemone", "coral"]

    def test_an_empty_run(self):
        out = summarize_run([])
        assert out["n_frames"] == 0
        assert out["distinct_phrases"] == []
