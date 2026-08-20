"""Message conversion: OpenAI-ish agent output -> Anthropic content blocks."""
from __future__ import annotations

import base64

import pytest
from PIL import Image

from marine_autolabel.llm.images import cap_images, image_to_base64
from marine_autolabel.llm.messages import (
    convert_block,
    flatten_system_content,
    normalize_content,
    to_anthropic,
)


@pytest.fixture
def png(tmp_path):
    path = tmp_path / "frame.png"
    Image.new("RGB", (40, 20), (10, 200, 10)).save(path)
    return str(path)


class TestSystemContent:
    def test_plain_string_passes_through(self):
        assert flatten_system_content("be exhaustive") == "be exhaustive"

    def test_block_list_is_joined(self):
        content = [{"type": "text", "text": "a"}, "b", {"type": "image", "image": "x.png"}]
        assert flatten_system_content(content) == "a\nb"

    def test_none_becomes_empty(self):
        assert flatten_system_content(None) == ""

    def test_multiple_system_messages_are_concatenated(self):
        system, _ = to_anthropic(
            [
                {"role": "system", "content": "first"},
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "second"},
            ],
            image_max_edge=None,
            max_images=None,
        )
        assert system == "first\n\nsecond"


class TestNormalize:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("hi", [{"type": "text", "text": "hi"}]),
            (None, []),
            ([{"type": "text", "text": "x"}], [{"type": "text", "text": "x"}]),
        ],
    )
    def test_shapes(self, value, expected):
        assert normalize_content(value) == expected


class TestConvertBlock:
    def test_whitespace_only_text_is_dropped(self):
        assert convert_block({"type": "text", "text": "   "}, image_max_edge=None) is None

    def test_bare_string_becomes_text(self):
        assert convert_block("hello", image_max_edge=None) == {"type": "text", "text": "hello"}

    def test_image_path_becomes_base64_source(self, png):
        block = convert_block({"type": "image", "image": png}, image_max_edge=None)
        assert block["type"] == "image"
        assert block["source"]["type"] == "base64"
        assert block["source"]["media_type"] == "image/png"
        base64.b64decode(block["source"]["data"])  # must be valid base64

    def test_missing_image_file_is_dropped_not_raised(self, tmp_path):
        assert convert_block(
            {"type": "image", "image": str(tmp_path / "nope.png")}, image_max_edge=None
        ) is None

    def test_data_uri_image_url_is_unwrapped(self):
        url = "data:image/jpeg;base64,QUJD"
        block = convert_block({"type": "image_url", "image_url": {"url": url}}, image_max_edge=None)
        assert block["source"] == {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"}

    def test_remote_image_url_is_refused(self):
        block = {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
        assert convert_block(block, image_max_edge=None) is None

    def test_unknown_block_type_is_dropped(self):
        assert convert_block({"type": "video", "video": "v.mp4"}, image_max_edge=None) is None


class TestDownscale:
    def test_max_edge_reencodes_as_jpeg(self, tmp_path):
        path = tmp_path / "big.png"
        Image.new("RGB", (2000, 1000)).save(path)
        data, mime = image_to_base64(str(path), max_edge=256)
        assert mime == "image/jpeg"
        assert data is not None

    def test_small_image_keeps_its_own_type(self, png):
        _, mime = image_to_base64(png, max_edge=4096)
        assert mime == "image/png"


class TestImageCap:
    @staticmethod
    def _messages(n):
        return [
            {"role": "user", "content": [{"type": "image", "image": f"{i}.png"}]} for i in range(n)
        ]

    def test_keeps_the_most_recent(self):
        capped = cap_images(self._messages(5), 2)
        kept = [p["image"] for m in capped for p in m["content"]]
        assert kept == ["3.png", "4.png"]

    def test_no_cap_is_a_passthrough(self):
        msgs = self._messages(5)
        assert cap_images(msgs, None) is msgs
        assert cap_images(msgs, 0) is msgs

    def test_text_survives_the_cap(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "keep me"},
                                         {"type": "image", "image": "0.png"}]},
            {"role": "user", "content": [{"type": "image", "image": "1.png"}]},
        ]
        capped = cap_images(msgs, 1)
        assert capped[0]["content"] == [{"type": "text", "text": "keep me"}]
        assert capped[1]["content"] == [{"type": "image", "image": "1.png"}]

    def test_assistant_images_are_not_counted_or_dropped(self):
        msgs = [
            {"role": "assistant", "content": [{"type": "image", "image": "a.png"}]},
            {"role": "user", "content": [{"type": "image", "image": "u.png"}]},
        ]
        assert cap_images(msgs, 1) == msgs

    def test_cap_runs_before_base64_so_dropped_images_are_never_read(self, png, tmp_path):
        """A dropped image must not be opened -- missing files must not even warn."""
        missing = str(tmp_path / "absent.png")
        _, messages = to_anthropic(
            [
                {"role": "user", "content": [{"type": "image", "image": missing}]},
                {"role": "user", "content": [{"type": "image", "image": png}]},
            ],
            image_max_edge=None,
            max_images=1,
        )
        assert len(messages) == 1
        assert messages[0]["content"][0]["source"]["media_type"] == "image/png"


class TestToAnthropic:
    def test_system_is_split_out_of_the_message_list(self, png):
        system, messages = to_anthropic(
            [
                {"role": "system", "content": "find every organism"},
                {"role": "user", "content": "here is the frame"},
            ],
            image_max_edge=None,
            max_images=None,
        )
        assert system == "find every organism"
        assert [m["role"] for m in messages] == ["user"]

    def test_messages_left_with_no_blocks_are_removed(self, tmp_path):
        _, messages = to_anthropic(
            [
                {"role": "user", "content": [{"type": "image", "image": str(tmp_path / "x.png")}]},
                {"role": "user", "content": "still here"},
            ],
            image_max_edge=None,
            max_images=None,
        )
        assert len(messages) == 1
        assert messages[0]["content"] == [{"type": "text", "text": "still here"}]

    def test_unknown_roles_are_ignored(self):
        _, messages = to_anthropic(
            [{"role": "tool", "content": "ignored"}, {"role": "user", "content": "kept"}],
            image_max_edge=None,
            max_images=None,
        )
        assert len(messages) == 1

    def test_question_mark_in_path_is_escaped(self, tmp_path):
        path = tmp_path / "f%3Frame.png"
        Image.new("RGB", (4, 4)).save(path)
        block = convert_block(
            {"type": "image", "image": str(tmp_path / "f?rame.png")}, image_max_edge=None
        )
        assert block is not None, "'?' should be escaped to %3F before opening"
